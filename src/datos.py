"""Acceso reproducible a la OBT de Snowflake mediante el servicio RSA de PSet #2.
No imprime credenciales ni escribe claves privadas a disco.
"""

import base64
import hashlib
import os

import pandas as pd
import snowflake.connector
from cryptography.hazmat.primitives import serialization

ORIGEN = "PSET2_CHURN.OBT.CUSTOMER_CHURN_OBT"
NOMBRE_TARGET = "CHURN_FLAG"
EXCLUIR_PREDICTORES = ["CUSTOMER_ID", "INGESTED_AT", NOMBRE_TARGET]


def _obligatoria(nombre: str) -> str:
    valor = os.getenv(nombre)
    if not valor:
        raise RuntimeError(f"Falta variable de entorno {nombre}. Revisa .env y .env.snowflake")
    return valor.strip()


def conectar_snowflake():
    pem = base64.b64decode(_obligatoria("SECRET_SNOWFLAKE_PRIVATE_KEY"), validate=True)
    clave_password = base64.b64decode(
        _obligatoria("SECRET_SNOWFLAKE_PRIVATE_KEY_PASSWORD"), validate=True
    )
    private_key = serialization.load_pem_private_key(pem, password=clave_password)
    private_der = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return snowflake.connector.connect(
        account=_obligatoria("SNOWFLAKE_ACCOUNT"),
        user="PSET2_KESTRA_SVC",
        role="PSET2_SPARK_ROLE",
        warehouse="PSET2_WH",
        database="PSET2_CHURN",
        schema="OBT",
        private_key=private_der,
        session_parameters={"QUERY_TAG": "PSET3_EDA"},
    )


def leer_obt() -> pd.DataFrame:
    """Lee una sola vez el corte transversal; no inventa historial de clientes."""
    conn = conectar_snowflake()
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {ORIGEN}")
            columnas = [c[0] for c in cursor.description]
            filas = cursor.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame.from_records(filas, columns=columnas)
    for col in ["TENURE_MONTHS", "MONTHLY_CHARGES", "TOTAL_CHARGES",
                "TOTAL_CHARGES_MISSING", "NEW_CUSTOMER_WITHOUT_CHARGES",
                "CHURN_FLAG"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.empty:
        raise RuntimeError(f"La tabla {ORIGEN} no contiene registros")
    return df


def hash_snapshot(df: pd.DataFrame) -> str:
    """Huella de contenido determinista: ignora INGESTED_AT, que es solo auditoria."""
    estable = df.drop(columns=["INGESTED_AT"], errors="ignore")
    estable = estable.reindex(sorted(estable.columns), axis=1)
    if "CUSTOMER_ID" in estable.columns:
        estable = estable.sort_values("CUSTOMER_ID", kind="stable")
    bytes_csv = estable.to_csv(index=False, na_rep="<NULL>", float_format="%.9g").encode("utf-8")
    return hashlib.sha256(bytes_csv).hexdigest()


def variables_modelo(df: pd.DataFrame):
    """Separa predictores del target; sin ID ni fecha tecnica."""
    if NOMBRE_TARGET not in df:
        raise ValueError(f"No se encontro {NOMBRE_TARGET} en el origen")
    X = df.drop(columns=EXCLUIR_PREDICTORES, errors="ignore")
    y = df[NOMBRE_TARGET].astype(int)
    if not set(y.dropna().unique()).issubset({0, 1}):
        raise ValueError("CHURN_FLAG tiene valores distintos de 0 y 1")
    return X, y
