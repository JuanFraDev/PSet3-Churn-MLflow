"""Features de churn y preprocesamiento reutilizable sin fuga de datos.

Los estimadores deben ajustarse mediante .fit SOLO en train o en el fold de CV.
Este archivo no entrena ni modifica la OBT de Snowflake.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICIOS = [
    "ONLINE_SECURITY", "ONLINE_BACKUP", "DEVICE_PROTECTION",
    "TECH_SUPPORT", "STREAMING_TV", "STREAMING_MOVIES",
]

NUMERICAS = [
    "SENIOR_CITIZEN", "TENURE_MONTHS", "MONTHLY_CHARGES",
    "TOTAL_CHARGES", "TOTAL_CHARGES_MISSING",
    "MONTHLY_NEW_CUSTOMER", "SERVICE_BUNDLE_COUNT",
]

CATEGORICAS = [
    "GENDER", "PARTNER", "DEPENDENTS", "CONTRACT",
    "PAPERLESS_BILLING", "PAYMENT_METHOD", "PHONE_SERVICE",
    "MULTIPLE_LINES", "INTERNET_SERVICE", "ONLINE_SECURITY",
    "ONLINE_BACKUP", "DEVICE_PROTECTION", "TECH_SUPPORT",
    "STREAMING_TV", "STREAMING_MOVIES",
]


class FeaturesDerivadas(BaseEstimator, TransformerMixin):
    """Construye indicadores exclusivamente con variables conocidas en el snapshot.

    Nota: la existencia de la información ANTES del abandono en producción debe
    verificarse contra el CRM; este dataset transversal no trae timestamps de evento.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        faltantes = {"CONTRACT", "TENURE_MONTHS", *SERVICIOS} - set(X.columns)
        if faltantes:
            raise ValueError(f"Faltan columnas necesarias: {sorted(faltantes)}")
        X = X.copy()
        X["MONTHLY_NEW_CUSTOMER"] = (
            X["CONTRACT"].eq("Month-to-month") & (X["TENURE_MONTHS"] <= 12)
        ).astype(int)
        X["SERVICE_BUNDLE_COUNT"] = X[SERVICIOS].eq("Yes").sum(axis=1).astype(int)
        # Las dos banderas de ausentes son idénticas en el snapshot observado.
        # Se conserva TOTAL_CHARGES_MISSING y se elimina la copia redundante.
        X = X.drop(columns=["NEW_CUSTOMER_WITHOUT_CHARGES"], errors="ignore")
        return X


def crear_preprocesador():
    """Retorna pasos sin ajustar; imputación/encoding/escalado ocurren en el fit del Pipeline."""
    numerico = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    categorico = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[("num", numerico, NUMERICAS), ("cat", categorico, CATEGORICAS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def crear_pipeline(modelo):
    """Fase 3+: feature engineering y preparación dentro de la CV, antes del estimador."""
    return Pipeline([
        ("features", FeaturesDerivadas()),
        ("preparacion", crear_preprocesador()),
        ("modelo", modelo),
    ])
