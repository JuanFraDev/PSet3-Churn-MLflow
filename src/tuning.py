"""Búsqueda aleatoria 30x5 y tracking reproducible de configuraciones en MLflow.

No utiliza hold-out: todo entrenamiento y selección sucede en X_train / y_train.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

from src.pipeline import crear_pipeline

N_CONFIG = 30
N_FOLDS = 5
SEED = 42
EXPERIMENTO = "PSet3_Churn_Telco"


def definiciones_modelos(seed=SEED):
    """Distribuciones explícitas. Valores continuos en escala logarítmica cuando procede."""
    return {
        "Candidato_Logistica": {
            "familia": "lineal_regularizado",
            "estimador": crear_pipeline(LogisticRegression(max_iter=1000, random_state=seed)),
            "parametros": {
                "modelo__C": loguniform(1e-3, 1e2),
                "modelo__class_weight": [None, "balanced"],
                "modelo__tol": loguniform(1e-5, 1e-2),
            },
            "justificacion": "C y tol en escala logarítmica; class_weight balancea errores entre clases.",
        },
        "Candidato_HistGradientBoosting": {
            "familia": "boosting",
            "estimador": crear_pipeline(HistGradientBoostingClassifier(random_state=seed)),
            "parametros": {
                "modelo__learning_rate": loguniform(0.01, 0.3),
                "modelo__max_iter": randint(60, 251),
                "modelo__max_leaf_nodes": randint(7, 40),
                "modelo__min_samples_leaf": randint(10, 101),
                "modelo__l2_regularization": loguniform(1e-5, 10),
                "modelo__max_depth": [3, 5, 8, None],
            },
            "justificacion": "learning_rate y L2 logarítmicas; hojas/iteraciones para controlar complejidad.",
        },
        "Candidato_SVC": {
            "familia": "kernel_svm",
            "estimador": crear_pipeline(SVC(probability=True, random_state=seed)),
            "parametros": {
                "modelo__C": loguniform(0.01, 100),
                "modelo__gamma": loguniform(1e-4, 1),
                "modelo__class_weight": [None, "balanced"],
            },
            "justificacion": "C y gamma varían por órdenes de magnitud; kernel RBF constante.",
        },
    }


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd="/workspace", stderr=subprocess.DEVNULL,
            text=True, timeout=5,
        ).strip()
    except Exception:
        return "SIN_COMMIT_LOCAL"


def _params_log(params):
    """MLflow soporta strings; serializamos de manera explícita."""
    return {k: str(v) for k, v in params.items()}


def _guardar_modelo(modelo, X, nombre, ruta_codigo="/workspace/src"):
    ejemplo = X.head(5)
    firma = infer_signature(ejemplo, modelo.predict(ejemplo))
    mlflow.sklearn.log_model(
        sk_model=modelo, name=nombre, signature=firma,
        input_example=ejemplo, code_paths=[ruta_codigo],
    )


def ajustar_y_registrar(nombre, X_train, y_train, folds, info, baseline_df,
                        out_dir="/workspace/docs", guardar_modelos_hijos=True):
    """1 búsqueda RandomizedSearchCV + 30 nested child runs + 1 parent run.

    El resultado incluye 30 corridas hijas en MLflow y refit del mejor en train.
    En hijos, el pipeline se vuelve a entrenar SOLO en train para registrar
    artefacto predictivo (ningún dato de test se utiliza).
    """
    catalogo = definiciones_modelos(info["seed"])
    if nombre not in catalogo:
        raise ValueError(f"Modelo no habilitado: {nombre}")
    d = catalogo[nombre]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prior = baseline_df.loc[baseline_df["nombre"].eq(nombre)]
    if len(prior) != 1:
        raise ValueError("Se necesita exactamente una línea de resultados del notebook 03")
    prior = prior.iloc[0]
    seed = int(info["seed"])
    mlflow.set_experiment(EXPERIMENTO)
    start = time.perf_counter()
    reg = []

    with mlflow.start_run(run_name=f"Tuning_{nombre}") as padre:
        padre_id = padre.info.run_id
        mlflow.set_tags({
            "fase": "tuning", "familia": d["familia"],
            "git_commit": git_sha(), "origen": info["origen"],
            "hash_dataset": info["hash_dataset"],
            "metodo": "RandomizedSearchCV", "nota": "SOLO train; hold-out reservado",
        })
        mlflow.log_params({
            "seed": seed, "n_configuraciones": N_CONFIG, "n_folds": N_FOLDS,
            "n_train": len(X_train), "hash_snapshot": info["hash_dataset"],
            "fecha_snapshot": "sin_fecha_de_negocio",
            "criterio": "average_precision", "esquema_cv": "StratifiedKFold5",
        })
        mlflow.log_dict(info, "split_info.json")
        mlflow.log_dict({"busqueda": d["justificacion"],
                         "rangos": {k: str(v) for k, v in d["parametros"].items()}},
                        "espacio_busqueda.json")
        busqueda = RandomizedSearchCV(
            estimator=d["estimador"], param_distributions=d["parametros"],
            n_iter=N_CONFIG, scoring="average_precision", cv=folds,
            refit=True, return_train_score=True, random_state=seed,
            n_jobs=1, error_score="raise", verbose=1,
        )
        busqueda.fit(X_train, y_train)
        cvres = busqueda.cv_results_
        correccion_std = np.sqrt(N_FOLDS / (N_FOLDS - 1))
        for i, params in enumerate(cvres["params"]):
            # Una corrida hija por configuración evaluada por la búsqueda.
            mean_ap = float(cvres["mean_test_score"][i])
            sd_ap = float(cvres["std_test_score"][i]) * correccion_std
            mean_train = float(cvres["mean_train_score"][i])
            with mlflow.start_run(run_name=f"{nombre}_config_{i+1:02d}", nested=True) as hijo:
                hijo_id = hijo.info.run_id
                mlflow.set_tags({
                    "fase": "tuning_configuracion", "familia": d["familia"],
                    "git_commit": git_sha(), "hash_dataset": info["hash_dataset"],
                    "origen": info["origen"], "parent_run_id": padre_id,
                })
                mlflow.log_params({
                    "seed": seed, "indice": i+1, "n_train": len(X_train),
                    "dataset_sha256": info["hash_dataset"],
                    "cv": "StratifiedKFold5", **_params_log(params),
                })
                mlflow.log_metrics({
                    "cv_ap_mean": mean_ap, "cv_ap_std": sd_ap,
                    "cv_train_ap_mean": mean_train,
                    "gap_train_vs_cv": mean_train - mean_ap,
                    "cv_fit_time_mean_sec": float(cvres["mean_fit_time"][i]),
                    "cv_fit_time_std_sec": float(cvres["std_fit_time"][i]),
                })
                mlflow.log_dict({"config_index": i+1, "params": _params_log(params),
                                 "cv_ap_mean":mean_ap, "cv_ap_std":sd_ap},
                                "configuracion.json")
                if guardar_modelos_hijos:
                    # Refit para contar con un artefacto usable en cada corrida hija.
                    modelo_hijo = clone(d["estimador"]).set_params(**params)
                    modelo_hijo.fit(X_train, y_train)
                    _guardar_modelo(modelo_hijo, X_train, "pipeline")
                reg.append({
                    "nombre": nombre, "config": i+1, "child_run_id":hijo_id,
                    "parent_run_id":padre_id, "cv_ap_mean":mean_ap,
                    "cv_ap_std":sd_ap,"train_ap_mean":mean_train,
                    "gap":mean_train-mean_ap,
                    "cv_fit_seconds":float(cvres["mean_fit_time"][i]),
                    "params_json":json.dumps(_params_log(params),sort_keys=True),
                })

        ix = int(busqueda.best_index_)
        tuned_ap = float(cvres["mean_test_score"][ix])
        tuned_std = float(cvres["std_test_score"][ix]) * correccion_std
        default_ap = float(prior["cv_ap_mean"])
        default_std = float(prior["cv_ap_std"])
        mejora = tuned_ap - default_ap
        supera_var = mejora > max(tuned_std,default_std)
        total = time.perf_counter() - start
        mlflow.log_metrics({
            "best_cv_ap_mean": tuned_ap,"best_cv_ap_std":tuned_std,
            "default_cv_ap_mean":default_ap,"default_cv_ap_std":default_std,
            "improvement_ap":mejora,
            "improvement_exceeds_fold_variability":int(supera_var),
            "best_cv_train_ap_mean":float(cvres["mean_train_score"][ix]),
            "best_cv_gap":float(cvres["mean_train_score"][ix])-tuned_ap,
            "best_refit_seconds":float(busqueda.refit_time_),
            "elapsed_seconds":total,
        })
        mlflow.log_params({f"best_{k}":str(v) for k,v in busqueda.best_params_.items()})
        _guardar_modelo(busqueda.best_estimator_, X_train, "mejor_pipeline")
        mlflow.log_dict({"best_params":_params_log(busqueda.best_params_),
                         "best_index":ix,"best_cv_ap":tuned_ap},"mejor_configuracion.json")

    detalle = pd.DataFrame(reg)
    detalle.to_csv(out/f"tuning_configuraciones_{nombre}.csv",index=False)
    resumen = {
        "nombre":nombre,"parent_run_id":padre_id,
        "best_child_run_id":reg[ix]["child_run_id"],
        "default_ap":default_ap,"default_std":default_std,
        "tuned_ap":tuned_ap,"tuned_std":tuned_std,
        "mejora_ap":mejora,"mejora_supera_variabilidad":bool(supera_var),
        "train_ap":float(cvres["mean_train_score"][ix]),
        "brecha_train_cv":float(cvres["mean_train_score"][ix])-tuned_ap,
        "fit_refit_seconds":float(busqueda.refit_time_),
        "elapsed_seconds":total,"num_config":len(reg),
        "best_params_json":json.dumps(_params_log(busqueda.best_params_),sort_keys=True),
    }
    print(f"{nombre}: {len(reg)} configs | AP {default_ap:.4f} -> {tuned_ap:.4f} "
          f"(mejora {mejora:+.4f}; sd={tuned_std:.4f}).")
    print(f"  ¿Mejora mayor que variación entre folds? {supera_var}")
    return resumen
