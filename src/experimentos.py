"""Helpers para comparar modelos SIN utilizar el hold-out.

La métrica principal es Average Precision (AP), que resume la curva PR sin
interpolación; se declara explícitamente para no confundirla con AUC trapecial.
"""
import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import cross_validate


def evaluar_cv(estimator, X, y, folds, costo_contacto=5.0,
              costo_abandono=120.0, eficacia_hipotetica=0.20):
    """Evalúa únicamente sobre TRAIN usando los pliegues recibidos.

    Costo ilustrativo: todo contacto cuesta costo_contacto, y de los TP
    contactados solo el porcentaje eficacia_hipotetica evitaría la pérdida.
    Ninguno de esos supuestos está identificado en el dataset público.
    """
    cv = cross_validate(
        estimator, X, y,
        cv=folds,
        scoring={"ap": "average_precision", "recall": "recall", "precision": "precision"},
        return_train_score=True,
        return_estimator=True,
        error_score="raise",
        n_jobs=1,
    )
    yn = np.asarray(y)
    tp_total = fp_total = fn_total = tn_total = 0
    for (_, idx_valid), modelo_plegado in zip(folds, cv["estimator"]):
        X_valid = X.iloc[idx_valid]
        y_valid = yn[idx_valid]
        pred = modelo_plegado.predict(X_valid)
        tn, fp, fn, tp = confusion_matrix(y_valid, pred, labels=[0, 1]).ravel()
        tp_total += int(tp)
        fp_total += int(fp)
        fn_total += int(fn)
        tn_total += int(tn)

    costo_estimado = (
        (tp_total + fp_total) * costo_contacto
        + (fn_total + tp_total * (1 - eficacia_hipotetica)) * costo_abandono
    )
    normalizado = 100 * costo_estimado / len(yn)
    return {
        "cv_ap_mean": float(np.mean(cv["test_ap"])),
        "cv_ap_std": float(np.std(cv["test_ap"], ddof=1)),
        "cv_ap_train_mean": float(np.mean(cv["train_ap"])),
        "cv_recall_mean": float(np.mean(cv["test_recall"])),
        "cv_precision_mean": float(np.mean(cv["test_precision"])),
        "cv_fit_sec_mean": float(np.mean(cv["fit_time"])),
        "cv_cost_per100_illustrative": float(normalizado),
        "cv_tp": tp_total, "cv_fp": fp_total,
        "cv_fn": fn_total, "cv_tn": tn_total,
    }


def entrenar_final_en_train(estimator, X_train, y_train):
    modelo = clone(estimator)
    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    segundos = time.perf_counter() - inicio
    if hasattr(modelo, "predict_proba"):
        puntajes = modelo.predict_proba(X_train)[:, 1]
    else:
        puntajes = modelo.decision_function(X_train)
    return modelo, segundos, float(average_precision_score(y_train, puntajes))
