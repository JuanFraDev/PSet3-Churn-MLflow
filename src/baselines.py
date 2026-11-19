"""Línea base comercial explícita para el PSet #3.

Regla del PSet #1: contrato mensual + antigüedad inferior a 12 meses.
No usa CHURN_FLAG y no aprende parámetros del dataset de prueba.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class ReglaNegocio(ClassifierMixin, BaseEstimator):
    def __init__(self, meses_limite=12):
        self.meses_limite = meses_limite

    def fit(self, X, y=None):
        for columna in ("CONTRACT", "TENURE_MONTHS"):
            if columna not in X.columns:
                raise ValueError(f"La regla necesita {columna}")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        return (
            X["CONTRACT"].eq("Month-to-month")
            & (X["TENURE_MONTHS"] < self.meses_limite)
        ).astype(int).to_numpy()

    def predict_proba(self, X):
        # Score binario: es una regla de decisión, no una probabilidad calibrada.
        unos = self.predict(X).astype(float)
        return np.column_stack([1.0 - unos, unos])
