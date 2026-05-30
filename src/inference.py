"""Funciones de inferencia desde  Registry."""

from __future__ import annotations

import joblib

from src.utils import cargar_json


def cargar_modelo_campeon(champion: dict):
    """Carga model.pkl y metadata.json del campeón activo."""
    modelo = joblib.load(champion["model_path"])
    metadata = cargar_json(champion["metadata_path"])
    return modelo, metadata


def alinear_features(X, feature_names: list[str]):
    """Alinea columnas de inferencia con las features del entrenamiento."""
    X_alineado = X.copy()
    for col in feature_names:
        if col not in X_alineado.columns:
            X_alineado[col] = 0
    return X_alineado[feature_names]


def predecir_probabilidades(modelo, X):
    """Genera probabilidad de clase positiva."""
    return modelo.predict_proba(X)[:, 1]
