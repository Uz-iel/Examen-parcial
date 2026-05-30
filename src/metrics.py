"""Métricas técnicas para clasificación binaria."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def calcular_metricas(y_true, y_proba, umbral: float = 0.50) -> dict:
    """Calcula métricas usando probabilidades y un umbral binario.
    """
    y_pred = (np.asarray(y_proba) >= umbral).astype(int)
    metricas = {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "umbral": float(umbral),
    }
    return metricas


def calcular_lift_por_deciles(df, score_col: str, target_col: str, n_deciles: int = 10) -> tuple:
    """Calcula tabla de deciles y lift para evaluación offline.

    """
    datos = df[[score_col, target_col]].copy()
    datos = datos.dropna(subset=[score_col, target_col])
    datos[target_col] = datos[target_col].astype(int)
    datos = datos.sort_values(score_col, ascending=False).reset_index(drop=True)

    datos["decil"] = pd_qcut_seguro(datos[score_col], n_deciles)
    resumen = (
        datos.groupby("decil", as_index=False)
        .agg(clientes=(target_col, "size"), compradores=(target_col, "sum"), score_promedio=(score_col, "mean"))
        .sort_values("decil")
    )
    resumen["tasa_compra_decil"] = resumen["compradores"] / resumen["clientes"]
    tasa_general = datos[target_col].mean()
    resumen["tasa_compra_general"] = tasa_general
    resumen["lift"] = resumen["tasa_compra_decil"] / tasa_general if tasa_general > 0 else 0
    return resumen, float(tasa_general)


def pd_qcut_seguro(serie, n_deciles: int):
    """Asigna deciles 1..n usando ranking descendente."""
    import pandas as pd

    ranks = serie.rank(method="first", ascending=False)
    return pd.qcut(ranks, q=n_deciles, labels=range(1, n_deciles + 1)).astype(int)
