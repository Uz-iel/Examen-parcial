"""Posprocesamiento: probabilidad, TLV, deciles y réplica de campaña."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.metrics import calcular_lift_por_deciles


def obtener_factor_frescura(grupo: str, config: dict) -> float:
    """Asigna factor de frescura según grupo de campaña.

    Usa
    G1=0.066, G2=0.028, G3=0.022, G4=0.008, G5=0.004.
    """
    post_cfg = config.get("postprocessing", {})
    mapa = post_cfg.get("frescura_map") or config.get("negocio", {}).get("frescura_por_grupo", {})
    default = post_cfg.get("frescura_default", mapa.get("OTRO", 0.004))
    return float(mapa.get(str(grupo).upper(), mapa.get("OTRO", default)))


def asignar_grupo_ejecucion(serie_score: pd.Series, n_grupos: int = 10) -> pd.Series:
    """Asigna grupos 1..10 según ranking descendente. Grupo 1 = mayor prioridad."""
    ranks = serie_score.rank(method="first", ascending=False)
    n_grupos = min(n_grupos, len(serie_score)) if len(serie_score) else n_grupos
    return pd.qcut(ranks, q=n_grupos, labels=range(1, n_grupos + 1)).astype(int)


def construir_scores(post_df: pd.DataFrame, probabilidades, config: dict, umbral: float) -> pd.DataFrame:
    """Construye salida de scores con TLV y grupo de ejecución."""
    df = post_df.copy()
    df["probabilidad_modelo"] = np.asarray(probabilidades, dtype=float)
    df["prediccion_umbral_050"] = (df["probabilidad_modelo"] >= umbral).astype(int)

    df["prob_value_contact"] = pd.to_numeric(df["prob_value_contact"], errors="coerce").fillna(1.0)
    df["monto"] = pd.to_numeric(df["monto"], errors="coerce").fillna(0.0)
    df["frescura"] = df["grp_campecs06m"].apply(lambda x: obtener_factor_frescura(x, config))

    df["puntuacion_tlv"] = (
        df["probabilidad_modelo"]
        * df["prob_value_contact"]
        * np.log1p(df["monto"].clip(lower=0))
        * df["frescura"]
    )
    n_grupos = int(config.get("postprocessing", {}).get("n_grupos_ejecucion", 10))
    df["grupo_ejec"] = asignar_grupo_ejecucion(df["puntuacion_tlv"], n_grupos=n_grupos)
    df = df.sort_values(["grupo_ejec", "puntuacion_tlv"], ascending=[True, False]).reset_index(drop=True)
    df["orden"] = np.arange(1, len(df) + 1)
    return df


def guardar_salidas_inferencia(scores_df: pd.DataFrame, modelo: str, etiqueta: str, output_dir: str) -> tuple[str, str]:
    """Guarda scores CSV y réplica TXT."""
    os.makedirs(output_dir, exist_ok=True)
    scores_path = os.path.join(output_dir, f"scores_{etiqueta}.csv")
    replica_path = os.path.join(output_dir, f"replica_{modelo}_{etiqueta}.txt")

    scores_df.to_csv(scores_path, index=False)

    columnas_replica = [
        "partition",
        "key_value",
        "codunicocli",
        "probabilidad_modelo",
        "puntuacion_tlv",
        "grupo_ejec",
        "orden",
        "monto",
        "prob_value_contact",
        "grp_campecs06m",
    ]
    columnas_replica = [c for c in columnas_replica if c in scores_df.columns]
    scores_df[columnas_replica].to_csv(replica_path, sep="|", index=False)
    return scores_path, replica_path


def _merge_target_por_id(scores_df: pd.DataFrame, target_real) -> pd.DataFrame:
    """Une target de inferencia por key_value/codunicocli para evitar desalineación por orden."""
    eval_df = scores_df.copy()
    if target_real is None:
        return eval_df

    if isinstance(target_real, pd.DataFrame):
        posibles_claves = [c for c in ["key_value", "codunicocli"] if c in eval_df.columns and c in target_real.columns]
        if posibles_claves:
            target_cols = posibles_claves + ["target_real"]
            return eval_df.merge(target_real[target_cols].drop_duplicates(subset=posibles_claves), on=posibles_claves, how="left")
        if "target_real" in target_real.columns and len(target_real) == len(eval_df):
            eval_df["target_real"] = target_real["target_real"].astype(int).values
            return eval_df

    if len(target_real) == len(eval_df):
        eval_df["target_real"] = pd.Series(target_real).astype(int).values
    return eval_df


def evaluar_offline_si_hay_target(scores_df: pd.DataFrame, target_real, output_dir: str, etiqueta: str, logger) -> str | None:
    """Evalúa score contra target real si el archivo de inferencia lo trae.

    """
    if target_real is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    eval_df = _merge_target_por_id(scores_df, target_real)
    if "target_real" not in eval_df.columns or eval_df["target_real"].isna().all():
        logger.warning("No se pudo alinear target real para evaluación offline.")
        return None

    eval_df["target_real"] = pd.to_numeric(eval_df["target_real"], errors="coerce").fillna(0).astype(int)
    try:
        auc = float(roc_auc_score(eval_df["target_real"], eval_df["probabilidad_modelo"]))
    except Exception:
        auc = None

    deciles, tasa_general = calcular_lift_por_deciles(eval_df, "probabilidad_modelo", "target_real", n_deciles=10)
    deciles["auc_inferencia"] = auc
    deciles["tasa_compra_general"] = tasa_general

    path = os.path.join(output_dir, f"evaluacion_offline_deciles_lift_{etiqueta}.csv")
    deciles.to_csv(path, index=False)
    logger.info("Evaluación offline guardada en: %s", path)
    return path
