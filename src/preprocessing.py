"""Preprocesamiento para entrenamiento e inferencia."""

from __future__ import annotations

import os
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from src.variables import COLUMNAS_POST, FEATURES_MODELO, TARGET, VARIABLE_CATEGORICA, VARIABLES_NUMERICAS


def construir_url_periodo(config: dict, periodo: int | str) -> str:
    """Construye la URL de descarga para un periodo."""
    base_url = str(config["data"].get("url_base", "")).rstrip("/")
    plantilla = config["data"].get("archivo_periodo", "p{period}_extrac.csv")
    if not base_url:
        return ""
    return f"{base_url}/{plantilla.format(period=periodo)}"


def buscar_o_descargar_periodo(periodo: int | str, carpeta: str, config: dict, logger) -> str:
    """Busca p{period}_extrac.csv localmente o lo descarga desde Git si está configurado."""
    os.makedirs(carpeta, exist_ok=True)
    nombre = config["data"].get("archivo_periodo", "p{period}_extrac.csv").format(period=periodo)
    path = os.path.join(carpeta, nombre)

    if os.path.exists(path):
        logger.info("Archivo encontrado: %s", path)
        return path

    auto_download = bool(config["data"].get("auto_download", True))
    url = construir_url_periodo(config, periodo)
    if auto_download and url:
        try:
            logger.info("Descargando: %s", url)
            urlretrieve(url, path)
            return path
        except Exception as error:
            logger.warning("No se pudo descargar %s: %s", url, error)

    raise FileNotFoundError(
        f"No se encontró {nombre} en {carpeta}. "
        "Coloca el CSV real en data/raw o ejecuta: python generate_sample_data.py --filas 8000 --periodos 10"
    )


def leer_csv_seguro(path: str, limite_filas: int | None = None) -> pd.DataFrame:
    """Lee un CSV con límite opcional de filas."""
    nrows = None if not limite_filas or int(limite_filas) <= 0 else int(limite_filas)
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def cargar_periodos(periodos: list[int], carpeta: str, config: dict, logger, limite_filas: int | None) -> pd.DataFrame:
    """Carga y concatena varios periodos."""
    dataframes = []
    for periodo in periodos:
        path = buscar_o_descargar_periodo(periodo, carpeta, config, logger)
        df = leer_csv_seguro(path, limite_filas=limite_filas)
        if "partition" not in df.columns:
            df["partition"] = f"p{periodo}"
        dataframes.append(df)
        logger.info("Periodo %s cargado: %s", periodo, df.shape)
    return pd.concat(dataframes, ignore_index=True)


def normalizar_target(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """Convierte target a entero 0/1 si existe."""
    df = df.copy()
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
        df[TARGET] = np.where(df[TARGET] > 0, 1, 0)
    elif logger:
        logger.warning("La data no trae target. Esto es esperado en inferencia real.")
    return df


def completar_columnas_post(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura columnas de negocio necesarias para postprocesamiento."""
    df = df.copy()
    for col in COLUMNAS_POST:
        if col not in df.columns:
            if col == "partition":
                df[col] = "sin_periodo"
            elif col == "prob_value_contact":
                df[col] = 1.0
            elif col == "monto":
                df[col] = 0.0
            elif col == "grp_campecs06m":
                df[col] = "OTRO"
            else:
                df[col] = "SIN_ID"
    df["prob_value_contact"] = pd.to_numeric(df["prob_value_contact"], errors="coerce").fillna(1.0)
    df["monto"] = pd.to_numeric(df["monto"], errors="coerce").fillna(0.0)
    df["grp_campecs06m"] = df["grp_campecs06m"].astype(str).str.upper().fillna("OTRO")
    return df


def procesar_variables(df: pd.DataFrame, incluir_target: bool = True, logger=None) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """Convierte data cruda en matriz de modelo, target y columnas de negocio.

    - El diccionario trae 69 campos originales.
    - No se usan como predictoras directas: target, identificadores, fechas y variables de negocio para TLV.
    - Quedan 58 variables numéricas + 2 dummies de la categórica ent_1erlntcrallsfm01 = 60 features finales.
    """
    df = df.copy()
    df = df.replace(["", "null", "None", "NULL", "nan"], np.nan)
    df = completar_columnas_post(df)
    df = normalizar_target(df, logger=logger)

    y = None
    if incluir_target and TARGET in df.columns:
        y = df[TARGET].astype(int)

    for col in VARIABLES_NUMERICAS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-9999999).astype("float32")

    if VARIABLE_CATEGORICA not in df.columns:
        df[VARIABLE_CATEGORICA] = "OTRO"
    df[VARIABLE_CATEGORICA] = df[VARIABLE_CATEGORICA].astype(str).str.upper().fillna("OTRO")
    df[VARIABLE_CATEGORICA] = np.where(df[VARIABLE_CATEGORICA].eq("INTERBANK"), "INTERBANK", "OTRO")

    dummies = pd.get_dummies(df[VARIABLE_CATEGORICA], prefix=VARIABLE_CATEGORICA).astype(int)
    for dummy_col in ["ent_1erlntcrallsfm01_INTERBANK", "ent_1erlntcrallsfm01_OTRO"]:
        if dummy_col not in dummies.columns:
            dummies[dummy_col] = 0

    X = pd.concat(
        [df[VARIABLES_NUMERICAS], dummies[["ent_1erlntcrallsfm01_INTERBANK", "ent_1erlntcrallsfm01_OTRO"]]],
        axis=1,
    )
    X = X[FEATURES_MODELO].copy()
    post_df = df[COLUMNAS_POST].copy()
    return X, y, post_df


def guardar_processed_entrenamiento(datasets: dict, config: dict) -> None:
    """Guarda artefactos intermedios auditables de entrenamiento."""
    out_dir = config["data"].get("processed_training_dir", "data/processed/training")
    os.makedirs(out_dir, exist_ok=True)
    for nombre in ["train", "valid", "oot"]:
        X = datasets[f"X_{nombre}"]
        y = datasets[f"y_{nombre}"]
        post = datasets[f"post_{nombre}"]
        X.to_csv(os.path.join(out_dir, f"X_{nombre}.csv"), index=False)
        y.to_frame(TARGET).to_csv(os.path.join(out_dir, f"y_{nombre}.csv"), index=False)
        post.to_csv(os.path.join(out_dir, f"post_{nombre}.csv"), index=False)


def guardar_processed_inferencia(X: pd.DataFrame, post_df: pd.DataFrame, etiqueta: str, config: dict) -> None:
    """Guarda dataset procesado de inferencia para auditoría."""
    out_dir = config["data"].get("processed_inference_dir", "data/processed/inference")
    os.makedirs(out_dir, exist_ok=True)
    X.to_csv(os.path.join(out_dir, f"X_inferencia_{etiqueta}.csv"), index=False)
    post_df.to_csv(os.path.join(out_dir, f"post_inferencia_{etiqueta}.csv"), index=False)


def preparar_datasets_entrenamiento(config: dict, logger) -> dict:
    """Carga periodos y devuelve datasets train, validación y OOT procesados."""
    data_cfg = config["data"]
    periodos = config["periodos"]
    limite = data_cfg.get("limite_filas_por_periodo")

    carpeta_training = data_cfg.get("raw_training_dir", "data/raw/training")

    df_train = cargar_periodos(periodos["entrenamiento"], carpeta_training, config, logger, limite)
    df_valid = cargar_periodos(periodos["validacion"], carpeta_training, config, logger, limite)
    df_oot = cargar_periodos(periodos["oot"], carpeta_training, config, logger, limite)

    X_train, y_train, post_train = procesar_variables(df_train, incluir_target=True, logger=logger)
    X_valid, y_valid, post_valid = procesar_variables(df_valid, incluir_target=True, logger=logger)
    X_oot, y_oot, post_oot = procesar_variables(df_oot, incluir_target=True, logger=logger)

    datasets = {
        "X_train": X_train,
        "y_train": y_train,
        "post_train": post_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "post_valid": post_valid,
        "X_oot": X_oot,
        "y_oot": y_oot,
        "post_oot": post_oot,
    }
    guardar_processed_entrenamiento(datasets, config)
    return datasets


def preparar_dataset_inferencia(config: dict, logger, periodo: int | str | None = None, input_file: str | None = None):
    """Carga y procesa un periodo de inferencia o archivo externo."""
    limite = config["data"].get("limite_filas_inferencia")

    if input_file:
        path = input_file
        etiqueta = os.path.splitext(os.path.basename(input_file))[0]
        logger.info("Usando archivo externo de inferencia: %s", path)
    else:
        if periodo is None:
            periodo = config["periodos"].get("inferencia_demo", 10)
        carpeta_inferencia = config["data"].get("raw_inference_dir", "data/raw/inference")
        path = buscar_o_descargar_periodo(periodo, carpeta_inferencia, config, logger)
        etiqueta = str(periodo)

    df = leer_csv_seguro(path, limite_filas=limite)
    target_col = None
    if TARGET in df.columns:
        logger.warning("La data de inferencia trae target. Se ignora para predecir y se conserva solo para evaluación offline.")
        target_col = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)

    X, _, post_df = procesar_variables(df, incluir_target=False, logger=logger)
    guardar_processed_inferencia(X, post_df, etiqueta, config)

    target_real = None
    if target_col is not None:
        target_real = post_df[[c for c in ["key_value", "codunicocli", "partition"] if c in post_df.columns]].copy()
        target_real["target_real"] = target_col.values

    return X, post_df, target_real, etiqueta
