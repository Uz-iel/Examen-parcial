"""Funciones utilitarias para configuración, logs, JSON y carpetas."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import yaml


def cargar_config(config_path: str) -> dict:
    """Carga un archivo YAML de configuración."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def crear_directorios(rutas: list[str]) -> None:
    """Crea carpetas si no existen."""
    for ruta in rutas:
        os.makedirs(ruta, exist_ok=True)


def fecha_actual() -> str:
    """Devuelve fecha y hora en formato legible."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_archivo() -> str:
    """Devuelve timestamp seguro para nombres de archivos o carpetas."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def configurar_logger(logs_dir: str, nombre_base: str) -> logging.Logger:
    """Configura logger en consola y archivo.

    El logger se reinicia en cada ejecución para evitar mensajes duplicados.
    """
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{nombre_base}_{timestamp_archivo()}.log")

    logger = logging.getLogger("pipeline_mlops_parcial")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formato = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formato)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formato)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.log_path = log_path  # atributo útil para metadata
    return logger


def convertir_json_seguro(objeto: Any) -> Any:
    """Convierte objetos no serializables a estructuras compatibles con JSON.

    Evita errores al guardar metadata con objetos numpy, sklearn o pipelines.
    """
    if isinstance(objeto, dict):
        return {str(k): convertir_json_seguro(v) for k, v in objeto.items()}

    if isinstance(objeto, (list, tuple, set)):
        return [convertir_json_seguro(v) for v in objeto]

    if isinstance(objeto, (str, int, float, bool)) or objeto is None:
        return objeto

    if isinstance(objeto, (np.integer, np.floating, np.bool_)):
        return objeto.item()

    if hasattr(objeto, "item"):
        try:
            return objeto.item()
        except Exception:
            pass

    return str(objeto)


def guardar_json(objeto: Any, path: str) -> None:
    """Guarda un objeto como JSON, convirtiendo valores complejos si es necesario."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(convertir_json_seguro(objeto), file, indent=4, ensure_ascii=False)


def cargar_json(path: str) -> dict:
    """Carga un archivo JSON."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
