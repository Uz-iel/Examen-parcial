"""Registro simple de modelos campeones."""

from __future__ import annotations

import os

from src.utils import cargar_json, guardar_json


def cargar_registry(best_model_dir: str = "best_model") -> dict:
    """Carga registry.json si existe."""
    path = os.path.join(best_model_dir, "registry.json")
    if not os.path.exists(path):
        return {"champion": None, "history": []}
    return cargar_json(path)


def guardar_registry(registry: dict, best_model_dir: str = "best_model") -> str:
    """Guarda registry.json."""
    os.makedirs(best_model_dir, exist_ok=True)
    path = os.path.join(best_model_dir, "registry.json")
    guardar_json(registry, path)
    return path


def actualizar_registry(metadata: dict, best_model_dir: str = "best_model") -> dict:
    """Registra un nuevo campeón y mueve el anterior al historial."""
    registry = cargar_registry(best_model_dir)
    historial = registry.get("history", [])
    champion_actual = registry.get("champion")

    if champion_actual:
        historial.append(champion_actual)

    nueva_version = f"v{len(historial) + 1}"
    nuevo_champion = {
        "version": nueva_version,
        "model_name": metadata["model_name"],
        "auc_train": metadata.get("auc_train"),
        "auc_test": metadata["auc_test"],
        "auc_oot": metadata["auc_oot"],
        "auc_gap": metadata["auc_gap"],
        "stability": metadata["stability"],
        "n_features": metadata["n_features"],
        "path": metadata["model_dir"],
        "model_path": metadata["model_path"],
        "metadata_path": metadata["metadata_path"],
        "registered_at": metadata["registered_at"],
    }

    registry = {"champion": nuevo_champion, "history": historial}
    guardar_registry(registry, best_model_dir)
    return registry


def obtener_campeon(best_model_dir: str = "best_model") -> dict:
    """Devuelve el campeón activo del registry."""
    registry = cargar_registry(best_model_dir)
    champion = registry.get("champion")
    if not champion:
        raise FileNotFoundError("No hay modelo campeón registrado. Ejecuta primero training_pipeline.py")
    return champion
