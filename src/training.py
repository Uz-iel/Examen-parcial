"""Entrenamiento, HPO, paralelismo con joblib y selección del campeón."""

from __future__ import annotations

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics import calcular_metricas
from src.registry import actualizar_registry
from src.utils import fecha_actual, guardar_json, timestamp_archivo


def construir_pipeline(modelo) -> Pipeline:
    """Construye pipeline sklearn con escalado y modelo."""
    return Pipeline([("scaler", StandardScaler()), ("model", modelo)])


def obtener_modelos_base(config: dict) -> dict:
    """Define modelos candidatos simples y defendibles."""
    random_state = config["proyecto"]["random_state"]
    return {
        "LogisticRegression": construir_pipeline(
            LogisticRegression(max_iter=500, C=0.1, class_weight="balanced", random_state=random_state)
        ),
        "RandomForest": construir_pipeline(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            )
        ),
        "ExtraTrees": construir_pipeline(
            ExtraTreesClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            )
        ),
        "GradientBoosting": construir_pipeline(
            GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=random_state)
        ),
    }


def filtrar_modelos_configurados(modelos: dict, config: dict) -> dict:
    """Filtra modelos según YAML."""
    lista = config["training"].get("modelos", list(modelos.keys()))
    return {nombre: modelos[nombre] for nombre in lista if nombre in modelos}


def evaluar_modelo(nombre: str, modelo, X_train, y_train, X_valid, y_valid, X_oot, y_oot, config: dict) -> dict:
    """Entrena un modelo y calcula métricas train, validación y OOT."""
    umbral = config["training"].get("umbral_clasificacion", 0.50)
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo_fit = time.time() - inicio

    proba_train = modelo.predict_proba(X_train)[:, 1]
    proba_valid = modelo.predict_proba(X_valid)[:, 1]
    proba_oot = modelo.predict_proba(X_oot)[:, 1]

    metricas_train = calcular_metricas(y_train, proba_train, umbral=umbral)
    metricas_valid = calcular_metricas(y_valid, proba_valid, umbral=umbral)
    metricas_oot = calcular_metricas(y_oot, proba_oot, umbral=umbral)
    auc_gap = abs(metricas_valid["auc"] - metricas_oot["auc"])

    return {
        "nombre": nombre,
        "modelo": modelo,
        "metricas_train": metricas_train,
        "metricas_test": metricas_valid,
        "metricas_oot": metricas_oot,
        "auc_test": metricas_valid["auc"],
        "auc_oot": metricas_oot["auc"],
        "auc_gap": auc_gap,
        "tiempo_fit_s": round(tiempo_fit, 3),
        "params": modelo.get_params(),
    }


def ejecutar_secuencial(modelos: dict, X_train, y_train, X_valid, y_valid, X_oot, y_oot, config: dict) -> tuple[list, float]:
    """Entrena modelos uno por uno para medir tiempo secuencial."""
    inicio = time.time()
    resultados = []
    for nombre, modelo in modelos.items():
        resultados.append(evaluar_modelo(nombre, modelo, X_train, y_train, X_valid, y_valid, X_oot, y_oot, config))
    return resultados, time.time() - inicio


def ejecutar_paralelo(modelos: dict, X_train, y_train, X_valid, y_valid, X_oot, y_oot, config: dict, logger) -> tuple[list, float, int, int]:
    """Entrena modelos en paralelo usando joblib.Parallel.

    El paralelismo se hace por modelo. Los estimadores de árboles usan n_jobs=1 para evitar doble paralelismo.
    """
    inicio = time.time()
    cpus = os.cpu_count() or 1
    max_workers_cfg = config["training"].get("max_workers")
    max_workers = min(max_workers_cfg or cpus, len(modelos), cpus)

    logger.info("CPUs detectadas: %s", cpus)
    logger.info("Workers usados en paralelo: %s", max_workers)
    logger.info("Modo de paralelismo: joblib.Parallel por modelo")

    tareas = [
        joblib.delayed(evaluar_modelo)(nombre, modelo, X_train, y_train, X_valid, y_valid, X_oot, y_oot, config)
        for nombre, modelo in modelos.items()
    ]
    resultados = joblib.Parallel(n_jobs=max_workers, backend="threading")(tareas)
    for resultado in resultados:
        logger.info("Modelo paralelo completado: %s", resultado["nombre"])

    return resultados, time.time() - inicio, cpus, max_workers


def optimizar_bayes_random_forest(X_train, y_train, X_valid, y_valid, X_oot, y_oot, config: dict, logger) -> dict:
    """Ejecuta HPO sobre RandomForest.

    Usa BayesSearchCV si scikit-optimize está instalado. Si no, usa RandomizedSearchCV como respaldo.
    """
    random_state = config["proyecto"]["random_state"]
    n_iter = config["training"].get("bayesian_n_iter", 2)
    cv_folds = config["training"].get("cv_folds", 2)
    sample_size = config["training"].get("hpo_sample_size")

    X_hpo = X_train
    y_hpo = y_train
    if sample_size and len(X_train) > sample_size:
        logger.info("HPO usando muestra de %s filas para reducir tiempo", sample_size)
        muestra = pd.Series(y_train).sample(n=sample_size, random_state=random_state).index
        X_hpo = X_train.loc[muestra]
        y_hpo = y_train.loc[muestra]

    modelo = construir_pipeline(RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=1))
    inicio = time.time()

    try:
        from skopt import BayesSearchCV
        from skopt.space import Integer

        espacio = {
            "model__n_estimators": Integer(60, 140),
            "model__max_depth": Integer(5, 14),
            "model__min_samples_split": Integer(2, 18),
            "model__min_samples_leaf": Integer(1, 8),
        }
        buscador = BayesSearchCV(
            estimator=modelo,
            search_spaces=espacio,
            n_iter=n_iter,
            cv=cv_folds,
            scoring="roc_auc",
            n_jobs=1,
            random_state=random_state,
            verbose=0,
        )
        logger.info("Inicio de Bayesian Optimization para RandomForest")
    except Exception as error:
        logger.warning("scikit-optimize no disponible (%s). Se usará RandomizedSearchCV como respaldo.", error)
        espacio = {
            "model__n_estimators": [60, 80, 100, 120, 140],
            "model__max_depth": [5, 8, 10, 12, 14],
            "model__min_samples_split": [2, 5, 10, 18],
            "model__min_samples_leaf": [1, 2, 4, 8],
        }
        buscador = RandomizedSearchCV(
            estimator=modelo,
            param_distributions=espacio,
            n_iter=n_iter,
            cv=cv_folds,
            scoring="roc_auc",
            n_jobs=1,
            random_state=random_state,
            verbose=0,
        )
        logger.info("Inicio de RandomizedSearchCV para RandomForest")

    buscador.fit(X_hpo, y_hpo)
    tiempo_hpo = time.time() - inicio
    logger.info("HPO finalizado en %.2f segundos", tiempo_hpo)
    logger.info("Mejores parámetros: %s", dict(buscador.best_params_))

    mejor_modelo = buscador.best_estimator_
    resultado = evaluar_modelo("RandomForest_HPO", mejor_modelo, X_train, y_train, X_valid, y_valid, X_oot, y_oot, config)
    resultado["tiempo_hpo_s"] = round(tiempo_hpo, 3)
    resultado["mejores_parametros"] = dict(buscador.best_params_)
    resultado["best_score_hpo"] = float(buscador.best_score_)
    return resultado


def seleccionar_campeon(resultados: list[dict], stability_threshold: float, logger) -> dict:
    """Selecciona campeón por estabilidad OOT y mayor AUC test."""
    logger.info("----------------------------------------------------------------------")
    logger.info("LEADERBOARD DE MODELOS")
    logger.info("Modelo                   AUC_train  AUC_test   AUC_oot    Gap        Estable")

    for r in resultados:
        r["stability"] = "ESTABLE" if r["auc_gap"] <= stability_threshold else "INESTABLE"

    resultados_ordenados = sorted(resultados, key=lambda x: x["auc_test"], reverse=True)
    for r in resultados_ordenados:
        logger.info(
            "%-24s %.4f     %.4f     %.4f     %.4f     %s",
            r["nombre"],
            r["metricas_train"]["auc"],
            r["auc_test"],
            r["auc_oot"],
            r["auc_gap"],
            "OK" if r["stability"] == "ESTABLE" else "NO",
        )

    estables = [r for r in resultados_ordenados if r["stability"] == "ESTABLE"]
    if estables:
        campeon = max(estables, key=lambda x: x["auc_test"])
    else:
        campeon = min(resultados_ordenados, key=lambda x: x["auc_gap"])
        campeon["stability"] = "INESTABLE"

    logger.info("Campeón seleccionado: %s | %s", campeon["nombre"], campeon["stability"])
    return campeon


def limpiar_resultado(resultado: dict) -> dict:
    """Quita objeto modelo para metadata."""
    return {k: v for k, v in resultado.items() if k != "modelo"}


def guardar_campeon(campeon: dict, resultados: list[dict], feature_names: list[str], config: dict, logger) -> tuple[str, str, dict]:
    """Guarda model.pkl, metadata.json y actualiza registry."""
    best_dir = config["paths"].get("best_model_dir", "best_model")
    os.makedirs(best_dir, exist_ok=True)
    timestamp = timestamp_archivo()
    model_dir = os.path.join(best_dir, f"{campeon['nombre']}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pkl")
    metadata_path = os.path.join(model_dir, "metadata.json")
    joblib.dump(campeon["modelo"], model_path)

    metadata = {
        "model_name": campeon["nombre"],
        "registered_at": fecha_actual(),
        "model_dir": model_dir,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "stability": campeon["stability"],
        "auc_train": float(campeon["metricas_train"]["auc"]),
        "auc_test": float(campeon["auc_test"]),
        "auc_oot": float(campeon["auc_oot"]),
        "auc_gap": float(campeon["auc_gap"]),
        "metricas_train": campeon["metricas_train"],
        "metricas_test": campeon["metricas_test"],
        "metricas_oot": campeon["metricas_oot"],
        "umbral_clasificacion": config["training"].get("umbral_clasificacion", 0.50),
        "stability_threshold": config["training"].get("stability_threshold", 0.03),
        "features": feature_names,
        "n_features": len(feature_names),
        "experimento": config.get("experimento", {}),
        "candidatos": [limpiar_resultado(r) for r in resultados],
        "mejores_parametros": campeon.get("mejores_parametros", {}),
    }
    guardar_json(metadata, metadata_path)
    logger.info("Modelo campeón guardado en: %s", model_path)

    registry = actualizar_registry(metadata, best_dir)
    logger.info("Registry actualizado -> %s: %s", registry["champion"]["version"], campeon["nombre"])
    return model_path, metadata_path, metadata


def registrar_mlflow(resultados: list[dict], campeon: dict, config: dict, logger) -> None:
    """Registra experimentos en MLflow si está disponible."""
    try:
        import mlflow
    except Exception as error:
        logger.warning("MLflow no disponible: %s", error)
        return

    tracking_uri = config.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow_store/mlflow.db")
    experiment_name = config.get("mlflow", {}).get("experiment_name", "parcial_mlops_propension")
    if tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    for r in resultados:
        with mlflow.start_run(run_name=r["nombre"]):
            mlflow.log_param("modelo", r["nombre"])
            mlflow.log_param("experimento", config.get("experimento", {}).get("nombre", "sin_nombre"))
            mlflow.log_param("umbral_clasificacion", config["training"].get("umbral_clasificacion", 0.50))
            mlflow.log_param("stability_threshold", config["training"].get("stability_threshold", 0.03))
            mlflow.log_metric("auc_train", r["metricas_train"]["auc"])
            mlflow.log_metric("auc_test", r["auc_test"])
            mlflow.log_metric("auc_oot", r["auc_oot"])
            mlflow.log_metric("auc_gap", r["auc_gap"])
            mlflow.log_metric("accuracy_test", r["metricas_test"]["accuracy"])
            mlflow.log_metric("f1_test", r["metricas_test"]["f1"])
            mlflow.log_metric("tiempo_fit_s", r.get("tiempo_fit_s", 0))
            mlflow.set_tag("is_champion", r["nombre"] == campeon["nombre"])
            mlflow.set_tag("stability", r.get("stability", "SIN_ESTADO"))
    logger.info("Experimentos registrados en MLflow: %s", tracking_uri)
