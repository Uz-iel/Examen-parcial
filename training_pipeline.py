"""Pipeline de entrenamiento del parcial MLOps.

Flujo:
1. Ingesta de datos por periodos.
2. Preprocesamiento.
3. Entrenamiento base secuencial y paralelo CPU.
4. HPO con Bayesian Optimization.
5. Evaluación con validación y OOT.
6. Selección del campeón.
7. Registro en Model Registry.
8. Tracking MLflow opcional.
"""

from __future__ import annotations

import argparse
import traceback

from src.monitoring import MonitorOperativo, guardar_monitoreo
from src.preprocessing import preparar_datasets_entrenamiento
from src.training import (
    ejecutar_paralelo,
    ejecutar_secuencial,
    filtrar_modelos_configurados,
    guardar_campeon,
    obtener_modelos_base,
    optimizar_bayes_random_forest,
    registrar_mlflow,
    seleccionar_campeon,
)
from src.utils import cargar_config, configurar_logger, crear_directorios, guardar_json
from src.reporting import guardar_tablas_entrenamiento
from src.variables import FEATURES_MODELO


def leer_argumentos():
    """Lee argumentos de consola."""
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento MLOps - Parcial")
    parser.add_argument("--config", default="config/config.yaml", help="Ruta del archivo YAML de configuración")
    parser.add_argument("--skip-mlflow", action="store_true", help="Omite tracking MLflow")
    parser.add_argument("--stability-threshold", type=float, default=None, help="Umbral de estabilidad OOT")
    return parser.parse_args()


def main() -> None:
    """Ejecuta entrenamiento completo."""
    args = leer_argumentos()
    config = cargar_config(args.config)
    if args.stability_threshold is not None:
        config["training"]["stability_threshold"] = args.stability_threshold

    paths = config["paths"]
    crear_directorios([
        paths.get("logs_dir", "logs"),
        paths.get("output_dir", "output"),
        paths.get("best_model_dir", "best_model"),
        config["data"].get("raw_training_dir", "data/raw/training"),
        config["data"].get("raw_inference_dir", "data/raw/inference"),
        "data/processed",
        "reports/evidence",
        "reports/tables",
        "reports/figures",
    ])

    logger = configurar_logger(paths.get("logs_dir", "logs"), "training")
    monitor = MonitorOperativo("training_pipeline")

    try:
        logger.info("======================================================================")
        logger.info("PIPELINE DE ENTRENAMIENTO -- Parcial MLOps")
        logger.info("======================================================================")
        logger.info("Config usado: %s", args.config)
        logger.info("Experimento: %s", config.get("experimento", {}).get("nombre", "sin_nombre"))

        logger.info("[STEP 1] Preparando datasets por periodos...")
        datasets = preparar_datasets_entrenamiento(config, logger)
        monitor.registrar_evento("datasets_preparados")

        X_train = datasets["X_train"]
        y_train = datasets["y_train"]
        X_valid = datasets["X_valid"]
        y_valid = datasets["y_valid"]
        X_oot = datasets["X_oot"]
        y_oot = datasets["y_oot"]

        logger.info("Train: %s | tasa target %.4f", X_train.shape, y_train.mean())
        logger.info("Valid: %s | tasa target %.4f", X_valid.shape, y_valid.mean())
        logger.info("OOT  : %s | tasa target %.4f", X_oot.shape, y_oot.mean())

        logger.info("[STEP 2] Entrenando modelos base...")
        modelos = filtrar_modelos_configurados(obtener_modelos_base(config), config)
        resultados_seq, tiempo_seq = ejecutar_secuencial(modelos, X_train, y_train, X_valid, y_valid, X_oot, y_oot, config)

        modelos_par = filtrar_modelos_configurados(obtener_modelos_base(config), config)
        resultados_par, tiempo_par, cpus, workers = ejecutar_paralelo(
            modelos_par, X_train, y_train, X_valid, y_valid, X_oot, y_oot, config, logger
        )
        speedup = tiempo_seq / tiempo_par if tiempo_par > 0 else 1.0
        logger.info("Tiempo secuencial: %.2f", tiempo_seq)
        logger.info("Tiempo paralelo: %.2f", tiempo_par)
        logger.info("Speedup: %.3fx", speedup)
        monitor.registrar_evento("modelos_base_entrenados", {"speedup": speedup})

        logger.info("[STEP 3] Ejecutando HPO con Bayesian Optimization...")
        resultado_bayes = optimizar_bayes_random_forest(X_train, y_train, X_valid, y_valid, X_oot, y_oot, config, logger)
        monitor.registrar_evento("hpo_finalizado")

        logger.info("[STEP 4] Seleccionando campeón con estabilidad OOT...")
        resultados = resultados_par + [resultado_bayes]
        stability_threshold = config["training"].get("stability_threshold", 0.03)
        campeon = seleccionar_campeon(resultados, stability_threshold, logger)

        logger.info("[STEP 5] Guardando modelo y metadata...")
        model_path, metadata_path, metadata = guardar_campeon(campeon, resultados, FEATURES_MODELO, config, logger)
        metadata["speedup"] = {
            "cpus_detectadas": cpus,
            "workers_usados": workers,
            "tiempo_secuencial_s": round(tiempo_seq, 3),
            "tiempo_paralelo_s": round(tiempo_par, 3),
            "speedup": round(speedup, 3),
        }
        guardar_json(metadata, metadata_path)
        guardar_tablas_entrenamiento(metadata, resultados, campeon, FEATURES_MODELO)
        logger.info("Tablas de evidencia guardadas en reports/tables")

        logger.info("[STEP 6] Registrando experimentos...")
        if args.skip_mlflow:
            logger.info("MLflow omitido por parámetro --skip-mlflow.")
        else:
            registrar_mlflow(resultados, campeon, config, logger)

        resumen_monitor = monitor.finalizar(registros=len(X_train), estado="OK")
        guardar_monitoreo(resumen_monitor, paths.get("logs_dir", "logs"), "training")

        logger.info("======================================================================")
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE EN %.2f segundos", resumen_monitor["tiempo_total_segundos"])
        logger.info("======================================================================")

    except Exception as error:
        resumen_monitor = monitor.finalizar(estado="ERROR")
        guardar_monitoreo(resumen_monitor, paths.get("logs_dir", "logs"), "training_error")
        logger.error("Error en training_pipeline: %s", error)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
