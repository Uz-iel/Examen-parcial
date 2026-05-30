"""Pipeline de inferencia del parcial MLOps.

Flujo:
1. Lee Model Registry.
2. Carga modelo campeón.
3. Preprocesa periodo o archivo externo.
4. Alinea features con metadata del campeón.
5. Genera probabilidades.
6. Calcula TLV, deciles y réplica comercial.
"""

from __future__ import annotations

import argparse
import traceback

from src.inference import alinear_features, cargar_modelo_campeon, predecir_probabilidades
from src.monitoring import MonitorOperativo, guardar_monitoreo
from src.postprocessing import construir_scores, evaluar_offline_si_hay_target, guardar_salidas_inferencia
from src.preprocessing import preparar_dataset_inferencia
from src.registry import obtener_campeon
from src.reporting import guardar_tablas_inferencia
from src.utils import cargar_config, configurar_logger, crear_directorios


def leer_argumentos():
    """Lee argumentos de consola."""
    parser = argparse.ArgumentParser(description="Pipeline de inferencia MLOps - Parcial")
    parser.add_argument("--config", default="config/config.yaml", help="Ruta del archivo YAML de configuración")
    parser.add_argument("--period", default=None, help="Periodo a inferir. Ejemplo: 10 para p10_extrac.csv")
    parser.add_argument("--input-file", default=None, help="Ruta de CSV externo para inferencia")
    parser.add_argument("--skip-mlflow",action="store_true", help="Omitir registro MLflow en inferencia" )
    return parser.parse_args()


def main() -> None:
    """Ejecuta inferencia."""
    args = leer_argumentos()
    config = cargar_config(args.config)
    paths = config["paths"]
    crear_directorios([
        paths.get("logs_dir", "logs"),
        paths.get("output_dir", "output"),
        paths.get("best_model_dir", "best_model"),
        config["data"].get("raw_inference_dir", "data/raw/inference"),
        "data/processed",
        "reports/tables",
        "reports/figures",
    ])

    etiqueta_log = args.period or "archivo_externo"
    logger = configurar_logger(paths.get("logs_dir", "logs"), "inference")
    monitor = MonitorOperativo("inference_pipeline")

    try:
        logger.info("======================================================================")
        logger.info("PIPELINE DE INFERENCIA -- periodo/archivo %s", etiqueta_log)
        logger.info("======================================================================")

        logger.info("[STEP 1] Verificando Model Registry...")
        champion = obtener_campeon(paths.get("best_model_dir", "best_model"))
        logger.info("Campeón actual: %s (%s)", champion["model_name"], champion["version"])
        logger.info("AUC Test: %.4f | AUC OOT: %.4f | Gap: %.4f", champion["auc_test"], champion["auc_oot"], champion["auc_gap"])

        logger.info("[STEP 2] Cargando modelo campeón...")
        modelo, metadata = cargar_modelo_campeon(champion)

        logger.info("[STEP 3] Preprocesando data de inferencia...")
        X, post_df, target_real, etiqueta = preparar_dataset_inferencia(config, logger, periodo=args.period, input_file=args.input_file)

        logger.info("[STEP 4] Alineando columnas con metadata del campeón...")
        X_modelo = alinear_features(X, metadata["features"])

        logger.info("[STEP 5] Generando predicciones...")
        probabilidades = predecir_probabilidades(modelo, X_modelo)
        logger.info("Registros: %s", len(probabilidades))
        logger.info("Score promedio: %.6f", float(probabilidades.mean()))

        logger.info("[STEP 6] Calculando TLV, deciles y grupos de ejecución...")
        umbral = metadata.get("umbral_clasificacion", config["training"].get("umbral_clasificacion", 0.50))
        scores_df = construir_scores(post_df, probabilidades, config, umbral=umbral)
        scores_path, replica_path = guardar_salidas_inferencia(scores_df, champion["model_name"], etiqueta, paths.get("output_dir", "output"))
        eval_path = evaluar_offline_si_hay_target(scores_df, target_real, paths.get("output_dir", "output"), etiqueta, logger)

        resumen_monitor = monitor.finalizar(registros=len(scores_df), estado="OK")
        guardar_monitoreo(resumen_monitor, paths.get("logs_dir", "logs"), "inference")
        guardar_tablas_inferencia(scores_df, etiqueta, eval_path=eval_path, monitor=resumen_monitor)
        logger.info("Tablas de inferencia guardadas en reports/tables")
        logger.info("Latencia aproximada por registro: %.8f segundos", resumen_monitor["latencia_segundos_por_registro"])
        logger.info("Scores guardados en: %s", scores_path)
        logger.info("Réplica guardada en: %s", replica_path)
        if eval_path:
            logger.info("Evaluación offline deciles/lift: %s", eval_path)

        logger.info("======================================================================")
        logger.info("PIPELINE DE INFERENCIA COMPLETADO EN %.2f segundos", resumen_monitor["tiempo_total_segundos"])
        logger.info("======================================================================")

    except Exception as error:
        resumen_monitor = monitor.finalizar(estado="ERROR")
        guardar_monitoreo(resumen_monitor, paths.get("logs_dir", "logs"), "inference_error")
        logger.error("Error en inference_pipeline: %s", error)
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
