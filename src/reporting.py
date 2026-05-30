"""Tablas de evidencia para storytelling y auditoría técnica.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

import numpy as np
import pandas as pd

from src.variables import DICCIONARIO_GRUPOS_VARIABLES


NOMBRES_AMIGABLES = {
    "nro_producto_6m": "Número de productos financieros en 6 meses",
    "prom_uso_tc_rccsf3m": "Promedio uso tarjeta crédito RCC o sistema financiero",
    "ctd_sms_received": "SMS recibidos por campañas",
    "max_usotcribksf06m": "Máximo uso de tarjeta Interbank o sistema financiero",
    "ctd_camptot06m": "Cantidad total de campañas en 6 meses",
    "dsv_svppallsf06m": "Desviación de saldos o promedios últimos 6 meses",
    "prm_svprmecs06m": "Promedio saldo ECS 6 meses",
    "ctd_app_productos_m1": "Productos usados desde app en el último mes",
    "ctd_campecsm01": "Cantidad de campañas ECS en el último mes",
    "lin_tcrrstsf03m": "Línea de tarjeta registrada últimos 3 meses",
    "mnt_ptm": "Monto PTM asociado al cliente",
    "dif_no_gestionado_4meses": "Tiempo desde última no gestión",
    "max_campecs06m": "Máximo número de campañas ECS en 6 meses",
    "beta_pctusotcr12m": "Tendencia uso tarjeta 12 meses",
    "rat_disefepnm01": "Ratio disponibilidad efectiva último mes",
    "flg_saltotppe12m": "Flag de saldo total PPE 12 meses",
    "prom_sow_lintcribksf3m": "Participación promedio de línea TC 3 meses",
    "openhtml_1m": "Aperturas de email HTML último mes",
    "nprod_1m": "Número de productos último mes",
    "nro_transfer_6m": "Número de transferencias en 6 meses",
    "max_usotcrrstsf03m": "Máximo uso de tarjeta últimos 3 meses",
    "prm_cnt_fee_amt_u7d": "Promedio de comisiones últimos 7 días",
    "pas_avg6m_max12m": "Pasivo promedio 6 meses vs máximo 12 meses",
    "beta_saltotppe12m": "Tendencia temporal saldo total PPE 12 meses",
    "seg_un": "Segmento del cliente",
    "ant_ultprdallsf": "Antigüedad desde último producto",
    "avg_sald_pas_3m": "Saldo pasivo promedio 3 meses",
    "pas_1m_avg3m": "Pasivo último mes vs promedio 3 meses",
    "num_incrsaldispefe06m": "Incrementos de saldo disponible en 6 meses",
    "cnl_age_p4m_p12m": "Uso relativo de agencia 4 vs 12 meses",
    "cnl_atm_p4m_p12m": "Uso relativo de ATM 4 vs 12 meses",
    "cre_lin_tc_rccibk_m07": "Línea de crédito TC RCC Interbank mes 7",
    "prm_svprmlibdis06m": "Promedio saldo libre disponibilidad 6 meses",
    "ingreso_neto": "Ingreso neto estimado",
    "max_nact_12m": "Máximo nivel de actividad 12 meses",
    "cre_sldtotfinprm03": "Saldo total financiero promedio 3 meses",
    "dif_contacto_efectivo_10meses": "Tiempo desde contacto efectivo en 10 meses",
    "act_1m_avg3m": "Actividad último mes vs promedio 3 meses",
    "monto_consumos_ecommerce_tc": "Monto consumos ecommerce TC",
    "ctd_camptotm01": "Campañas totales último mes",
    "prop_atm_4m": "Proporción de uso ATM 4 meses",
    "prom_pct_saldopprcc6m": "Porcentaje promedio saldo préstamo RCC 6 meses",
    "apppag_1m": "Pagos por app último mes",
    "nro_configuracion_6m": "Número de configuraciones 6 meses",
    "act_avg6m_max12m": "Actividad promedio 6 meses vs máximo 12 meses",
    "sldvig_tcrsrcf": "Saldo vigente tarjeta sistema financiero",
    "prom_score_acepta_12meses": "Score histórico de aceptación 12 meses",
    "telefonos_6meses": "Teléfonos disponibles en 6 meses",
    "pas_1m_avg6m": "Pasivo último mes vs promedio 6 meses",
    "ctd_camptototrcnl06m": "Campañas por otros canales en 6 meses",
    "prm_saltotrdpj03m": "Promedio saldo total DPJ 3 meses",
    "bpitrx_1m": "Transacciones BPI último mes",
    "prm_lintcribksf03m": "Promedio línea TC 3 meses",
    "ctd_entrdm01": "Entradas digitales último mes",
    "avg_openhtml_6m": "Apertura promedio HTML 6 meses",
    "tea": "Tasa efectiva anual de la oferta",
    "pct_usotcrm01": "Porcentaje uso tarjeta último mes",
    "senthtml_1m": "Emails enviados último mes",
    "ent_1erlntcrallsfm01_INTERBANK": "Entidad financiera principal: Interbank",
    "ent_1erlntcrallsfm01_OTRO": "Entidad financiera principal: Otro",
}


def _reports_tables_dir() -> str:
    path = os.path.join("reports", "tables")
    os.makedirs(path, exist_ok=True)
    return path


def _familia_de_variable(variable: str) -> str:
    for familia, variables in DICCIONARIO_GRUPOS_VARIABLES.items():
        if variable in variables:
            return familia
    return "Otras variables"


def _nombre_amigable(variable: str) -> str:
    return NOMBRES_AMIGABLES.get(variable, variable)


def _extraer_feature_importance(modelo: Any, feature_names: list[str]) -> pd.DataFrame:
    """Extrae importancia de variables si el modelo campeón lo permite."""
    estimador = modelo
    if hasattr(modelo, "named_steps") and "model" in modelo.named_steps:
        estimador = modelo.named_steps["model"]

    if not hasattr(estimador, "feature_importances_"):
        return pd.DataFrame(columns=["variable", "nombre_amigable", "familia", "importancia"])

    importancias = np.asarray(estimador.feature_importances_, dtype=float)
    tabla = pd.DataFrame({"variable": feature_names, "importancia": importancias})
    tabla["nombre_amigable"] = tabla["variable"].map(_nombre_amigable)
    tabla["familia"] = tabla["variable"].map(_familia_de_variable)
    tabla = tabla[["variable", "nombre_amigable", "familia", "importancia"]]
    return tabla.sort_values("importancia", ascending=False).reset_index(drop=True)


def guardar_tablas_entrenamiento(metadata: dict, resultados: list[dict], campeon: dict, feature_names: list[str]) -> None:
    """Guarda tablas CSV de resultados de entrenamiento para visualización externa."""
    table_dir = _reports_tables_dir()

    kpis = pd.DataFrame([
        {
            "modelo_campeon": metadata.get("model_name"),
            "estado": metadata.get("stability"),
            "auc_train": metadata.get("metricas_train", {}).get("auc"),
            "auc_test": metadata.get("auc_test"),
            "auc_oot": metadata.get("auc_oot"),
            "auc_gap": metadata.get("auc_gap"),
            "umbral_clasificacion": metadata.get("umbral_clasificacion"),
            "stability_threshold": metadata.get("stability_threshold"),
            "n_features": metadata.get("n_features"),
            "speedup": metadata.get("speedup", {}).get("speedup"),
            "workers_usados": metadata.get("speedup", {}).get("workers_usados"),
        }
    ])
    kpis.to_csv(os.path.join(table_dir, "00_kpis_campeon.csv"), index=False)

    leaderboard_rows = []
    for r in resultados:
        leaderboard_rows.append(
            {
                "modelo": r.get("nombre"),
                "auc_train": r.get("metricas_train", {}).get("auc"),
                "auc_test": r.get("auc_test"),
                "auc_oot": r.get("auc_oot"),
                "auc_gap": r.get("auc_gap"),
                "accuracy_test": r.get("metricas_test", {}).get("accuracy"),
                "f1_test": r.get("metricas_test", {}).get("f1"),
                "tiempo_fit_s": r.get("tiempo_fit_s"),
                "stability": r.get("stability"),
                "es_campeon": r.get("nombre") == metadata.get("model_name"),
            }
        )
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("auc_test", ascending=False)
    leaderboard.to_csv(os.path.join(table_dir, "01_leaderboard_modelos.csv"), index=False)

    auc_campeon = pd.DataFrame(
        [
            {"conjunto": "Train", "auc": metadata.get("metricas_train", {}).get("auc")},
            {"conjunto": "Validación", "auc": metadata.get("auc_test")},
            {"conjunto": "OOT", "auc": metadata.get("auc_oot")},
        ]
    )
    auc_campeon.to_csv(os.path.join(table_dir, "02_auc_train_test_oot_campeon.csv"), index=False)

    fi = _extraer_feature_importance(campeon["modelo"], feature_names)
    fi.to_csv(os.path.join(table_dir, "03_importancia_variables.csv"), index=False)

    if not fi.empty:
        familias = fi.groupby("familia", as_index=False).agg(importancia=("importancia", "sum"))
        total = familias["importancia"].sum()
        familias["porcentaje"] = familias["importancia"] / total if total else 0
        familias = familias.sort_values("importancia", ascending=False)
    else:
        familias = pd.DataFrame(columns=["familia", "importancia", "porcentaje"])
    familias.to_csv(os.path.join(table_dir, "04_importancia_familias.csv"), index=False)

    speedup = metadata.get("speedup", {})
    if speedup:
        pd.DataFrame([speedup]).to_csv(os.path.join(table_dir, "05_speedup_entrenamiento.csv"), index=False)


def guardar_tablas_inferencia(scores_df: pd.DataFrame, etiqueta: str, eval_path: str | None = None, monitor: dict | None = None) -> None:
    """Guarda tablas CSV de inferencia para dashboard externo."""
    table_dir = _reports_tables_dir()

    resumen = pd.DataFrame(
        [
            {
                "etiqueta": etiqueta,
                "registros": len(scores_df),
                "score_promedio": float(scores_df["probabilidad_modelo"].mean()) if "probabilidad_modelo" in scores_df else None,
                "tlv_promedio": float(scores_df["puntuacion_tlv"].mean()) if "puntuacion_tlv" in scores_df else None,
                "grupo_1_clientes": int((scores_df.get("grupo_ejec") == 1).sum()) if "grupo_ejec" in scores_df else None,
                "tiempo_inferencia_s": monitor.get("tiempo_total_segundos") if monitor else None,
                "latencia_s_registro": monitor.get("latencia_segundos_por_registro") if monitor else None,
            }
        ]
    )
    resumen.to_csv(os.path.join(table_dir, f"06_resumen_inferencia_{etiqueta}.csv"), index=False)

    if {"grupo_ejec", "puntuacion_tlv", "probabilidad_modelo"}.issubset(scores_df.columns):
        grupos = (
            scores_df.groupby("grupo_ejec", as_index=False)
            .agg(
                clientes=("grupo_ejec", "size"),
                score_promedio=("probabilidad_modelo", "mean"),
                tlv_promedio=("puntuacion_tlv", "mean"),
            )
            .sort_values("grupo_ejec")
        )
        grupos.to_csv(os.path.join(table_dir, f"07_grupos_ejecucion_{etiqueta}.csv"), index=False)

    if eval_path and os.path.exists(eval_path):
        destino = os.path.join(table_dir, f"08_deciles_lift_{etiqueta}.csv")
        shutil.copyfile(eval_path, destino)
