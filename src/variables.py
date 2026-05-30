"""Definición  de variables del proyecto.

"""

TARGET = "target"
VARIABLE_CATEGORICA = "ent_1erlntcrallsfm01"

COLUMNAS_POST = [
    "partition",
    "key_value",
    "codunicocli",
    "grp_campecs06m",
    "prob_value_contact",
    "monto",
]

VARIABLES_NUMERICAS = [
    "nro_producto_6m",
    "prom_uso_tc_rccsf3m",
    "ctd_sms_received",
    "max_usotcribksf06m",
    "ctd_camptot06m",
    "dsv_svppallsf06m",
    "prm_svprmecs06m",
    "ctd_app_productos_m1",
    "ctd_campecsm01",
    "lin_tcrrstsf03m",
    "mnt_ptm",
    "dif_no_gestionado_4meses",
    "max_campecs06m",
    "beta_pctusotcr12m",
    "rat_disefepnm01",
    "flg_saltotppe12m",
    "prom_sow_lintcribksf3m",
    "openhtml_1m",
    "nprod_1m",
    "nro_transfer_6m",
    "max_usotcrrstsf03m",
    "prm_cnt_fee_amt_u7d",
    "pas_avg6m_max12m",
    "beta_saltotppe12m",
    "seg_un",
    "ant_ultprdallsf",
    "avg_sald_pas_3m",
    "pas_1m_avg3m",
    "num_incrsaldispefe06m",
    "cnl_age_p4m_p12m",
    "cnl_atm_p4m_p12m",
    "cre_lin_tc_rccibk_m07",
    "prm_svprmlibdis06m",
    "ingreso_neto",
    "max_nact_12m",
    "cre_sldtotfinprm03",
    "dif_contacto_efectivo_10meses",
    "act_1m_avg3m",
    "monto_consumos_ecommerce_tc",
    "ctd_camptotm01",
    "prop_atm_4m",
    "prom_pct_saldopprcc6m",
    "apppag_1m",
    "nro_configuracion_6m",
    "act_avg6m_max12m",
    "sldvig_tcrsrcf",
    "prom_score_acepta_12meses",
    "telefonos_6meses",
    "pas_1m_avg6m",
    "ctd_camptototrcnl06m",
    "prm_saltotrdpj03m",
    "bpitrx_1m",
    "prm_lintcribksf03m",
    "ctd_entrdm01",
    "avg_openhtml_6m",
    "tea",
    "pct_usotcrm01",
    "senthtml_1m",
]

FEATURES_MODELO = VARIABLES_NUMERICAS + [
    "ent_1erlntcrallsfm01_INTERBANK",
    "ent_1erlntcrallsfm01_OTRO",
]

DICCIONARIO_GRUPOS_VARIABLES = {
    "Identificación y negocio": ["partition", "key_value", "codunicocli", "monto", "prob_value_contact", "grp_campecs06m"],
    "Ingreso, tasa y valor económico": ["tea", "ingreso_neto", "mnt_ptm"],
    "Variable categórica principal": ["ent_1erlntcrallsfm01", "ent_1erlntcrallsfm01_INTERBANK", "ent_1erlntcrallsfm01_OTRO"],
    "Productos y actividad": ["ant_ultprdallsf", "nro_configuracion_6m", "nro_transfer_6m", "nro_producto_6m", "ctd_app_productos_m1", "act_avg6m_max12m", "act_1m_avg3m", "max_nact_12m", "nprod_1m", "apppag_1m", "seg_un"],
    "Campañas y contacto": ["ctd_campecsm01", "max_campecs06m", "ctd_camptotm01", "ctd_camptot06m", "ctd_camptototrcnl06m", "ctd_sms_received", "telefonos_6meses", "prom_score_acepta_12meses", "dif_no_gestionado_4meses", "dif_contacto_efectivo_10meses", "senthtml_1m", "openhtml_1m", "avg_openhtml_6m"],
    "Canales y transacciones": ["prm_cnt_fee_amt_u7d", "prop_atm_4m", "cnl_atm_p4m_p12m", "cnl_age_p4m_p12m", "bpitrx_1m", "ctd_entrdm01", "num_incrsaldispefe06m"],
    "Tarjeta, línea y crédito": ["cre_lin_tc_rccibk_m07", "lin_tcrrstsf03m", "max_usotcribksf06m", "max_usotcrrstsf03m", "prm_lintcribksf03m", "prom_sow_lintcribksf3m", "prom_uso_tc_rccsf3m", "beta_pctusotcr12m", "pct_usotcrm01"],
    "Saldos, deuda y consumo": ["sldvig_tcrsrcf", "cre_sldtotfinprm03", "monto_consumos_ecommerce_tc", "avg_sald_pas_3m", "pas_avg6m_max12m", "pas_1m_avg6m", "pas_1m_avg3m", "prom_pct_saldopprcc6m", "dsv_svppallsf06m", "prm_svprmlibdis06m", "prm_svprmecs06m", "flg_saltotppe12m", "beta_saltotppe12m", "prm_saltotrdpj03m", "rat_disefepnm01"],
    "Fechas y trazabilidad": ["fch_creacion", "p_fecinformacion"],
}

COLUMNAS_EXCLUIDAS_MODELO = [
    "partition", "tip_doc", "key_value", "codunicocli", "monto",
    "prob_value_contact", "grp_campecs06m", "fch_creacion", "p_fecinformacion", TARGET,
]

COLUMNAS_NEGOCIO_TLV = ["monto", "prob_value_contact", "grp_campecs06m"]
COLUMNAS_IDENTIFICACION = ["key_value", "codunicocli"]

EXPLICACION_69_A_60 = (
    "El diccionario original tiene 69 campos. El modelo no usa identificadores, target, fechas "
    "ni variables de negocio del TLV como features directas. Quedan 58 variables numéricas "
    "y 2 dummies de ent_1erlntcrallsfm01, total 60 features finales."
)
