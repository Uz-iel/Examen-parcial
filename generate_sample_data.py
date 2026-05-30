"""Genera data sintética 
0

La data sintética sirve solo para probar reproducibilidad técnica.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.variables import TARGET, VARIABLE_CATEGORICA, VARIABLES_NUMERICAS


def leer_argumentos():
    parser = argparse.ArgumentParser(description="Generador de data sintética para el parcial MLOps")
    parser.add_argument("--filas", type=int, default=8000, help="Filas por periodo")
    parser.add_argument("--periodos", type=int, default=10, help="Cantidad de periodos p1..pN")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible")
    parser.add_argument("--inferencia-con-target", action="store_true", help="Incluye target en el último periodo para evaluación offline")
    return parser.parse_args()


def hash_id(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest().upper()


def generar_periodo(periodo: int, filas: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame()
    df["partition"] = f"p{periodo}"
    df["tip_doc"] = "DNI"
    base_cliente = periodo * 1_000_000
    df["codunicocli"] = np.arange(base_cliente, base_cliente + filas)
    df["key_value"] = [hash_id(f"p{periodo}_{x}") for x in df["codunicocli"]]

    grupos = np.array(["G1", "G2", "G3", "G4", "G5"])
    df["grp_campecs06m"] = rng.choice(grupos, size=filas, p=[0.12, 0.18, 0.28, 0.22, 0.20])
    df["prob_value_contact"] = np.clip(rng.beta(2, 5, size=filas), 0.01, 0.99)
    df["monto"] = np.round(rng.gamma(shape=2.2, scale=9000, size=filas), 2)
    df[VARIABLE_CATEGORICA] = rng.choice(["INTERBANK", "OTRO"], size=filas, p=[0.55, 0.45])

    # Variables numéricas con señales simples.
    for col in VARIABLES_NUMERICAS:
        df[col] = rng.normal(loc=0, scale=1, size=filas)

    # Dar escala a algunas variables de negocio.
    df["ingreso_neto"] = np.round(rng.gamma(3, 1200, filas), 2)
    df["tea"] = np.round(rng.uniform(0.08, 0.55, filas), 4)
    df["ctd_camptot06m"] = rng.poisson(3, filas)
    df["ctd_sms_received"] = rng.poisson(2, filas)
    df["telefonos_6meses"] = rng.integers(0, 4, filas)
    df["prom_score_acepta_12meses"] = rng.uniform(0, 1, filas)
    df["monto_consumos_ecommerce_tc"] = np.round(rng.gamma(2, 500, filas), 2)
    df["nro_producto_6m"] = rng.poisson(2, filas)
    df["pct_usotcrm01"] = rng.uniform(0, 1, filas)

    # Target sintético con tasa baja, parecida a campañas reales.
    grupo_bonus = df["grp_campecs06m"].map({"G1": 1.2, "G2": 0.7, "G3": 0.25, "G4": -0.3, "G5": -0.7}).astype(float)
    logit = (
        -6.0
        + 1.1 * df["prom_score_acepta_12meses"]
        + 0.45 * df["prob_value_contact"]
        + 0.00007 * df["monto"].clip(0, 60000)
        + 0.15 * df["nro_producto_6m"]
        + grupo_bonus
        + rng.normal(0, 0.6, filas)
    )
    prob = 1 / (1 + np.exp(-logit))
    df[TARGET] = rng.binomial(1, np.clip(prob, 0.001, 0.7))

    fecha_base = datetime(2026, 1, 1) + timedelta(days=30 * periodo)
    df["fch_creacion"] = fecha_base.strftime("%Y-%m-%d")
    df["p_fecinformacion"] = fecha_base.strftime("%Y%m")
    return df


def main():
    args = leer_argumentos()
    rng = np.random.default_rng(args.seed)
    train_dir = os.path.join("data", "raw", "training")
    inf_dir = os.path.join("data", "raw", "inference")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(inf_dir, exist_ok=True)

    for periodo in range(1, args.periodos + 1):
        df = generar_periodo(periodo, args.filas, rng)
        nombre = f"p{periodo}_extrac.csv"
        if periodo < args.periodos:
            path = os.path.join(train_dir, nombre)
        else:
            path = os.path.join(inf_dir, nombre)
            if not args.inferencia_con_target and TARGET in df.columns:
                df = df.drop(columns=[TARGET])
        df.to_csv(path, index=False)
        print(f"Generado {path}: {df.shape}")

    print("Data sintética generada correctamente.")


if __name__ == "__main__":
    main()
