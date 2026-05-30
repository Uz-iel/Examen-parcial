# Resumen de evidencia

Este archivo se actualiza de forma conceptual. Los valores exactos deben leerse desde:

- `logs/training_*.log`
- `logs/inference_*.log`
- `reports/tables/00_kpis_campeon.csv`
- `reports/tables/01_leaderboard_modelos.csv`
- `reports/tables/05_speedup_entrenamiento.csv`

El speed-up no se deja como número fijo porque depende del hardware. Se calcula como:

```text
speedup = tiempo_secuencial / tiempo_paralelo
```

El campeón se selecciona por AUC Test entre modelos estables según OOT.
