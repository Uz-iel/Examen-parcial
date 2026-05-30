# Proyecto Parcial MLOps - Propensión de venta bancaria

Alumno: **Gariazzo Anarcaya, Uzziell Neyrho**

Este repositorio implementa un pipeline MLOps reproducible para campañas de venta bancaria. El flujo se ejecuta por consola y cubre: ingesta, preprocesamiento, entrenamiento con HPO, evaluación temporal, Model Registry, inferencia, TLV, réplica de campaña y monitoreo operativo.

---

## 1. Modos de ejecución

El proyecto tiene tres modos claros:

| Modo | Configuración | Uso |
|---|---|---|
| Data sintética | `config/config_sintetico.yaml` | Prueba técnica si no se tiene la data real. |
| Rápido con data real | `config/config.yaml` | Demo/revisión local. Descarga desde Git si no encuentra CSV locales. |
| Completo con data real | `config/config_completo.yaml` | Corrida final con más periodos y más filas. |

El código es el mismo. Solo cambia el YAML.

---

## 2. Estructura principal

```text
Proyecto_Parcial_MLOps_Gariazzo_FINAL/
├── training_pipeline.py
├── inference_pipeline.py
├── generate_sample_data.py
├── requirements.txt
├── COMANDOS_EJECUCION.md
├── config/
│   ├── config.yaml
│   ├── config_completo.yaml
│   └── config_sintetico.yaml
├── data/
│   ├── raw/training/
│   ├── raw/inference/
│   └── processed/
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── inference.py
│   ├── postprocessing.py
│   ├── registry.py
│   ├── metrics.py
│   ├── monitoring.py
│   ├── reporting.py
│   ├── variables.py
│   └── utils.py
├── best_model/
├── output/
├── logs/
└── reports/
```

---

## 3. Instalación

Usar Python 3.10 o 3.11.

```powershell
py -3.10 -m venv venv
.env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --no-cache-dir
```

Verificación:

```powershell
python -c "import pandas, numpy, sklearn, joblib, yaml, mlflow; print('OK')"
```

Si Windows muestra error por ruta larga, ejecutar desde una ruta corta, por ejemplo `C:\mlops\parcial`.

---

## 4. Opción A: prueba con data sintética

Sirve si el profesor quiere validar el código sin descargar la data real.

```powershell
python generate_sample_data.py --filas 8000 --periodos 10
python training_pipeline.py --config config/config_sintetico.yaml --skip-mlflow
python inference_pipeline.py --config config/config_sintetico.yaml --period 10
```

La data sintética se crea en:

```text
data/raw/training/p1_extrac.csv ... p9_extrac.csv
data/raw/inference/p10_extrac.csv
```

Importante: la data sintética no reemplaza el caso real; solo prueba reproducibilidad técnica.

---

## 5. Opción B: corrida rápida con data real desde Git

Si los CSV no existen localmente, el pipeline intenta descargarlos desde el Git configurado en `url_base`.

```powershell
python training_pipeline.py --skip-mlflow
python inference_pipeline.py --period 5
```

Esta opción usa:

```text
config/config.yaml
entrenamiento: p1, p2
validación: p3
OOT: p4
inferencia: p5
```

---

## 6. Opción C: corrida completa con data real desde Git

```powershell
python training_pipeline.py --config config/config_completo.yaml --skip-mlflow
python inference_pipeline.py --config config/config_completo.yaml --period 10
```

Esta opción usa:

```text
config/config_completo.yaml
entrenamiento: p1 a p6
validación: p7
OOT: p8, p9
inferencia: p10
```

---

## 7. Si se desea colocar CSV manualmente

Colocar archivos históricos con target en:

```text
data/raw/training/p1_extrac.csv
data/raw/training/p2_extrac.csv
...
```

Colocar archivo nuevo de inferencia en:

```text
data/raw/inference/p10_extrac.csv
```

También se puede inferir un archivo externo:

```powershell
python inference_pipeline.py --input-file "C:
utarchivo_profesor.csv"
```

---

## 8. Lógica del entrenamiento

1. Lee configuración YAML.
2. Carga periodos de entrenamiento, validación y OOT.
3. Preprocesa variables.
4. Guarda artefactos en `data/processed/`.
5. Entrena modelos base.
6. Compara entrenamiento secuencial y paralelo con `joblib.Parallel`.
7. Ejecuta HPO sobre RandomForest.
8. Calcula AUC Train, AUC Test y AUC OOT.
9. Selecciona campeón estable según `abs(AUC_test - AUC_oot) <= 0.03`.
10. Guarda `model.pkl`, `metadata.json` y actualiza `best_model/registry.json`.
11. Registra en MLflow si no se usa `--skip-mlflow`.

---

## 9. AUC Train, Test y OOT

| Métrica | Qué mide | Lectura |
|---|---|---|
| AUC Train | Aprendizaje sobre data de entrenamiento | Si es alto, el modelo aprendió patrones. |
| AUC Test | Generalización en validación | Es la métrica principal de comparación. |
| AUC OOT | Estabilidad en periodos futuros | Simula producción. |
| Gap OOT | `abs(AUC_test - AUC_oot)` | Si es bajo, el modelo es estable. |

Interpretación:

```text
Buen ajuste: AUC Train, Test y OOT altos y cercanos.
Overfitting: AUC Train muy alto, pero Test/OOT baja.
Degradación temporal: Test bueno, pero OOT cae mucho.
Underfitting: Train, Test y OOT bajos.
```

---

## 10. ¿Por qué el diccionario tiene 69 campos y el modelo usa 60 features?

El dataset original tiene 69 campos. No todos deben entrar como predictores.

Se excluyen como features directas:

```text
partition, tip_doc, key_value, codunicocli, monto,
prob_value_contact, grp_campecs06m, fch_creacion,
p_fecinformacion, target
```

Motivo:

- `target` es la variable objetivo.
- `key_value` y `codunicocli` son identificadores.
- `partition` y fechas son trazabilidad.
- `monto`, `prob_value_contact` y `grp_campecs06m` se conservan para TLV, no para entrenar propensión.

Resultado:

```text
58 variables numéricas del diccionario
+ 2 dummies de ent_1erlntcrallsfm01
= 60 features finales del modelo
```

---

## 11. TLV y frescura

TLV significa **puntuación de valor comercial esperado**. No es una columna original del dataset; es una variable derivada en inferencia.

Fórmula:

```text
puntuacion_tlv = probabilidad_modelo * prob_value_contact * log(monto + 1) * frescura
```

Componentes:

| Componente | Uso |
|---|---|
| `probabilidad_modelo` | Propensión estimada por el modelo. |
| `prob_value_contact` | Facilidad/probabilidad de contacto. |
| `log(monto + 1)` | Valor económico suavizado. |
| `frescura` | Factor original por grupo de campaña. |

Frescura original usada:

```yaml
G1: 0.066
G2: 0.028
G3: 0.022
G4: 0.008
G5: 0.004
```

El TLV no cambia el entrenamiento. Solo ordena la salida comercial después de predecir.

---

## 12. Inferencia

1. Lee `best_model/registry.json`.
2. Carga el campeón activo.
3. Lee `p{period}_extrac.csv` o archivo externo.
4. Si encuentra `target`, lo ignora para predecir.
5. Alinea columnas con `metadata.json`.
6. Genera `probabilidad_modelo`.
7. Calcula TLV.
8. Asigna grupo de ejecución 1 a 10.
9. Guarda scores y réplica.
10. Si había target, calcula evaluación offline por merge de cliente, no por posición.

Salidas:

```text
output/scores_{period}.csv
output/replica_{modelo}_{period}.txt
output/evaluacion_offline_deciles_lift_{period}.csv  # solo si hay target
```

---

## 13. MLflow

Con tracking:

```powershell
python training_pipeline.py
python -m mlflow ui --backend-store-uri sqlite:///mlflow_store/mlflow.db
```

MLflow registra métricas y parámetros. El campeón oficial queda en `best_model/registry.json`.

---

## 14. Funciones principales

| Archivo | Función |
|---|---|
| `preprocessing.py` | Carga CSV, descarga desde Git si falta, limpia nulos, genera 60 features y guarda processed. |
| `training.py` | Define modelos, entrena con joblib, ejecuta HPO, calcula AUC Train/Test/OOT y selecciona campeón. |
| `inference.py` | Carga modelo campeón y genera probabilidades. |
| `postprocessing.py` | Calcula TLV, grupos de ejecución, réplica y lift offline. |
| `registry.py` | Mantiene champion e historial en `registry.json`. |
| `metrics.py` | Calcula AUC, accuracy, precision, recall, F1 y lift. |
| `monitoring.py` | Guarda tiempo, CPU, memoria y latencia. |
| `reporting.py` | Exporta tablas para dashboard/notebook externo. |
| `variables.py` | Centraliza variables, diccionario funcional y explicación 69 → 60. |

---
