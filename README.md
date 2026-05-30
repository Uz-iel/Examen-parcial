**Alumno:** Gariazzo Anarcaya, Uzziell Neyrho  
**Repositorio:** https://github.com/Uz-iel/Examen-parcial

Este proyecto implementa un pipeline MLOps reproducible para una campaña de venta bancaria.  
El flujo se ejecuta por consola y cubre ingesta, preprocesamiento, entrenamiento con HPO, evaluación temporal, Model Registry, inferencia, TLV, réplica de campaña y monitoreo operativo.

---

## 1. Modos de ejecución

El proyecto tiene tres modos:

| Modo | Configuración | Uso |
|---|---|---|
| Rápido con data real | `config/config.yaml` | Modo por defecto. Busca CSV locales y, si faltan, intenta descargarlos desde Git. |
| Completo con data real | `config/config_completo.yaml` | Corrida final con más periodos, más filas y HPO mayor. |
| Sintético | `config/config_sintetico.yaml` | Alternativa para validar el código si no hay acceso a la data real. |

El código es el mismo. Cambia únicamente el YAML.

---

## 2. Instalación

```powershell
git clone https://github.com/Uz-iel/Examen-parcial.git
cd Examen-parcial

py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt --no-cache-dir
```

Verificación:

```powershell
python -c "import pandas, numpy, sklearn, joblib, yaml, mlflow; print('OK entorno listo')"
```

---

## 3. Data real y descarga automática

Los CSV reales pueden no estar incluidos en Git por tamaño.

El pipeline rápido y completo usan **prioridad de descarga desde el repositorio**.

```text
1. Intenta descargar el CSV desde url_base.
2. Si la descarga funciona, guarda el archivo en data/raw/ y lo usa.
3. Si la descarga falla y existe copia local, usa la copia local como respaldo.
4. Si no hay descarga ni copia local, muestra error claro.
5. Como alternativa explícita, se puede ejecutar el modo sintético.
```

La data sintética no se usa automáticamente en las corridas real/rápida/completa.

Para que la descarga automática funcione, los archivos deben existir en la URL configurada:

```text
https://media.githubusercontent.com/media/Uz-iel/MLOps/main/p1_extrac.csv
https://media.githubusercontent.com/media/Uz-iel/MLOps/main/p2_extrac.csv
...
https://media.githubusercontent.com/media/Uz-iel/MLOps/main/p10_extrac.csv
```

Si esos archivos no están publicados en esa ruta, aparecerá `HTTP Error 404`. Para que la descarga por defecto funcione, los CSV reales deben estar publicados en el repositorio o en una ubicación compatible con `url_base`. Si no se publican, se debe colocar la data manualmente o usar el modo sintético.

---

## 4. Modo rápido por defecto con data real

Este es el modo recomendado para que el profesor clone y pruebe rápido.

```powershell
python training_pipeline.py --skip-mlflow
python inference_pipeline.py --period 5 --skip-mlflow
```

Usa:

```text
config/config.yaml
Entrenamiento: p1, p2
Validación: p3
OOT: p4
Inferencia: p5
```

---

## 5. Corrida completa con data real

```powershell
python training_pipeline.py --config config/config_completo.yaml --skip-mlflow
python inference_pipeline.py --config config/config_completo.yaml --period 10 --skip-mlflow
```

Usa:

```text
config/config_completo.yaml
Entrenamiento: p1, p2, p3, p4, p5, p6
Validación: p7
OOT: p8, p9
Inferencia: p10
```

---

## 6. Alternativa con data sintética

Este modo se usa solo si no hay acceso a los CSV reales.

```powershell
python generate_sample_data.py --filas 8000 --periodos 10
python training_pipeline.py --config config/config_sintetico.yaml --skip-mlflow
python inference_pipeline.py --config config/config_sintetico.yaml --period 10 --skip-mlflow
```

La data sintética no reemplaza el caso real. Solo permite validar la ejecución técnica del pipeline.

---

## 7. Colocar CSV manualmente

CSV históricos con target:

```text
data/raw/training/p1_extrac.csv
data/raw/training/p2_extrac.csv
...
data/raw/training/p9_extrac.csv
```

CSV nuevo para inferencia:

```text
data/raw/inference/p10_extrac.csv
```

Archivo externo con cualquier nombre:

```powershell
python inference_pipeline.py --input-file "C:\ruta\archivo_profesor.csv" --skip-mlflow
```

---

## 8. Pipeline de entrenamiento

```text
1. Lee configuración YAML.
2. Carga periodos de entrenamiento, validación y OOT.
3. Preprocesa variables.
4. Guarda datasets procesados.
5. Entrena modelos base.
6. Usa joblib.Parallel para paralelismo por modelo.
7. Ejecuta HPO simple.
8. Calcula AUC Train, AUC Test y AUC OOT.
9. Calcula gap de estabilidad temporal.
10. Selecciona campeón.
11. Guarda model.pkl, metadata.json y registry.json.
12. Guarda logs, tablas y monitoreo operativo.
```

---

## 9. AUC Train, AUC Test y AUC OOT

| Métrica | Qué mide |
|---|---|
| AUC Train | Aprendizaje sobre datos de entrenamiento. |
| AUC Test | Generalización en validación. |
| AUC OOT | Estabilidad en periodos futuros. |
| Gap OOT | `abs(AUC_test - AUC_oot)`. |

Lectura:

```text
Buen ajuste: AUC Train, Test y OOT altos y cercanos.
Overfitting: AUC Train alto, pero Test/OOT bajo.
Degradación temporal: Test bueno, pero OOT cae.
Underfitting: Train, Test y OOT bajos.
```

---

## 10. Inferencia

```text
1. Lee best_model/registry.json.
2. Carga el modelo campeón.
3. Lee p{period}_extrac.csv o archivo externo.
4. Ignora target si existe.
5. Alinea columnas con metadata.json.
6. Genera probabilidad_modelo.
7. Calcula TLV.
8. Asigna grupos de ejecución 1 a 10.
9. Guarda scores, réplica y monitoreo.
10. Si hay target, calcula lift offline por merge de cliente.
```

---

## 11. Diccionario de datos: 69 campos a 60 features

Se excluyen como predictores directos:

```text
partition, tip_doc, key_value, codunicocli, monto,
prob_value_contact, grp_campecs06m, fch_creacion,
p_fecinformacion, target
```

Resultado:

```text
58 variables numéricas
+ 2 dummies de ent_1erlntcrallsfm01
= 60 features finales
```

---

## 12. TLV

TLV es una puntuación derivada de negocio, no una columna original.

```text
puntuacion_tlv = probabilidad_modelo * prob_value_contact * log(monto + 1) * frescura
```

Frescura usada:

```yaml
G1: 0.066
G2: 0.028
G3: 0.022
G4: 0.008
G5: 0.004
OTRO: 0.004
```

El TLV no cambia el entrenamiento. Solo prioriza clientes en inferencia.

---

## 13. MLflow

Con MLflow:

```powershell
python training_pipeline.py
python inference_pipeline.py --period 5
python -m mlflow ui --backend-store-uri sqlite:///mlflow_store/mlflow.db
```

Abrir:

```text
http://127.0.0.1:5000
```

Si se usa `--skip-mlflow`, se omite MLflow, pero se mantienen logs, registry, tablas y monitoreo local.

---

## 14. Evidencia incluida

El repositorio puede incluir evidencia ligera en:

```text
logs/evidence/
reports/evidence/
reports/tables/
output/evidence/
best_model/registry.json
```

Las salidas completas se regeneran ejecutando el pipeline.
