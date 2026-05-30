# Proyecto Parcial MLOps — Propensión de venta bancaria

**Alumno:** Gariazzo Anarcaya, Uzziell Neyrho  
**Repositorio de código:** https://github.com/Uz-iel/Examen-parcial  
**Repositorio de datos reales:** https://github.com/Uz-iel/MLOps  

Este proyecto implementa un flujo MLOps reproducible para una campaña de venta bancaria. El repositorio principal contiene el código, configuración, evidencia ligera, logs finales, tablas y Model Registry. La data real pesada se separa en otro repositorio público con Git LFS para evitar subir CSV grandes al repositorio de código.

---

## 1. Arquitectura de repositorios

```text
Examen-parcial
├── Código del pipeline MLOps
├── Configuraciones YAML
├── README y comandos de ejecución
├── Model Registry
├── Evidencia ligera
├── Logs finales seleccionados
├── Tablas y figuras de sustentación
└── MLflow local como evidencia, si se incluye mlflow_store/mlflow.db

MLOps
└── CSV reales pesados del parcial subidos con Git LFS
```


---

## 2. Flujo general

```text
Data histórica con target
→ preprocesamiento
→ entrenamiento de modelos
→ evaluación Train/Test/OOT
→ selección del campeón
→ registro en Model Registry
→ evidencia en logs, reports y MLflow

Data nueva
→ carga del campeón
→ inferencia
→ probabilidad de compra
→ TLV
→ grupos de ejecución
→ réplica comercial
→ evaluación offline si viene target
```

---

## 3. Instalación desde cero

Se recomienda usar Python 3.10 o 3.11 y una ruta corta en Windows.

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

## 4. Data real

Los CSV reales están en:

```text
https://github.com/Uz-iel/MLOps
```

La URL base usada por el pipeline es:

```text
https://media.githubusercontent.com/media/Uz-iel/MLOps/main
```

El pipeline descarga archivos como:

```text
https://media.githubusercontent.com/media/Uz-iel/MLOps/main/p1_extrac.csv
https://media.githubusercontent.com/media/Uz-iel/MLOps/main/p2_extrac.csv
...
https://media.githubusercontent.com/media/Uz-iel/MLOps/main/p10_extrac.csv
```

En `config/config.yaml` y `config/config_completo.yaml` debe estar:

```yaml
data:
  url_base: "https://media.githubusercontent.com/media/Uz-iel/MLOps/main"
  auto_download: true
  prioridad_descarga: true
  force_download: false
```

La lógica es:

```text
1. Intenta descargar desde el repositorio MLOps.
2. Si descarga correctamente, guarda en data/raw/.
3. Si falla la descarga y existe copia local, usa la copia local.
4. Si no existe data local ni remota, muestra error claro.
5. La data sintética se usa solo como alternativa explícita.
```

---

## 5. Modos de ejecución

| Modo | Configuración | Uso |
|---|---|---|
| Rápido con data real | `config/config.yaml` | Revisión rápida. Descarga data real desde MLOps. |
| Completo con data real | `config/config_completo.yaml` | Corrida final con más periodos y más HPO. |
| Sintético | `config/config_sintetico.yaml` | Respaldo técnico si no hay acceso a data real. |

---

## 6. Ejecución rápida con data real

```powershell
python training_pipeline.py --skip-mlflow
python inference_pipeline.py --period 5 --skip-mlflow
```

Usa:

```text
Entrenamiento: p1, p2
Validación:    p3
OOT:           p4
Inferencia:    p5
```

---

## 7. Ejecución completa con data real

```powershell
python training_pipeline.py --config config/config_completo.yaml --skip-mlflow
python inference_pipeline.py --config config/config_completo.yaml --period 10 --skip-mlflow
```

Usa:

```text
Entrenamiento: p1, p2, p3, p4, p5, p6
Validación:    p7
OOT:           p8, p9
Inferencia:    p10
```

---

## 8. Ejecución alternativa con data sintética

```powershell
python generate_sample_data.py --filas 8000 --periodos 10
python training_pipeline.py --config config/config_sintetico.yaml --skip-mlflow
python inference_pipeline.py --config config/config_sintetico.yaml --period 10 --skip-mlflow
```

La data sintética no reemplaza la data real. Solo permite comprobar que el pipeline es reproducible.

---

## 9. Ejecución con MLflow

Para registrar corridas en MLflow, se debe ejecutar entrenamiento sin `--skip-mlflow`.

### Corrida rápida con MLflow

```powershell
python training_pipeline.py
python inference_pipeline.py --period 5
```

### Corrida completa con MLflow

```powershell
python training_pipeline.py --config config/config_completo.yaml
python inference_pipeline.py --config config/config_completo.yaml --period 10
```

Luego abrir la interfaz:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow_store/mlflow.db
```

Abrir en el navegador:

```text
http://127.0.0.1:5000
```

###  MLflow

En la interfaz debe aparecer el experimento:

```text
parcial_mlops_propension
```

y dentro de las corridas debe poder revisar:

```text
parámetros de configuración
modelo candidato
AUC Train
AUC Test
AUC OOT
Gap OOT
accuracy
precision
recall
F1
tiempo de entrenamiento
modelo campeón
```

La inferencia deja evidencia operativa en logs, outputs y tablas. En esta implementación, el registro principal de MLflow corresponde al entrenamiento y selección del campeón.

---

## 10. Evidencia de MLflow incluida

En el repositorio se incluye:

```text
mlflow_store/mlflow.db
```

Se puede abrir directamente la evidencia de corridas locales con:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow_store/mlflow.db
```

También se puede revisar el resumen exportado en:

```text
reports/evidence/mlflow_runs_training.csv
```

Si  se quiere generar una corrida nueva `mlflow_store/mlflow.db` , basta con ejecutar:

```powershell
python training_pipeline.py
python -m mlflow ui --backend-store-uri sqlite:///mlflow_store/mlflow.db
```

---

## 11. Pipeline de entrenamiento

Archivo principal:

```text
training_pipeline.py
```

Flujo:

```text
1. Lee YAML de configuración.
2. Crea carpetas necesarias.
3. Descarga o carga periodos de entrenamiento, validación y OOT.
4. Preprocesa variables.
5. Guarda data procesada.
6. Entrena modelos candidatos.
7. Compara entrenamiento secuencial y paralelo.
8. Usa joblib.Parallel para paralelismo por modelo.
9. Ejecuta HPO simple.
10. Calcula AUC Train, AUC Test y AUC OOT.
11. Calcula Gap OOT.
12. Selecciona campeón estable.
13. Guarda model.pkl y metadata.json.
14. Actualiza best_model/registry.json.
15. Registra en MLflow si no se usa --skip-mlflow.
16. Guarda logs, monitoreo y tablas.
```

Modelos candidatos:

```text
LogisticRegression
RandomForest
ExtraTrees
GradientBoosting
RandomForest_HPO
```

---

## 12. Pipeline de inferencia

Archivo principal:

```text
inference_pipeline.py
```

Flujo:

```text
1. Lee configuración YAML.
2. Lee best_model/registry.json.
3. Carga el modelo campeón.
4. Descarga o carga el periodo de inferencia.
5. Si existe target, lo ignora para predecir.
6. Alinea columnas con metadata del campeón.
7. Genera probabilidad_modelo.
8. Calcula TLV.
9. Asigna grupos de ejecución.
10. Guarda scores.
11. Genera réplica comercial.
12. Si hay target, calcula evaluación offline por deciles y lift.
13. Guarda logs y monitoreo.
```

Comando con periodo:

```powershell
python inference_pipeline.py --period 5 --skip-mlflow
```

Comando con archivo externo:

```powershell
python inference_pipeline.py --input-file "C:\ruta\archivo_profesor.csv" --skip-mlflow
```

---

## 13. Model Registry

El modelo campeón se registra en:

```text
best_model/registry.json
```

El registry contiene:

```text
modelo campeón
versión
fecha
métricas
ruta del modelo
metadata
features usadas
historial
```

La inferencia no reentrena. Siempre carga el campeón desde el registry.

---

## 14. Métricas Train/Test/OOT

| Métrica | Significado |
|---|---|
| AUC Train | Aprendizaje en datos de entrenamiento. |
| AUC Test | Generalización en validación. |
| AUC OOT | Estabilidad en periodos posteriores. |
| Gap OOT | Diferencia absoluta entre AUC Test y AUC OOT. |

Interpretación:

```text
Buen ajuste:
AUC Train, Test y OOT altos y cercanos.

Overfitting:
AUC Train alto, pero Test/OOT menor.

Degradación temporal:
AUC Test bueno, pero AUC OOT cae mucho.

Underfitting:
AUC Train, Test y OOT bajos.
```

Criterio de estabilidad:

```text
abs(AUC Test - AUC OOT) <= 0.03
```

---

## 15. Diccionario de variables: 69 campos a 60 features

El dataset original tiene 69 campos. No todos entran como predictores.

Se excluyen del modelo:

```text
partition
tip_doc
key_value
codunicocli
monto
prob_value_contact
grp_campecs06m
fch_creacion
p_fecinformacion
target
```

Motivo:

| Campo | Uso |
|---|---|
| `target` | Variable objetivo. |
| `key_value`, `codunicocli` | Identificadores y trazabilidad. |
| `partition`, fechas | Trazabilidad temporal. |
| `monto` | Se usa para TLV. |
| `prob_value_contact` | Se usa para contactabilidad en TLV. |
| `grp_campecs06m` | Se transforma en frescura. |

Resultado:

```text
58 variables numéricas
+ 2 dummies de ent_1erlntcrallsfm01
= 60 features finales
```

---

## 16. TLV

TLV es una puntuación derivada de negocio. No es una columna original.

```text
puntuacion_tlv =
probabilidad_modelo
× prob_value_contact
× log(monto + 1)
× frescura
```

Frescura:

```yaml
G1: 0.066
G2: 0.028
G3: 0.022
G4: 0.008
G5: 0.004
OTRO: 0.004
```

La probabilidad responde:

```text
¿quién tiene mayor probabilidad de comprar?
```

El TLV responde:

```text
¿a quién conviene contactar primero considerando probabilidad, contacto, monto y frescura?
```

---

## 17. Logs, tablas, figuras y evidencia

El proyecto deja evidencia en:

```text
logs/evidence/
output/evidence/
reports/evidence/
reports/tables/
reports/figures_storytelling/
reports/tables_storytelling/
mlflow_store/mlflow.db
```

Qué revisar:

```text
logs/evidence/training_final.log
logs/evidence/inference_final.log
logs/evidence/monitoring_training_final.json
logs/evidence/monitoring_inference_final.json
reports/evidence/mlflow_runs_training.csv
reports/tables/01_leaderboard_modelos.csv
reports/tables/02_auc_train_test_oot_campeon.csv
reports/tables/08_deciles_lift_10.csv
output/evidence/scores_10_sample.csv
output/evidence/replica_GradientBoosting_10_sample.txt
```

Las salidas completas se regeneran ejecutando los pipelines. No se suben CSV reales ni outputs pesados completos.

---

## 18. Funciones principales

| Archivo | Función |
|---|---|
| `training_pipeline.py` | Orquesta entrenamiento completo. |
| `inference_pipeline.py` | Orquesta inferencia con modelo campeón. |
| `generate_sample_data.py` | Genera data sintética compatible. |
| `src/preprocessing.py` | Carga, descarga, valida y transforma datos. |
| `src/training.py` | Entrena modelos, usa joblib, ejecuta HPO y selecciona campeón. |
| `src/inference.py` | Carga modelo y genera probabilidades. |
| `src/postprocessing.py` | Calcula TLV, grupos, réplica y lift offline. |
| `src/registry.py` | Administra Model Registry local. |
| `src/metrics.py` | Calcula métricas ML y lift. |
| `src/monitoring.py` | Registra CPU, memoria, tiempos y latencia. |
| `src/reporting.py` | Exporta tablas de evidencia. |
| `src/variables.py` | Centraliza variables y explicación 69 a 60. |
| `src/utils.py` | Utilidades para YAML, JSON, carpetas y logs. |

---

## 19. Errores frecuentes


### MLflow no muestra corridas

Verificar que exista:

```text
mlflow_store/mlflow.db
```

Abrir con:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow_store/mlflow.db
```

Si no hay corridas, generar una nueva:

```powershell
python training_pipeline.py
```

### Ruta larga en Windows

Usar una carpeta corta:

```text
C:\mlops\Examen-parcial
```

---

## 20. Comandos finales 

```powershell
git clone https://github.com/Uz-iel/Examen-parcial.git
cd Examen-parcial

py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt --no-cache-dir

python training_pipeline.py --skip-mlflow
python inference_pipeline.py --period 5 --skip-mlflow
```

Para revisar MLflow:

```powershell
python training_pipeline.py
python -m mlflow ui --backend-store-uri sqlite:///mlflow_store/mlflow.db
```

---

## 21. Conclusion

> El proyecto implementa un flujo MLOps completo y reproducible y el repositorio principal contiene código, configuración y evidencia ligera. La data real pesada está separada en el repositorio `MLOps` con Git LFS. El entrenamiento descarga datos reales, evalúa modelos en Train/Test/OOT, selecciona un campeón estable y lo registra en un Model Registry. La inferencia carga ese campeón, genera probabilidades, calcula TLV y produce grupos de ejecución. Además, el proyecto deja logs, tablas, monitoreo operativo y evidencia MLflow para auditoría.
