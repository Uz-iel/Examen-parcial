# Comandos de ejecución

## Crear entorno

```powershell
py -3.10 -m venv venv
.env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --no-cache-dir
python -c "import pandas, numpy, sklearn, joblib, yaml, mlflow; print('OK')"
```

## Opción A: prueba con data sintética

```powershell
python generate_sample_data.py --filas 8000 --periodos 10
python training_pipeline.py --config config/config_sintetico.yaml --skip-mlflow
python inference_pipeline.py --config config/config_sintetico.yaml --period 10
```

## Opción B: rápida con data real desde Git

```powershell
python training_pipeline.py --skip-mlflow
python inference_pipeline.py --period 5
```

## Opción C: completa con data real desde Git

```powershell
python training_pipeline.py --config config/config_completo.yaml --skip-mlflow
python inference_pipeline.py --config config/config_completo.yaml --period 10
```

## MLflow

```powershell
python training_pipeline.py
python -m mlflow ui --backend-store-uri sqlite:///mlflow_store/mlflow.db
```

Abrir: http://127.0.0.1:5000

## Inferencia con archivo externo

```powershell
python inference_pipeline.py --input-file "C:utarchivo_profesor.csv"
```

## Limpiar artefactos generados sin borrar data cruda

```powershell
Remove-Item -Recurse -Force .est_model, .\output, .\logs, .\data\processed, .eports	ables, .eportsigures -ErrorAction SilentlyContinue
```
