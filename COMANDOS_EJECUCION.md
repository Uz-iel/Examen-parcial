# Comandos de ejecución — Parcial MLOps

## Instalación

```powershell
git clone https://github.com/Uz-iel/Examen-parcial.git
cd Examen-parcial

py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt --no-cache-dir
```

## Modo rápido por defecto con data real

```powershell
python training_pipeline.py --skip-mlflow
python inference_pipeline.py --period 5 --skip-mlflow
```

## Modo completo con data real

```powershell
python training_pipeline.py --config config/config_completo.yaml --skip-mlflow
python inference_pipeline.py --config config/config_completo.yaml --period 10 --skip-mlflow
```

## Modo alternativo con data sintética

```powershell
python generate_sample_data.py --filas 8000 --periodos 10
python training_pipeline.py --config config/config_sintetico.yaml --skip-mlflow
python inference_pipeline.py --config config/config_sintetico.yaml --period 10 --skip-mlflow
```

## MLflow

```powershell
python training_pipeline.py
python inference_pipeline.py --period 5
python -m mlflow ui --backend-store-uri sqlite:///mlflow_store/mlflow.db
```


## Nota sobre data real

Las configuraciones `config.yaml` y `config_completo.yaml` tienen `prioridad_descarga: true`.
Eso significa que intentan descargar primero desde la URL configurada en `url_base`.

Si la URL no contiene los CSV reales, usar una de estas opciones:

```powershell
# Alternativa sintética
python generate_sample_data.py --filas 8000 --periodos 10
python training_pipeline.py --config config/config_sintetico.yaml --skip-mlflow
python inference_pipeline.py --config config/config_sintetico.yaml --period 10 --skip-mlflow
```

o colocar manualmente los CSV en `data/raw/training/` y `data/raw/inference/`.
