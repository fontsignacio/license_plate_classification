# License Plate Classification

## Descripción
Este proyecto implementa un clasificador de imágenes basado en deep learning para detectar matrículas vehiculares (license plates) en imágenes. Utiliza modelos preentrenados (VGG16 y MobileNetV2) y se entrena sobre un dataset sintético de matrículas indias descargado automáticamente desde Kaggle.

El flujo principal descarga los datos, prepara la estructura de carpetas, entrena los modelos y guarda los resultados y métricas en la carpeta `output/`.

## Requisitos
- Docker y Docker Compose instalados (se recomienda GPU para acelerar el entrenamiento).
- Credenciales de Kaggle para descargar el dataset.

## Estructura esperada del archivo `.env`
Debes crear un archivo `.env` en la raíz del proyecto con el siguiente contenido (reemplaza con tus credenciales de Kaggle):

```
KAGGLE_USERNAME=tu_usuario_kaggle
KAGGLE_KEY=tu_api_key_kaggle
```

## Instalación y ejecución con Docker
1. **Clona el repositorio y entra a la carpeta del proyecto:**
   ```bash
   git clone <url-del-repo>
   cd license_plate_classification
   ```

2. **Crea el archivo `.env` con las credenciales de Kaggle:**
   ```bash
   echo "KAGGLE_USERNAME=tu_usuario_kaggle" > .env
   echo "KAGGLE_KEY=tu_api_key_kaggle" >> .env
   ```

3. **Levanta el entorno con Docker Compose:**
   ```bash
   docker compose up --build
   ```
   Esto descargará las dependencias, el dataset, preparará los datos y entrenará los modelos automáticamente.

4. **Resultados:**
   - Los modelos entrenados y las métricas se guardarán en la carpeta `output/`.
   - Los datos descargados y procesados estarán en la carpeta `data/`.

## Notas adicionales
- El entrenamiento utiliza aceleración por GPU si está disponible (requiere drivers y runtime NVIDIA configurados).
- Puedes modificar los hiperparámetros y la arquitectura en el archivo `train.py`.
- El script principal que orquesta todo el flujo es `download_data.sh`, que al final ejecuta `train.py`.

## Dependencias principales (requirements.txt)
- tensorflow
- numpy
- pandas
- matplotlib
- scikit-learn
- kaggle
- Pillow
- seaborn
- opencv-python-headless

---
API Key de Kaggle, consulta: https://www.kaggle.com/docs/api#authentication