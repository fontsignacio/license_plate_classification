FROM tensorflow/tensorflow:2.13.0-gpu

# Instalar dependencias
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    libgl1-mesa-glx  # Requerido para OpenCV

# Configurar entorno
WORKDIR /app
COPY . /app

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip cache purge  # Limpiar cache

# Dar permisos de ejecución
RUN chmod +x download_data.sh

# Cargar variables de entorno y ejecutar
CMD ["bash", "-c", "source .env && ./download_data.sh"]