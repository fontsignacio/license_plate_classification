#!/bin/bash

# 0. Verificar si ya existe la carpeta completa
if [ -d "./data/dataset/train" ] && [ -d "./data/dataset/val" ] && [ -d "./data/dataset/test" ]; then
  echo "Dataset ya preparado completamente. Omite descarga y particionado."
  echo "Si desea volver a generarlo, elimine la carpeta ./data/dataset y ./data/license_plates/generated"
  echo "Ejecutando entrenamiento..."
  python train.py
  exit 0
fi

# 1. Configurar API Key de Kaggle
echo "Verificando credenciales de Kaggle..."
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "Error: Kaggle credentials not found in environment variables"
    exit 1
fi

mkdir -p /root/.kaggle
echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json

# 2. Descargar y descomprimir solo si no existe
if [ ! -d ./data/license_plates/generated ]; then
  echo "Descargando y descomprimiendo dataset..."
  mkdir -p ./data
  kaggle datasets download -d abtexp/synthetic-indian-license-plates -p ./data
  unzip -oq ./data/synthetic-indian-license-plates.zip -d ./data/license_plates
else
  echo "El dataset ya está descomprimido en ./data/license_plates/generated, omitiendo descarga."
fi

# 3. Definir las clases reales del dataset
CLASSES=(commercial commercial_electrical private private_electrical rentable)

# 4. Crear carpetas globales por clase
for split in train val test; do
  for class in "${CLASSES[@]}"; do
    mkdir -p ./data/dataset/$split/$class
  done
done

# 5. Limpiar archivos temporales previos
for class in "${CLASSES[@]}"; do
  rm -f "./data/all_${class}_images.txt"
done

# 6. Recorrer y recolectar rutas de imágenes
for state_dir in ./data/license_plates/generated/*; do
  [ -d "$state_dir" ] || continue
  for class in "${CLASSES[@]}"; do
    src_dir="$state_dir/$class"
    [ -d "$src_dir" ] || continue
    for img in "$src_dir"/*.png; do
      [ -e "$img" ] || continue
      echo "$img" >> "./data/all_${class}_images.txt"
    done
  done
done

# 7. Split estratificado por clase
for class in "${CLASSES[@]}"; do
  if [ ! -f "./data/all_${class}_images.txt" ]; then
    echo "No hay imágenes para la clase $class"
    continue
  fi
  shuf "./data/all_${class}_images.txt" > "./data/shuf_${class}.txt"
  total=$(wc -l < "./data/shuf_${class}.txt")
  if [ "$total" -eq 0 ]; then
    echo "No hay imágenes para la clase $class"
    continue
  fi
  train_count=$((total * 80 / 100))
  val_count=$((total * 10 / 100))
  test_count=$((total - train_count - val_count))
  n=0
  while read img; do
    if [ $n -lt $train_count ]; then
      cp "$img" "./data/dataset/train/$class/"
    elif [ $n -lt $((train_count + val_count)) ]; then
      cp "$img" "./data/dataset/val/$class/"
    else
      cp "$img" "./data/dataset/test/$class/"
    fi
    n=$((n+1))
  done < "./data/shuf_${class}.txt"
done

# 8. Resumen
for split in train val test; do
  for class in "${CLASSES[@]}"; do
    count=$(ls ./data/dataset/$split/$class | wc -l)
    echo "$split/$class: $count"
  done
done

# 9. Ejecutar entrenamiento
echo "Iniciando entrenamiento..."
python train.py
