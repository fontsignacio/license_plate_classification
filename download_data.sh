#!/bin/bash

# Configurar API Key de Kaggle
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "Error: Kaggle credentials not found in environment variables"
    exit 1
fi

mkdir -p /root/.kaggle
echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json

# Descargar dataset
echo "Descargando dataset..."
kaggle datasets download -d abtexp/synthetic-indian-license-plates -p ./data
unzip -q ./data/synthetic-indian-license-plates.zip -d ./data/license_plates

# Preparar estructura de directorios
echo "Preparando estructura de directorios..."
mkdir -p ./data/dataset/train/license_plate
mkdir -p ./data/dataset/train/background
mkdir -p ./data/dataset/val/license_plate
mkdir -p ./data/dataset/val/background
mkdir -p ./data/dataset/test/license_plate
mkdir -p ./data/dataset/test/background

# Contadores
total_plates=0
total_background=0

# Función para mover archivos con manejo de espacios
move_files() {
    local src_dir="$1"
    local dest_dir="$2"
    
    find "$src_dir" -type f -name "*.png" -print0 | while IFS= read -r -d $'\0' file; do
        if [[ "$file" == *"license_plate"* ]] || [[ "$file" == *"generated"* ]]; then
            ((total_plates++))
            dest="./data/dataset"
            if ((total_plates % 10 < 8)); then
                dest+="/train/license_plate"
            elif ((total_plates % 10 < 9)); then
                dest+="/val/license_plate"
            else
                dest+="/test/license_plate"
            fi
        else
            ((total_background++))
            dest="./data/dataset"
            if ((total_background % 10 < 8)); then
                dest+="/train/background"
            elif ((total_background % 10 < 9)); then
                dest+="/val/background"
            else
                dest+="/test/background"
            fi
        fi
        
        # Mover preservando nombre y directorios
        mv "$file" "$dest/$(basename "$file")"
    done
}

# Procesar todas las imágenes
echo "Procesando imágenes de matrículas y fondos..."
find "./data/license_plates/generated" -type f -name "*.png" -print0 | while IFS= read -r -d $'\0' file; do
    ((total_plates++))
    dest="./data/dataset"
    if ((total_plates % 10 < 8)); then
        dest+="/train/license_plate"
    elif ((total_plates % 10 < 9)); then
        dest+="/val/license_plate"
    else
        dest+="/test/license_plate"
    fi
    mv "$file" "$dest/$(basename "$file")"
done

find "./data/license_plates/background" -type f -name "*.png" -print0 | while IFS= read -r -d $'\0' file; do
    ((total_background++))
    dest="./data/dataset"
    if ((total_background % 10 < 8)); then
        dest+="/train/background"
    elif ((total_background % 10 < 9)); then
        dest+="/val/background"
    else
        dest+="/test/background"
    fi
    mv "$file" "$dest/$(basename "$file")"
done

# Verificar resultados
count_train_plates=$(find ./data/dataset/train/license_plate -type f | wc -l)
count_train_background=$(find ./data/dataset/train/background -type f | wc -l)
count_val_plates=$(find ./data/dataset/val/license_plate -type f | wc -l)
count_val_background=$(find ./data/dataset/val/background -type f | wc -l)
count_test_plates=$(find ./data/dataset/test/license_plate -type f | wc -l)
count_test_background=$(find ./data/dataset/test/background -type f | wc -l)

echo "Dataset preparado:"
echo "  Total placas: $total_plates"
echo "  Total fondos: $total_background"
echo "  Train: $count_train_plates placas, $count_train_background fondos"
echo "  Val:   $count_val_plates placas, $count_val_background fondos"
echo "  Test:  $count_test_plates placas, $count_test_background fondos"

# Ejecutar script de entrenamiento
python train.py