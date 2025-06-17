import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuración
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2  # Placa vs Fondo

# Rutas de datos
base_dir = 'data/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Verificar directorios
for path in [train_dir, val_dir, test_dir]:
    if not os.path.exists(path):
        print(f"Error: Directorio no encontrado: {path}")
        exit(1)
    print(f"{path}: {len(os.listdir(os.path.join(path, 'license_plate')))} placas, "
        f"{len(os.listdir(os.path.join(path, 'background')))} fondos")

# Generadores de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    dtype='float32'
)

val_test_datagen = ImageDataGenerator(
    rescale=1./255,
    dtype='float32'
)

print("\nCreando generadores de datos...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=['background', 'license_plate']  # Orden explícito
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=['background', 'license_plate']  # Orden explícito
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    classes=['background', 'license_plate']  # Orden explícito
)

# Función para construir modelos
def build_model(base_model, model_name):
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs, outputs, name=model_name)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("\nConstruyendo modelos...")
vgg_model = build_model(VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)), "VGG16")
resnet_model = build_model(ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)), "ResNet50")

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=2, verbose=1),
    ModelCheckpoint('output/best_vgg.h5', save_best_only=True, monitor='val_accuracy'),
    ModelCheckpoint('output/best_resnet.h5', save_best_only=True, monitor='val_accuracy')
]

# Entrenamiento
print("\nEntrenando VGG16...")
vgg_history = vgg_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nEntrenando ResNet50...")
resnet_history = resnet_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Fine-tuning
print("\nFine-tuning VGG16...")
for layer in vgg_model.layers[1].layers[-4:]:
    layer.trainable = True
vgg_model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
vgg_history_ft = vgg_model.fit(
    train_generator,
    epochs=3,
    validation_data=val_generator,
    verbose=1
)

print("\nFine-tuning ResNet50...")
for layer in resnet_model.layers[1].layers[-10:]:
    layer.trainable = True
resnet_model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
resnet_history_ft = resnet_model.fit(
    train_generator,
    epochs=3,
    validation_data=val_generator,
    verbose=1
)

# Evaluación
print("\nEvaluando modelos...")
vgg_test_loss, vgg_test_acc = vgg_model.evaluate(test_generator, verbose=0)
resnet_test_loss, resnet_test_acc = resnet_model.evaluate(test_generator, verbose=0)

print(f"\nVGG16 - Precisión en test: {vgg_test_acc:.4f}")
print(f"ResNet50 - Precisión en test: {resnet_test_acc:.4f}")

# Guardar modelos finales
vgg_model.save('output/vgg16_plate_classifier.h5')
resnet_model.save('output/resnet50_plate_classifier.h5')

# Graficar resultados
def plot_history(history, model_name):
    plt.figure(figsize=(12, 4))
    
    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title(f'Precisión - {model_name}')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend()
    
    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title(f'Pérdida - {model_name}')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'output/{model_name}_history.png')
    plt.close()

plot_history(vgg_history, "VGG16")
plot_history(resnet_history, "ResNet50")

# Reportes de clasificación
def generate_classification_report(model, generator, model_name):
    # Obtener predicciones
    y_pred = model.predict(generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = generator.classes
    
    # Reporte
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true, y_pred_classes, 
                                target_names=generator.class_indices.keys(),
                                digits=4))
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=generator.class_indices.keys(), 
                yticklabels=generator.class_indices.keys())
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Verdaderos')
    plt.xlabel('Predicciones')
    plt.savefig(f'output/{model_name}_confusion_matrix.png')
    plt.close()

generate_classification_report(vgg_model, test_generator, "VGG16")
generate_classification_report(resnet_model, test_generator, "ResNet50")

print("\nEntrenamiento y evaluación completados!")