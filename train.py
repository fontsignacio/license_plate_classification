import tensorflow as tf
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuración
IMG_SIZE = (128, 512)
BATCH_SIZE = 64
EPOCHS = 15 
LR = 1e-4

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

# Detectar clases automáticamente
def get_classes_from_directory(directory):
    return sorted([
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and not d.startswith('.')
    ])

print('Obteniendo clases...')    
CLASSES = get_classes_from_directory(train_dir)
NUM_CLASSES = len(CLASSES)
print(f"Clases detectadas: {CLASSES}")

# Generadores de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=False,  # Placas no deberían reflejarse
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255, dtype='float32')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    classes=CLASSES
)

# Calcular class_weight
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights = dict(enumerate(class_weights_array))
print("Pesos por clase:", class_weights)

# Construcción del modelo
def build_model(base_model, model_name):
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs, outputs, name=model_name)
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("\nConstruyendo modelos...")
vgg_model = build_model(VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)), "VGG16")
mobilenet_model = build_model(MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)), "MobileNetV2")

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    ModelCheckpoint('output/best_vgg.h5', save_best_only=True, monitor='val_accuracy'),
    ModelCheckpoint('output/best_mobilenet.h5', save_best_only=True, monitor='val_accuracy'),
]


# Entrenamiento inicial
print("\nEntrenando VGG16...")
vgg_history = vgg_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

print("\nEntrenando MobileNetV2...")
mobilenet_history = mobilenet_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Gráficos
def plot_history(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title(f'Precisión - {model_name}')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend()
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
plot_history(mobilenet_history, "MobileNetV2")

# Evaluación
def generate_classification_report(model, generator, model_name):
    y_pred = model.predict(generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = generator.classes
    labels = list(generator.class_indices.values())
    target_names = list(generator.class_indices.keys())
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(
        y_true, y_pred_classes,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0
    ))
    cm = confusion_matrix(y_true, y_pred_classes, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Verdaderos')
    plt.xlabel('Predicciones')
    plt.savefig(f'output/{model_name}_confusion_matrix.png')
    plt.close()

generate_classification_report(vgg_model, test_generator, "VGG16")
generate_classification_report(mobilenet_model, test_generator, "MobileNetV2")

print("\nEntrenamiento y evaluación completados!")