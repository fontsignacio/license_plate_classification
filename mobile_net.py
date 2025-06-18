import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score

# Parámetros
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5

TRAIN_DIR = 'data/dataset/train'
VAL_DIR = 'data/dataset/val'
TEST_DIR = 'data/dataset/test'

# Preprocesamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Modelo base
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
]

history1 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# Unificar métricas
def plot_history(histories, epochs_per_phase):
    acc, val_acc, loss, val_loss = [], [], [], []
    for history in histories:
        acc += history.history['accuracy']
        val_acc += history.history['val_accuracy']
        loss += history.history['loss']
        val_loss += history.history['val_loss']

    epochs_total = range(len(acc))

    plt.figure(figsize=(16, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_total, acc, label='Train Accuracy')
    plt.plot(epochs_total, val_acc, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_total, loss, label='Train Loss')
    plt.plot(epochs_total, val_loss, label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

plot_history([history1, history2], EPOCHS)

# Evaluación
loss, accuracy = model.evaluate(test_generator)
print(f"Precisión en el conjunto de test: {accuracy*100:.2f}%")

# Predicciones
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title('Matriz de Confusión')
plt.show()

# Métricas por clase
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
specificity = []

for i in range(len(class_names)):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    specificity.append(TN / (TN + FP) if (TN + FP) != 0 else 0)

# Gráfico de métricas por clase
x = np.arange(len(class_names))
width = 0.25

plt.figure(figsize=(14, 6))
plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, specificity, width, label='Specificity')

plt.xticks(x, class_names, rotation=45)
plt.ylabel('Valor')
plt.title('Métricas por clase')
plt.legend()
plt.tight_layout()
plt.show()

# Guardar modelo
model.save('mobilenetv2_license_type_model.h5')
