import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

CLASSES = ['commercial', 'commercial_electrical', 'private', 'private_electrical', 'rentable']

# Ruta absoluta del directorio raíz (donde está este script)
directorio_raiz = os.path.dirname(os.path.abspath(__file__))

# Ruta de la imagen
img_path = os.path.join(directorio_raiz, 'data/dataset/test/rentable/AN20UN6757.png')

# Cargar imagen y preprocesar
img = image.load_img(img_path, target_size=(128, 512))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalización
img_array = np.expand_dims(img_array, axis=0)  # Añadir batch dimension

# Cargar modelos
vgg_model = load_model(os.path.join(directorio_raiz, 'output/best_vgg.h5'))
mobilenet_model = load_model(os.path.join(directorio_raiz, 'output/best_mobilenet.h5'))

# Predicción con VGG
pred_vgg = vgg_model.predict(img_array)
predicted_class_vgg = CLASSES[np.argmax(pred_vgg)]
confidence_vgg = np.max(pred_vgg)

# Predicción con MobileNet
pred_mobilenet = mobilenet_model.predict(img_array)
predicted_class_mobilenet = CLASSES[np.argmax(pred_mobilenet)]
confidence_mobilenet = np.max(pred_mobilenet)

print(f'[VGG] Clase predicha: {predicted_class_vgg} (confianza: {confidence_vgg:.2f})')
print(f'[MobileNet] Clase predicha: {predicted_class_mobilenet} (confianza: {confidence_mobilenet:.2f})')
