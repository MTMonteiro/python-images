import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Defina o caminho para o diretório que contém suas imagens de treinamento
data_dir = './data'

# Defina as dimensões desejadas para redimensionar suas imagens
image_height = 64
image_width = 64

# Função para carregar e pré-processar as imagens
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_width, image_height))
    image = image / 255.0  # Normalização (escala de 0 a 1)
    return image

# Carregar as imagens e rótulos
images = []
labels = []

for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for image_name in os.listdir(class_dir):
            if image_name.endswith('.jpg') or image_name.endswith('.jpeg'):
                image_path = os.path.join(class_dir, image_name)
                image = load_and_preprocess_image(image_path)
                images.append(image)
                labels.append(class_name)

# Converter rótulos em números (pode ser necessário mais pré-processamento)
label_to_index = {label: index for index, label in enumerate(set(labels))}
labels = [label_to_index[label] for label in labels]

# Dividir os dados em conjuntos de treinamento e teste
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Converta os dados para arrays NumPy
train_images = np.array(train_images)
test_images = np.array(test_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Defina o modelo de rede neural (semelhante ao exemplo anterior)
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(set(labels)), activation='softmax')  # Número de classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treine o modelo
model.fit(train_images, train_labels, epochs=100, validation_data=(test_images, test_labels))

model.save('model.h5')