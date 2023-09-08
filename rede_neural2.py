import os
import cv2
import numpy as np

data_dir = 'dataset'  # Pasta com as imagens
image_size = (28, 28)

# Função para carregar e pré-processar as imagens
def load_and_preprocess_images(data_dir):
    images = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)
            img = img.astype('float32') / 255.0  # Normalização
            images.append(img)

            # Extrair o rótulo da imagem (assumindo que o nome do arquivo é o rótulo)
            label = os.path.splitext(filename)[0]
            labels.append(label)

    return np.array(images), labels

# Carregamento e pré-processamento das imagens
images, labels = load_and_preprocess_images(data_dir)

# Converter os rótulos de texto em sequências de números
label_to_sequence = {str(i): i for i in range(10)}  # Mapeamento de dígitos
labels = [[label_to_sequence[digit] for digit in label] for label in labels]


import tensorflow as tf
from tensorflow.keras import layers, models

# Definição do modelo RNN
model = models.Sequential([
    layers.Input(shape=(28, 28), name='input_layer'),
    layers.LSTM(128, return_sequences=True),  # Camada LSTM para processar sequências
    layers.Dense(10, activation='softmax')  # 10 classes para os dígitos de 0 a 9
])

# Compilação do modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Treinamento do modelo
model.fit(images, labels, epochs=10, validation_split=0.1)


# Carregar uma nova imagem para previsão
new_image = cv2.imread('medidor.jpeg', cv2.IMREAD_GRAYSCALE)
new_image = cv2.resize(new_image, image_size)
new_image = new_image.astype('float32') / 255.0  # Normalização
new_image = np.expand_dims(new_image, axis=0)  # Adicionar dimensão de lote

# Fazer uma previsão
predicted_sequence = model.predict(new_image)

print("Sequência Numérica Reconhecida:", predicted_sequence)
