import tensorflow as tf
import os
import cv2
import numpy as np
# from sklearn.model_selection import train_test_split

# Colete as imagens de um diretório
# your_images = [
#   cv2.imread(os.path.join('data', image)) for image in os.listdir('data')]

data_dir = 'data'  # Pasta com as imagens
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
your_images, your_labels = load_and_preprocess_images(data_dir)

# your_images = np.array([
#   cv2.imread('data/371.jpg'),
# ])

# your_images = [
#   cv2.imread(os.path.join('data', image)) for image in os.listdir()]

print(your_images)
# Obter os valores medidos dos nomes das imagens
# your_labels = [int(image.split('.')[0]) for image in your_images]
# your_labels = [371]
# your_labels = [int(image.split('.')[0]) for image in your_images]

# # Dividir o conjunto de dados em treino e teste
# x_train, x_test, y_train, y_test = train_test_split(
#   your_images, your_labels, test_size=0.25)

# Redimensionar as imagens
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# # Normalizar as imagens
# x_train = x_train / 255.0
# x_test = x_test / 255.0

# Adicione as fotos ao conjunto de dados
x_train = []
y_train = []

x_train = np.concatenate([x_train, your_images])
y_train = np.concatenate([y_train, your_labels])

# # Redimensionar as imagens
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# # Normalizar as imagens
# x_train = x_train / 255.0
# x_test = x_test / 255.0

# Construir a rede neural
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train, y_train, epochs=10)

# Avaliar o modelo
model.evaluate(x_test, y_test)
model.save('model')