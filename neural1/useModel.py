import cv2
import numpy as np
import tensorflow as tf

# Carregar o modelo treinado
model = tf.keras.models.load_model('model.h5')

# Capturar uma imagem do medidor de água
image = cv2.imread('../medidor.jpeg')

# Pré-processar a imagem
image = cv2.resize(image, (28, 28))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image / 255.0

# Passar a imagem para o modelo
prediction = model.predict(image.reshape(1, 28, 28, 1))

# Extrair os valores predito
value = np.argmax(prediction)

# Imprimir os valores predito
print(value)