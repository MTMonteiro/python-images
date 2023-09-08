from tensorflow import keras
import cv2
import numpy as np
import os

# Carregar as imagens e rótulos
images = []
labels = []
data_dir = './data'

for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for image_name in os.listdir(class_dir):
            if image_name.endswith('.jpg') or image_name.endswith('.jpeg'):
                # image_path = os.path.join(class_dir, image_name)
                # image = load_and_preprocess_image(image_path)
                # images.append(image)
                labels.append(class_name)

# Converter rótulos em números (pode ser necessário mais pré-processamento)
label_to_index = {label: index for index, label in enumerate(set(labels))}
labels = [label_to_index[label] for label in labels]

image_height = 64
image_width = 64

# Função para carregar e pré-processar as imagens
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_width, image_height))
    image = image / 255.0  # Normalização (escala de 0 a 1)
    return image

# Carregar o modelo salvo
modelo = keras.models.load_model('model.h5')

# Pré-processar e fazer previsões em uma nova imagem
# nova_imagem = load_and_preprocess_image('../1.jpg')  # Carregue e pré-processe sua nova imagem
# nova_imagem = load_and_preprocess_image('../medidor.jpeg')  # Carregue e pré-processe sua nova imagem
nova_imagem = load_and_preprocess_image('../image.jpg')  # Carregue e pré-processe sua nova imagem
previsoes = modelo.predict(np.array([nova_imagem]))  # Faça previsões em um array NumPy

# Descubra a classe prevista (índice) com maior probabilidade
classe_prevista = np.argmax(previsoes)

print(f"classe_prevista: {classe_prevista}")
# Mapeie o índice da classe prevista de volta para o rótulo, se necessário
indice_para_rotulo = {index: label for label, index in label_to_index.items()}
rotulo_previsto = indice_para_rotulo[classe_prevista]

print(f"Placa prevista: {rotulo_previsto}")
