import cv2
import numpy as np
import pytesseract

# Carregar a imagem
image = cv2.imread('medidor.jpeg')

# Converter para escala de cinza para melhorar o reconhecimento de texto
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Aplicar um filtro Gaussiano para suavizar a imagem
# gray = cv2.GaussianBlur(gray, (5, 5), 0)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 0)

print(gray)

# # Realizar a detecção de texto
# text = pytesseract.image_to_string(gray)
# print(text)

# Calcule as dimensões da imagem
altura, largura, _ = image.shape

# Calcule as coordenadas para a ROI com base no centro da imagem
roi_width = 150  # Largura da ROI
roi_height = 90  # Altura da ROI

# Calcule as coordenadas do canto superior esquerdo da ROI para centralizá-la
x = (largura - roi_width) // 2  # O operador // realiza uma divisão inteira
y = (altura - roi_height) // 2

# Defina a ROI
roi = image[y:y + roi_height, x:x + roi_width]

# Realize a detecção de texto apenas na ROI
# text = pytesseract.image_to_string(roi)
text = pytesseract.image_to_string(roi, config='--psm 6')

print(text)

# Visualize a imagem com a ROI destacada
cv2.rectangle(image, (x, y), (x + roi_width, y + roi_height), (0, 255, 0), 2)  # Retângulo verde na ROI

# Mostrar a imagem com a ROI destacada
cv2.imshow('Imagem com ROI', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Texto extraído da ROI:", text)