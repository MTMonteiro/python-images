
import cv2
import numpy as np
import pytesseract

image = cv2.imread("medidor.jpeg")

def detect_water_meter(image):
    # Converta a imagem para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detecte os contornos na imagem
    contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Encontre o maior contorno
    max_contour = max(contours, key=cv2.contourArea)

    # Retorne o contorno do medidor de água
    return max_contour

def extract_water_meter_data(meter_contour):
    # Converta o contorno para uma imagem
    meter_image = cv2.drawContours(image, [meter_contour], -1, (0, 255, 0), 2)

    # Recorte a imagem do medidor de água
    meter_cropped = meter_image[meter_contour.minY:meter_contour.maxY, meter_contour.minX:meter_contour.maxX]

    # Converta a imagem para preto e branco
    meter_gray = cv2.cvtColor(meter_cropped, cv2.COLOR_BGR2GRAY)

    # Use o OCR para extrair os dados de medição
    meter_data = pytesseract.image_to_string(meter_gray)

    meter_data = meter_data[meter_contour.minY:meter_contour.maxY].split()[0]

    return meter_data

meter_contour = detect_water_meter(image)
meter_data = extract_water_meter_data(meter_contour)