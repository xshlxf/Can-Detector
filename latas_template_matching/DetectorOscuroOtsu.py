#
# Diego Iván Martínez Escobar, 2025
# Programa de Honores
# Universidad de las Américas Puebla

import cv2
import numpy as np

def detector_otsu_con_objetos_oscuros(imagen, param_dict):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    lower_black = param_dict.get('lower_black', np.array([0, 0, 0], dtype=np.uint8))
    upper_black = param_dict.get('upper_black', np.array([180, 255, 50], dtype=np.uint8))

    mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmented = cv2.bitwise_and(imagen, imagen, mask=mask)

    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)

    # Aplicar Blur Gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar el threshold al frame utilizando el método de Otsu
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

    # Operaciones morfológicas para reducir más ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Encontrar los contornos
    contours, _ = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    latas_detectadas = []

    area_minima = param_dict.get("min_area", 500)

    # Añadimos los contornos que tengan un área mayor a la mínima.
    for cnt in contours:
        if cv2.contourArea(cnt) < area_minima:
            continue

        # Get bounding box for each detected object
        x, y, w, h = cv2.boundingRect(cnt)
        latas_detectadas.append((x, y, x + w, y + h))

    return latas_detectadas