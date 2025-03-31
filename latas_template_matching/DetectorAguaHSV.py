# DetectorAguaHSV.py
#
# Diego Iván Martínez Escobar, 2025.
# Universidad de las Américas Puebla
#

import cv2
import numpy as np

def detector_agua_hsv(imagen, params):
    """
    Genera bounding boxes donde hay agua, determinado por el color.
    Se puede elegir un rango de colores que serán considerados agua.

    El diccionario de parámetros puede contener la siguiente
    información:

    lower_blue: Un numpy.array con un color RGB, que determinará la cota inferior de lo que se considera azul.
    upper_blue: Un numpy.array con un color RGB que determinará la cota superior de lo que se considera azul.

    :param imagen: Un numpy.ndarray con la imagen, en RGB
    :param params: El diccionario de parámetros
    :return:
    """

    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    lower_blue = params.get("lower_blue", np.array([23, 100, 130], dtype=np.uint8))
    upper_blue = params.get("upper_blue", np.array([144, 150, 255], dtype=np.uint8))

    mascara = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)

    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentado = cv2.bitwise_and(imagen, imagen, mask=mascara)

    agua_detectada = []

    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        agua_detectada.append((x, y, x + w, y + h))

    return agua_detectada