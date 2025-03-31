import cv2
import numpy as np

def detector_objetos_oscuros(imagen, param_dict):
    """
    Este detector separa a los objetos oscuros de la arena.
    Nos vamos a aprovechar del hecho que las latas están pintadas de negro, y vamos a asumir
    que los objetos oscuros que se encuentren dentro del rango HSV determinado por el parámetro de diccionarios
    son latas.

    Parámetros del diccionario:
    lower_black: Un np.array que tenga el valor más bajo dentro del HSV que será considerado negro, en RGB. Valor default: RGB(0, 0, 0).
    upper_black: Un np.array que tenga el valor más alto dentro del HSV que será considerado negro, en RGB. Valor default: RGB(180, 255, 50).
    min_area: Un número que representa el área mínima para que una detección sea considerada válida. Valor default: 500.

    :param imagen: La imagen en formato RGB
    :param param_dict: El diccionario de parámetros
    :return:
    """

    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    lower_black = param_dict.get('lower_black', np.array([0, 0, 0], dtype=np.uint8))
    upper_black = param_dict.get('upper_black', np.array([180, 255, 50], dtype=np.uint8))

    mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmented = cv2.bitwise_and(imagen, imagen, mask=mask)

    latas_detectadas = []
    area_minima = param_dict.get("min_area", 500)

    for cnt in contours:
        # Ignorar detecciones muy pequeñas

        if cv2.contourArea(cnt) < area_minima:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        latas_detectadas.append((x, y, x+w, y+h))

    return latas_detectadas