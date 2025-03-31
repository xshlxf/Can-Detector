# DetectorOtsu.py
#
# Diego Iván Martínez Escobar, 2025
#
# Programa de Honores
# Universidad de las Américas Puebla
import cv2


def deteccion_otsu(imagen, param_dict):
    """
    Genera detecciones usando el threshold con el método de Otsu.
    Este detector usa los siguientes parámetros en el diccionario:

    min_area: Un número que representa el área mínima desde la cuál una detección será considerada válida.
    Las detecciones cuya bounding box no tenga esta área mínima serán ignoradas.

    :param imagen: un np.ndarray
    :param param_dict: El diccionario de parámetros (véase @Detector)
    :return: La lista de detecciones de la forma [(x1, y1, x2, y2), ...]
    """
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar Blur Gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar el threshold al frame utilizando el método de Otsu
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Operaciones morfológicas para reducir más ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Encontrar los contornos
    contours, _ = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    latas_detectadas = []

    # Obtener el área mínima del diccionario de parámetros
    # Si no está disponible, utilizará 500
    area_minima = param_dict.get("min_area", 500)

    # Añadimos los contornos que tengan un área mayor a la mínima.
    for cnt in contours:
        if cv2.contourArea(cnt) < area_minima:
            continue

        # Get bounding box for each detected object
        x, y, w, h = cv2.boundingRect(cnt)
        latas_detectadas.append((x, y, x + w, y + h))

    return latas_detectadas