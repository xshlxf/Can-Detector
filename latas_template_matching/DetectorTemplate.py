import cv2
import numpy as np

def detector_template_matching(image, param_dict):
    """
    Genera detecciones basadas en template matching. También elimina las detecciones duplicadas utilizando IoU.

    Parámetros del Diccionario:

    * max_intersection: Un valor de [0, 1] que indique la IoU en la que una detección se considera igual a otra.
    Default: 0.7.
    * filter: Una función de la forma np.ndarray -> np.ndarray para transformar la imagen de detección. El valor
    por defecto es un filtro en blanco y negro.
    * templates: Una lista de np.ndarray, donde cada np.ndarray es una plantilla. Este parámetro no tiene valor por defecto,
    por lo que si no es provisto no se podrá hacer ninguna detección.

    :param image: La imagen RGB.
    :param param_dict: El diccionario de parámetros
    :return: Una lista de detecciones de la forma [(x1, y1, x2, y2), ...]
    """

    detecciones = []
    if param_dict.get("templates"):
        templates = param_dict["templates"]
    else:
        return None

    filtro = param_dict.get("filter", _filtro_gray_scale)
    filtrada = filtro(image)
    max_iou = param_dict.get("max_intersection", 0.7)

    for template in templates:
        nueva_deteccion = _deteccion_para_plantilla(filtrada, template)
        # Aquí evitamos que lleguen las detecciones "duplicadas"
        # TODO: Buscar un algoritmo más eficiente para hacer esto.
        for deteccion in detecciones:
            if calculate_iou(nueva_deteccion, deteccion) > max_iou:
                continue
            detecciones.append(nueva_deteccion)

    return detecciones

def calculate_iou(boxA, boxB):
    # Extract bounding boxes coordinates
    x0_A, y0_A, x1_A, y1_A = boxA
    x0_B, y0_B, x1_B, y1_B = boxB

    # Get the coordinates of the intersection rectangle
    x0_I = max(x0_A, x0_B)
    y0_I = max(y0_A, y0_B)
    x1_I = min(x1_A, x1_B)
    y1_I = min(y1_A, y1_B)
    # Calculate width and height of the intersection area.
    width_I = x1_I - x0_I
    height_I = y1_I - y0_I
    # Handle the negative value width or height of the intersection area
    # if width_I<0 : width_I=0
    # if height_I<0 : height_I=0
    width_I = width_I if width_I > 0 else 0
    height_I = height_I if height_I > 0 else 0
    # Calculate the intersection area:
    intersection = width_I * height_I
    # Calculate the union area:
    width_A, height_A = x1_A - x0_A, y1_A - y0_A
    width_B, height_B = x1_B - x0_B, y1_B - y0_B
    union = (width_A * height_A) + (width_B * height_B) - intersection
    # Calculate the IoU:
    IoU = intersection / union
    # for plotting purpose
    # Return the IoU and intersection box
    return IoU

def _filtro_gray_scale(imagen):
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

def _deteccion_para_plantilla(imagen, plantilla):
    w, h = plantilla.shape[::-1]

    metodo = eval('cv.TM_CCORR')
    res = cv2.matchTemplate(imagen, plantilla, metodo)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    return max_loc[0], max_loc[1], max_loc[0] + w, max_loc[1] + h