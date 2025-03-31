import numpy as np
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk

def aplicar_filtro_entropia(image):
    """
    Aplica un filtro de entropía a la imagen dada
    :param image: Una imagen (np.ndarray). La imagen debe estar en color blanco y negro.
    :return: La imagen con filtro de entropía aplicado
    """

    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Ensure the image is of type uint8.
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Define the local neighborhood (disk of radius 5).
    selem = disk(5)

    # Apply the entropy filter.
    entropy_img = entropy(gray, selem)

    # Normalize the entropy image to the range [0, 255].
    entropy_normalized = (entropy_img / np.max(entropy_img) * 255).astype(np.uint8)

    return entropy_normalized