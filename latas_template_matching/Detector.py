# Detector.py
#
# Diego Iván Martínez Escobar, 2025
#
# Programa de Honores
# Universidad de las Américas Puebla

import warnings

class Detector:
    def __init__(self, detector_latas, detector_agua = None, param_dict = None):
        """
        La clase detector es la encargada de generar las detecciones usando
        alguno de los métodos disponibles.

        El diccionario de parámetros contiene los valores adicionales que las funciones detector necesitan
        para funcionar. Estos parámetros están documentados en cada una de las definiciones de las funciones
        detector.

        :param detector_latas: Función detector de latas de la forma def detector(np.ndarray, param_dict)
        :param detector_agua: Función detector de agua de la forma def detector(np.ndarray, param_dict)
        :param param_dict: El diccionario de parámetros, de la forma {string: Any, ...}
        """
        if param_dict is None:
            param_dict = {}
        self.detector_latas = detector_latas

        if detector_agua is None:
            warnings.warn("Se ha creado un detector sin detector de agua")

        self.detector_agua = detector_agua
        self.param_dict = param_dict if param_dict is not None else {}

    def detectar_objetos(self, imagen):
        """
        Genera una detección basada en su función detector interna. El valor de retorno es una lista de tuplas, cada
        una con cuatro elementos de la forma ("clase", x1, y1, x2, y2).  Su interpretación se muestra a continuación:
         (x1, y1)
            #────────────────────────────┐
            │                            │
            │                            │
            │                            │
            │                            │
            │          Detección         │
            │                            │
            │                            │
            │                            │
            │                            │
            │                            │
            └────────────────────────────#
                                    (x2, y2)
        :param imagen: un np.ndarray que representa una imagen R8G8B8
        :return: Una lista de tuplas de la forma [("clase", x1, y1, x2, y2), ...], donde (x1, y1) es la esquina superior izquierda
        de la detección, (x2, y2) la esquina inferior derecha, y "clase" es la etiqueta de la clase, como String.
        """

        assert self.detector_latas is not None

        lista_objetos = []

        for lata in self.detector_latas(imagen, self.param_dict):
            x1, y1, x2, y2 = lata
            lista_objetos.append(("lata", x1, y1, x2, y2))

        if self.detector_agua is None:
            return lista_objetos

        for agua in self.detector_agua(imagen, self.param_dict):
            x1, y1, x2, y2 = agua
            lista_objetos.append(("agua", x1, y1, x2, y2))

        return lista_objetos