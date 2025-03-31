# Detector de Latas

Se pueden detectar con diferentes métodos:

- Template Matching
- Segmentación por Umbralización de Método Otsu
- Segmentación por objetos oscuros

El proyecto usa [Poetry](https://python-poetry.org/) para crear ambientes Python
con las dependencias requeridas.

## Uso General

La clase `Detector` envuelve las llamadas a las funciones de detección,
que toman los siguientes parámetros:

* Una imagen RGB (como `numpy.ndarray`)
* Un diccionario de parámetros (de la forma `{string: valor, ...}`)

Los parámetros de cada función están documentados en sus respectivos
archivos.

Los valores que se devuelven de las funciones son listas de 4-tuplas
de la forma `(x1, y1, x2, y2)` donde `(x1, y1)` es la esquina
superior izquierda y `(x2, y2)` la esquina inferior derecha.

## Template Matching (DetectorTemplate)

Esta función toma los siguientes parámetros:

* `templates`: Una lista de `numpy.ndarray`s que contenga las plantillas como imágenes RGB
* `filter`: El filtro que se le aplica a la imagen de donde se obtendrán los matches. Si no se provee, el valor por defecto será un filtro blanco y negro. El programa también provee la opción de un filtro de entropía.
* `max_intersection`: El valor máximo de IoU hasta el cual dos detecciones son consideradas diferentes, si dos detecciones comparten más del N% en IoU, serán consideradas detecciones al mismo objeto.


## Segmentación por Umbralización Otsu (DetectorOtsu)

Utiliza el método de Otsu para separar la arena y el fondo de las
latas. Toma los siguientes parámetros:

* `min_area`: El área mínima a partir de la cuál considerará una detección como válida, para reducir ruido.

## Segmentación por Objetos Oscuros.

Este método considera que todos los objetos cuyo valor en la altura de negro a blanco en HSV esté
en el intervalo `[lower_black, upper_black]` son latas. Podemos
hacer esta generalización ya que todas las latas del reto
son negras.

Los parámetros que toma son los siguientes

* `lower_black`: El valor más oscuro que considerará una lata
* `upper_black`: El valor más claro que considerará una lata
* `min_area`: El área mínima a partir de la cuál considerará una detección como válida

## Ejemplo

El programa de ejemplo muestra la detección de latas en un video
seleccionado por el usuario
