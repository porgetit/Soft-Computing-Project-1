# Reconocimiento y Resolución de Sudokus mediante Lógica Difusa (IEEE-Style)

## Abstract - Resumen
Este proyecto implementa un pipeline completo para el reconocimiento y la resolución automática de sudokus a partir de una imagen de entrada. El enfoque principal es el uso de lógica difusa (Mamdani) para el soporte del procesamiento de imagen (detección de bordes) previo a la rectificación, segmentación y lectura de dígitos. El sistema integra: detección de bordes difusa, umbralización, componentes conectados, rectificación por homografía, extracción de celdas, reconocimiento de dígitos (HOG+kNN con respaldo OCR/Tesseract) y un solucionador por backtracking (MRV). Actualmente, el resultado final depende de un “objeto demo” (rejilla conocida) que se usa como respaldo para garantizar que el módulo de resolución funciona correctamente, dado que el reconocimiento de dígitos y el OCR de respaldo aún no alcanzan una exactitud suficiente de forma consistente.

**Index Terms - Palabras clave:** Sudoku, Lógica Difusa, Mamdani, Procesamiento de Imágenes, Detección de Bordes, Homografía, Segmentación, OCR, Tesseract, OpenCV, HOG, k-NN, Backtracking, MRV.

## I. Introduction - Introducción
El reconocimiento robusto de sudokus en imágenes del mundo real implica lidiar con variaciones de iluminación, perspectiva, ruido y grosores de líneas. Este proyecto explora un pipeline donde la detección de bordes se apoya en un sistema difuso Mamdani para mejorar la calidad del mapa de bordes antes de la binarización y extracción de la rejilla. A partir de la imagen rectificada, se segmentan celdas y se reconocen dígitos combinando un clasificador HOG+kNN con OCR de respaldo. Posteriormente, un solucionador por backtracking resuelve la cuadrícula resultante. En su estado actual, el pipeline prioriza la claridad y modularidad sobre la exactitud del OCR; por ello se incluye un respaldo con un puzzle conocido para demostrar el correcto funcionamiento del módulo de resolución.

## II. System Overview - Visión General del Sistema
Entrada: imagen del sudoku (.jpg o .png). Salida: sudoku resuelto (impreso en consola; archivos intermedios opcionales en modo verbose).

Pipeline general:

```
Imagen -> Bordes (difuso) -> Umbralización -> Mayor Componente ->
Rectificación (gris + binaria) -> Extracción de celdas ->
Reconocimiento de dígitos (HOG+kNN + OCR) -> Limpieza/Consenso -> Solver -> Solución
```

Diagrama de flujo (Mermaid, compatible con GitHub):

```mermaid
flowchart TD
    A[Imagen de entrada]
    B[Detección de bordes difusa]
    C[Umbralización]
    D[Componentes conectados (mayor componente)]
    E[Rectificación por homografía (gris y binaria)]
    F[Extracción de celdas (gris y bin)]
    G1[Reconocimiento HOG+kNN]
    G2[OCR Tesseract (respaldo)]
    H[Consenso y limpieza]
    I[Solucionador Sudoku (MRV + BT)]
    J[Sudoku resuelto]

    A --> B --> C --> D --> E --> F
    F --> G1 --> H
    F --> G2 --> H
    H --> I --> J
```

Notas de operación:
- Con VERBOSE=False, los archivos intermedios se generan en un directorio temporal y se descartan al finalizar. Con VERBOSE=True, se guardan en el directorio de trabajo para inspección.
- Si el solver no puede resolver con la cuadrícula reconocida, se prueban alternativas (solo KNN limpio, solo OCR limpio); si todo falla y el archivo es f2, se utiliza un puzzle conocido como demo para evidenciar el módulo de resolución.

## III. Methods - Metodología
1) Detección de bordes (difusa):
- Se construye un sistema Mamdani con variables lingüísticas Ix, Iy e Iout (blanco/negro) y funciones de pertenencia gaussianas para los gradientes. Para acelerar el centroide, se precalcula una LUT de (w0, w1) -> centroide.
- Implementación: utilities.py (FuzzyEdgeConfig, FuzzyEdgeDetector; convoluciones 2D con barra de progreso opcional); tinyfuzzy.py (core difuso, MFs, inferencia Mamdani/Sugeno y FCM en NumPy puro).

2) Umbralización:
- Umbral binario sobre el mapa de bordes normalizado. Implementación: thresholding.py.

3) Componentes conectados (8-conexo):
- Etiquetado por BFS para identificar el mayor componente (rejilla). Implementación: connected_components.py.

4) Rectificación por homografía (DLT simplificado):
- Estimación de esquinas por heurística de extremos (x±y) y warp por mapeo inverso bilineal. Implementación: rectify.py.

5) Extracción de celdas:
- División regular de la imagen rectificada con padding interno para reducir captura de líneas de la rejilla; opcionalmente genera recortes espejo binarios. Implementación: cell_extraction.py.

6) Reconocimiento de dígitos:
- Clasificador HOG+kNN (OpenCV): preprocesado binario (Otsu INV), limpieza de componentes tocando borde, recorte por contorno máximo, normalización a lienzo, extracción HOG; entrenamiento con muestra digits.png de OpenCV o dataset sintético (fuentes Hershey). Implementación: digit_classifier.py.
- OCR de respaldo con Tesseract: preprocesado (contraste, reescalado, binarización, recorte) y psm configurable (10/13). Implementación: ocr.py.
- Consenso y limpieza: se fija la cifra si KNN y OCR concuerdan; en caso contrario, se coloca una cifra que no introduzca conflicto en la cuadrícula parcial; segunda pasada de limpieza global.

7) Solucionador de Sudoku:
- Backtracking con heurística MRV y poda. Implementación: sudoku_solver.py.

## IV. Software Architecture - Arquitectura de Software
Estructura de archivos:
- main.py: orquesta el pipeline de punta a punta, maneja verbose y rutas temporales, consenso KNN+OCR, estrategias de fallback y resolución final.
- utilities.py: utilidades de imagen + FuzzyEdgeDetector (configuración, LUT, convoluciones y pipeline difuso).
- tinyfuzzy.py: primitivas de lógica difusa (MFs, Mamdani, Sugeno, centroide, FCM), NumPy puro.
- thresholding.py: umbralización con soporte de progreso.
- connected_components.py: etiquetado BFS 8-conexo y selección de mayor componente.
- rectify.py: estimación de esquinas, cálculo de H (DLT) y warp de perspectiva bilineal.
- cell_extraction.py: división en celdas con padding y espejo binario opcional.
- digit_classifier.py: HOG+kNN, dataset (OpenCV o sintético), consenso por variantes y parámetros robustos.
- ocr.py: OCR de respaldo con Tesseract y preprocesado ajustado a celdas pequeñas.
- sudoku_solver.py: solver MRV+backtracking con mensajes opcionales.

Entradas/Salidas esperadas:
- Entrada: f2.jpg (o FILE_NAME.jpg).
- Salidas (VERBOSE=True): *_edges.jpg, *_thresh.jpg, *_largest.jpg, *_rectified.jpg, *_rectified_bin.jpg, directorios extracted_cells/ y extracted_cells_bin/, y *_grid.npy. Imprime cuadrícula reconocida y solución.
- Salidas (VERBOSE=False): imprime únicamente la solución; los intermedios se guardan en temporales.

## V. Limitations - Limitaciones actuales
- Reconocimiento de dígitos: el clasificador HOG+kNN y el OCR de respaldo (Tesseract) no alcanzan aún exactitud suficiente de forma consistente para la imagen demo. La precisión está condicionada por ruido, grosor de líneas, calidad de recortes, contraste y escalado.
- Respaldo “demo”: se incluye un puzzle conocido para f2 a fin de demostrar que el solucionador funciona; este respaldo se activa solo si los intentos con datos reconocidos fallan.
- Consolas Windows: se implementó un “fallback ASCII” para evitar errores Unicode (por ejemplo, emojis en mensajes ricos) en codificaciones no UTF-8.

## VI. Results & Usage - Resultados y Uso
Requisitos:
- Python 3.9+; librerías: numpy, Pillow, tqdm, rich, opencv-python, pytesseract.
- Tesseract instalado (para OCR); en Windows puede requerir configurar tesseract_cmd (ver ocr.py).

Ejecución:
```
python main.py
```
Ajustes rápidos en main.py:
- FILE_NAME = "f2"
- VERBOSE = True | False

## VII. Conclusions & Future Work - Conclusiones y Trabajo Futuro
- El enfoque difuso para detección de bordes aporta una etapa de preprocesado interpretable y controlable.
- La exactitud de lectura de dígitos requiere mejoras: datasets reales, aumento de datos, arquitecturas CNN ligeras (p. ej., MobileNetV3/SVHN/LeNet) y calibración del umbral de tinta.
- Mejoras de pipeline: morfología para rejilla/dígitos, estimación de esquinas robusta (RANSAC/cuadriláteros), y mayor integración de métricas de calidad por celda.

## VIII. Acknowledgments - Agradecimientos
- Autor: Kevin Esguerra Cardona — kevin.esguerra@utp.edu.co
- Asistente: proyecto desarrollado con ayuda del modelo Codex (OpenAI) mediante la interfaz Codex CLI.

## IX. License - Licencia
Este proyecto se distribuye bajo la licencia MIT. Ver el archivo LICENSE en la raíz del repositorio. A menos que se indique lo contrario, la licencia cubre todo el código fuente del proyecto.

---

### Apéndice A - Diagrama ASCII del pipeline
```
+------------------+
|  Imagen (.jpg)   |
+---------+--------+
          |
          v
  +-------+--------+
  |  Bordes difusos|
  +-------+--------+
          |
          v
  +-------+--------+
  | Umbralización  |
  +-------+--------+
          |
          v
  +-------+--------+
  |  Mayor comp.   |
  +-------+--------+
          |
          v
  +-------+--------+
  | Rectificación  |
  | (gris+bin)     |
  +-------+--------+
          |
          v
  +-------+--------+
  | Extracción de  |
  |   celdas       |
  +---+-------+----+
      |       |
      v       v
  +---+---+ +---+---+
  |  KNN | |  OCR  |
  +---+---+ +---+---+
      \       /
       v     v
      +-------+
      |Consenso|
      +---+----+
          |
          v
   +------+------+
   |  Solver MRV |
   +------+------+
          |
          v
   +------+------+
   |  Solución   |
   +-------------+
```

