# __init__.py
# Diego Iván Martínez Escobar
# Programa de Honores
# Universidad de las Américas Puebla

from latas_template_matching.Detector import Detector
from latas_template_matching import DetectorOtsu, DetectorTemplate, DetectorObjetosOscuros, DetectorOscuroOtsu, DetectorAguaHSV

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def choose_video():
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Ask the user to choose a video file (filter common video file extensions)
    file_path = filedialog.askopenfilename(
        title="Selecciona un archivo de video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm")]
    )

    if not file_path:
        raise Exception("No se ha seleccionado un archivo de video.")

    return file_path

def __main__():
    parametros = {
        "min_area": 1000,
        "lower_black": np.array([0, 0, 0], dtype=np.uint8),
        "upper_black": np.array([180, 255, 50], dtype=np.uint8),
        "lower_blue": np.array([30, 60, 120], dtype=np.uint8),
        "upper_blue": np.array([144, 200, 255], dtype=np.uint8)
    }
    detector = Detector(DetectorObjetosOscuros.detector_objetos_oscuros, DetectorAguaHSV.detector_agua_hsv, parametros)

    # Open the video file
    cap = cv2.VideoCapture("/home/diegoivan/Documentos/Programa de Honores/Template Matching/compressed/20250315_235359000_iOS.mp4")

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get detections for the current frame
        detections = detector.detectar_objetos(frame)

        # Iterate over each detection and draw the rectangle on the frame
        for (clase, x1, y1, x2, y2) in detections:
            cv2.putText(frame, clase, (x2, y2), cv2.FONT_ITALIC, 1, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow("Detections", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    __main__()