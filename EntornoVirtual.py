import cv2
import mediapipe as mp
import numpy as np

# Inicializar la cámara y el lienzo
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

canvas = np.zeros((height, width, 3), dtype=np.uint8)
prev_index = None

# Inicializar la detección de dedos con MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Función para calcular el grosor de la línea en función de la distancia
def calcular_grosor(distancia, max_grosor, max_distancia):
    # Ajustar el factor de escala para controlar la rapidez del cambio de grosor
    factor_escala = 6 # Puedes ajustar este valor según sea necesario

    # Calcular el grosor de manera exponencial en función de la distancia
    grosor = max_grosor * np.exp(-factor_escala * distancia / max_distancia)
    return grosor

def calcular_color(grosor, max_grosor):
    # Calcular el valor de verde en función del grosor
    verde = int(255 * (grosor / max_grosor))
    # Crear el color verde con variación de intensidad
    return (0, verde, 0)


# Bucle Principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reflejar el fotograma horizontalmente para obtener el modo espejo
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obtener la posición del dedo índice
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convertir las coordenadas del dedo índice a píxeles en el lienzo
            index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)

            # Calcular el rectángulo delimitador solo para el dedo índice
            x = index_x - 10 
            y = index_y - 10 
            w = 20  # Ancho del rectángulo
            h = 20  # Altura del rectángulo

            # Calcular la distancia del dedo a la cámara (coordenada z)
            distance = index_tip.z * width

            max_thickness = 10

            # Ajustar el grosor de la línea en función de la distancia
            line_thickness = int(calcular_grosor(distance, 3, width))

            # Calcular el color verde en función del grosor
            line_color = calcular_color(line_thickness, max_thickness)

            # Dibujar el rectángulo en la imagen original
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Dibujar una línea desde las coordenadas anteriores del dedo índice si hay coordenizadas anteriores
            if prev_index is not None:
                cv2.line(canvas, prev_index, (index_x, index_y), line_color, line_thickness)
                 
            # Actualizar las coordenadas anteriores del dedo índice
            prev_index = (index_x, index_y)

    # Suavizar el trazo antes de mostrarlo
    if prev_index is not None:
        trazo_x = [prev_index[0], index_x]
        trazo_y = [prev_index[1], index_y]
        trazo_x_smooth = cv2.GaussianBlur(np.array(trazo_x).astype(np.float32), (5, 5), 0)
        trazo_y_smooth = cv2.GaussianBlur(np.array(trazo_y).astype(np.float32), (5, 5), 0)
        for i in range(len(trazo_x_smooth) - 1):
            pt1 = (int(trazo_x_smooth[i]), int(trazo_y_smooth[i]))
            pt2 = (int(trazo_x_smooth[i + 1]), int(trazo_y_smooth[i + 1]))
            cv2.line(canvas, pt1, pt2, line_color, line_thickness)

    cv2.imshow('Canvas', canvas)
    cv2.imshow('Camera', frame)

    # Verificar si se presiona la tecla 'p' para limpiar el lienzo
    key = cv2.waitKey(1)
    if key == ord('p'):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
    elif key == 27:  # Salir si se presiona la tecla 'Esc'
        break

cv2.destroyAllWindows()
cap.release()
