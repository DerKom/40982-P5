# Práctica 5. Detección y caracterización de caras

Este repositorio contiene la **Práctica 5** donde se utilizan técnicas de procesamiento de imágenes y aprendizaje automático para la detección y caracterización de caras humanas. Se implementan diferentes detectores faciales y se realizan análisis faciales utilizando conjuntos de datos y modelos preentrenados.

## Índice

- [Práctica 5. Detección y caracterización de caras](#práctica-5-detección-y-caracterización-de-caras)

  - [Referencias y bibliografía](#referencias-y-bibliografía)

## Librerías utilizadas

[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![Imutils](https://img.shields.io/badge/Imutils-FFD700?style=for-the-badge&logo=python)](https://github.com/jrosebr1/imutils)
[![MTCNN](https://img.shields.io/badge/MTCNN-FF4500?style=for-the-badge&logo=tensorflow)](https://github.com/ipazc/mtcnn)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![DeepFace](https://img.shields.io/badge/DeepFace-000000?style=for-the-badge&logo=python)](https://github.com/serengil/deepface)
[![Dlib](https://img.shields.io/badge/Dlib-008080?style=for-the-badge&logo=python)](http://dlib.net/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Scikit-image](https://img.shields.io/badge/Scikit--image-5C3EE8?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-image.org/)

## Autores
Este proyecto fue desarrollado por:

- [![GitHub](https://img.shields.io/badge/GitHub-Francisco%20Javier%20L%C3%B3pez%E2%80%93Dufour%20Morales-%23DC143C?style=flat-square&logo=github)](https://github.com/gitfrandu4)
- [![GitHub](https://img.shields.io/badge/GitHub-Marcos%20V%C3%A1zquez%20Tasc%C3%B3n-%232C3E50?style=flat-square&logo=github)](https://github.com/DerKom)

## Minijuego: Come Manzanas

### Como se Juega

1. **Inicia el juego** y asegúrate de estar en un entorno con buena iluminación.
2. **Junta tus manos** frente a la cámara para empezar (palma con palma).
3. Comenzaran a caer **Manzanas** y **Piedras** del cielo.
4. Debes comer las **Manzanas** y evitar que estas caigan al suelo.
5. NO deberías comer las **Piedras**.
6. Puedes tocar las manzanas y las piedras con las manos, esto genera un efecto que las dirije hacia tu boca.
6.1. Si la manzana se dirije a tu boca y NO la comes, pierdes una vida.
6.2. Si la piedra se dirije a tu boca y la comes, pierdes una vida.
9. Si pierdes todas las ** vidas (3)**, el juego termina.

### Explicación Del Código

#### Importación de librerías y módulos necesarios

```python
import cv2
import time
import mediapipe as mp
import numpy as np
import random
```

- **cv2**: Librería OpenCV para procesamiento de imágenes y video.
- **time**: Para manejar tiempos y pausas en el juego.
- **mediapipe as mp**: Librería de Google para detección y seguimiento de manos y rostro.
- **numpy as np**: Para operaciones numéricas y matrices.
- **random**: Para generar números aleatorios (posición de manzanas y piedras).

---

#### Inicialización de Mediapipe para detección de manos y rostro

```python
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
```

- **mp_hands y mp_face_mesh**: Inicializan los módulos de Mediapipe para manos y rostro.
- **hands**: Configura el detector de manos.
  - **static_image_mode=False**: Optimiza para video en tiempo real.
  - **max_num_hands=2**: Detecta hasta 2 manos.
  - **min_detection_confidence y min_tracking_confidence**: Umbrales de confianza para detección y seguimiento.
- **face_mesh**: Configura el detector de malla facial.
  - **max_num_faces=1**: Detecta solo un rostro.

---

#### Carga de imágenes de recursos (manzana, piedra y vida)

```python
# Cargar la imagen de la manzana
apple_img = cv2.imread('apple.png', cv2.IMREAD_UNCHANGED)
apple_img = cv2.resize(apple_img, (50, 50))  # Ajustar el tamaño de la manzana
```

- **cv2.imread**: Carga la imagen con transparencia (canal alfa) si está disponible.
- **cv2.resize**: Ajusta el tamaño de la imagen para que tenga dimensiones manejables en el juego.

Si no se puede cargar la imagen (por ejemplo, si el archivo no existe), se crea una representación por defecto:

```python
if apple_img is None or apple_img.shape[2] != 4:
    apple_img = np.zeros((50, 50, 4), dtype=np.uint8)
    cv2.circle(apple_img, (25, 25), 25, (0, 0, 255, 255), -1)
```

- Crea una imagen vacía con 4 canales (RGBA).
- Dibuja un círculo rojo (como una manzana) en el centro.

Lo mismo se hace para la piedra y el corazón (vida), adaptando las formas y colores correspondientes.

---

#### Carga de la imagen de la boca gigante

```python
big_mouth_img = cv2.imread('comemanzanas.png', cv2.IMREAD_UNCHANGED)
big_mouth_original_size = (150, 150)  # Tamaño por defecto si no se puede cargar
if big_mouth_img is None or big_mouth_img.shape[2] != 4:
    big_mouth_img = np.zeros((big_mouth_original_size[1], big_mouth_original_size[0], 4), dtype=np.uint8)
    cv2.circle(big_mouth_img, (big_mouth_original_size[0]//2, big_mouth_original_size[1]//2),
               min(big_mouth_original_size)//2, (0, 0, 255, 255), -1)  # Círculo rojo por defecto
```

- Carga la imagen que se mostrará cuando la boca esté abierta.
- Si no se encuentra, crea una imagen por defecto (un círculo rojo).

---

#### Parámetros del juego

```python
lives = 3  # Vidas del jugador
apples = []  # Lista de manzanas en pantalla
stones = []  # Lista de piedras en pantalla

# Intervalos de aparición
apple_spawn_interval = 5  # Segundos entre apariciones de manzanas
stone_spawn_interval = 8  # Segundos entre apariciones de piedras

# Tiempos de última aparición
last_apple_spawn_time = 0
last_stone_spawn_time = 0

game_started = False  # Indicador de inicio del juego
game_over = False     # Indicador de fin del juego
start_time = time.time()  # Tiempo de inicio del juego

# Dificultad
difficulty_increase_interval = 15  # Cada 15 segundos aumenta la dificultad
apple_fall_speed = 2  # Velocidad inicial de caída de las manzanas
stone_fall_speed = 2  # Velocidad inicial de caída de las piedras

# Límites en pantalla
max_apples_on_screen = 1  # Número máximo inicial de manzanas en pantalla
max_stones_on_screen = 0  # Las piedras comienzan a aparecer después
```

- Se definen las variables que controlarán el estado del juego, la dificultad y los objetos en pantalla.

---

#### Configuración de la captura de video

```python
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print('Error al abrir la cámara')
        exit(0)
```

- **cv2.VideoCapture(0)**: Intenta abrir la cámara principal.
- Si no funciona, intenta con **cv2.VideoCapture(1)** (otra cámara).
- Si no se puede abrir ninguna, muestra un mensaje de error y sale.

---

#### Función para superponer imágenes con transparencia

```python
def overlay_image_alpha(img, img_overlay, pos):
    # ...
```

- Esta función permite superponer una imagen con canal alfa (transparencia) sobre otra.
- **img**: Imagen de fondo (frame actual).
- **img_overlay**: Imagen a superponer (manzana, piedra, boca gigante).
- **pos**: Posición (x, y) donde se superpondrá la imagen.

La función maneja casos donde la imagen se sale de los límites y mezcla los canales de color teniendo en cuenta la transparencia.

---

#### Clase GameObject

```python
class GameObject:
    def __init__(self, x, y, fall_speed, obj_type='apple'):
        # ...
```

- Representa objetos en el juego: manzanas y piedras.
- **Atributos**:
  - **x, y**: Posición actual del objeto.
  - **state**: Estado del objeto ('falling' o 'to_mouth').
  - **fall_speed**: Velocidad de caída.
  - **path**: Trayectoria cuando se dirige hacia la boca.
  - **current_path_index**: Índice actual en la trayectoria.
  - **type**: Tipo de objeto ('apple' o 'stone').

#### Métodos de GameObject

1. **update_position(self)**:
   - Actualiza la posición del objeto.
   - Si está cayendo, incrementa **y** según la velocidad de caída.
   - Si está moviéndose hacia la boca, sigue la trayectoria predefinida.
   - Devuelve 'continue' si debe seguir actualizándose, o 'reached_mouth' si ha llegado a la boca.

2. **generate_parabola(self, mouth_x, mouth_y)**:
   - Genera una trayectoria parabólica desde la posición actual hasta la boca.
   - Calcula puntos intermedios para simular el movimiento hacia la boca.

---

#### Variables para el estado de la boca y la caja del rostro

```python
mouth_open = False  # Indica si la boca está abierta
current_face_bbox = None  # Almacena la caja delimitadora del rostro
```

---

#### Bucle principal del juego

```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # ...
```

- Se inicia un bucle infinito que captura frames de la cámara.
- Si no se puede leer un frame, se rompe el bucle.

#### Procesamiento del frame

```python
# Flip horizontal para efecto espejo
frame = cv2.flip(frame, 1)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

- **cv2.flip(frame, 1)**: Invierte horizontalmente el frame para que actúe como un espejo.
- **cv2.cvtColor**: Convierte el frame de BGR a RGB, formato que requiere Mediapipe.

#### Procesamiento de detección de manos y rostro

```python
# Procesar detección de manos
hand_results = hands.process(rgb_frame)

# Procesar detección de rostro
face_results = face_mesh.process(rgb_frame)
```

- Se procesan las detecciones de manos y rostro utilizando Mediapipe.

#### Obtener dimensiones del frame

```python
h, w, _ = frame.shape
```

- Se extraen las dimensiones del frame para escalado y posicionamiento.

#### Reiniciar variables

```python
current_face_bbox = None
hand_positions = []
mouth_coords = {}
```

- **current_face_bbox**: Se reinicia la caja delimitadora del rostro.
- **hand_positions**: Lista para almacenar las posiciones de las manos detectadas.
- **mouth_coords**: Diccionario para almacenar las coordenadas de la boca.

---

#### Detección y procesamiento de manos

```python
if hand_results.multi_hand_landmarks:
    for hand_landmarks in hand_results.multi_hand_landmarks:
        # Obtener coordenadas de los puntos de la mano
        for lm in hand_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            hand_positions.append((x, y))
        # Dibujar puntos de la mano
        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
```

- Si se detectan manos, se recorren los puntos de referencia (**landmarks**) de cada mano.
- Se convierten las coordenadas normalizadas (0 a 1) a píxeles multiplicando por el ancho y alto del frame.
- Se almacenan las posiciones en **hand_positions**.
- Se dibujan las conexiones de la mano en el frame para visualización.

---

#### Detección y procesamiento del rostro y la boca

```python
if face_results.multi_face_landmarks:
    for face_landmarks in face_results.multi_face_landmarks:
        # Obtener coordenadas de los labios
        lip_top = face_landmarks.landmark[13]
        lip_bottom = face_landmarks.landmark[14]
        lip_left = face_landmarks.landmark[78]
        lip_right = face_landmarks.landmark[308]
        # ...
```

- Se extraen puntos específicos de los labios superiores e inferiores y las comisuras.
- Se convierten las coordenadas a píxeles.

##### Cálculo de la apertura de la boca

```python
# Calcular apertura de la boca
mouth_opening = abs(top_lip_y - bottom_lip_y)

# Umbral para considerar boca abierta
if mouth_opening > 7:
    mouth_open = True
else:
    mouth_open = False
```

- Se calcula la distancia vertical entre el labio superior e inferior.
- Si esta distancia supera un umbral (7 píxeles), se considera que la boca está abierta.

##### Coordenadas de la boca

```python
# Coordenadas de la boca
mouth_coords = {'x': int((left_lip_x + right_lip_x) / 2),
                'y': int((top_lip_y + bottom_lip_y) / 2)}
```

- Se calcula el centro de la boca promediando las coordenadas horizontales y verticales.

##### Dibujar puntos de la boca

```python
cv2.circle(frame, (mouth_coords['x'], mouth_coords['y']), 2, (0, 255, 0), -1)
```

- Se dibuja un pequeño círculo verde en el centro de la boca para visualización.

##### Calcular la caja delimitadora del rostro

```python
x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
min_x, max_x = min(x_coords), max(x_coords)
min_y, max_y = min(y_coords), max(y_coords)
```

- Se obtienen todas las coordenadas x e y de los puntos del rostro.
- Se calcula el mínimo y máximo para definir la caja delimitadora.

##### Almacenar la caja delimitadora actual

```python
current_face_bbox = (min_x, min_y, max_x, max_y)
```

- Se guarda la caja delimitadora para uso posterior (por ejemplo, cuando los objetos se dirigen hacia la boca).

##### Ajustar y superponer la boca gigante

```python
# Ajustar el tamaño de la boca gigante según el rostro
big_mouth_width = int(face_width * 1.6)
big_mouth_height = int(face_height * 1.6)
resized_big_mouth = cv2.resize(big_mouth_img, (big_mouth_width, big_mouth_height))

# Calcular posición para centrar la boca gigante
mouth_center_x = int((min_x + max_x) / 2)
mouth_center_y = int((min_y + max_y) / 2)
top_left_x = mouth_center_x - big_mouth_width // 2
top_left_y = mouth_center_y - big_mouth_height // 2

# Superponer la boca gigante si la boca está abierta
if mouth_open:
    frame = overlay_image_alpha(frame, resized_big_mouth, (top_left_x, top_left_y))
```

- Se ajusta el tamaño de la imagen de la boca gigante para que se adapte al tamaño del rostro detectado.
- Se calcula la posición superior izquierda para centrar la boca gigante en el rostro.
- Si la boca está abierta, se superpone la imagen de la boca gigante sobre el frame.

---
#### Estado inicial del juego: esperando el gesto de inicio

```python
if not game_started and not game_over:
    if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2:
        # Calcular distancia entre las dos primeras posiciones detectadas
        x1, y1 = hand_positions[0]
        x2, y2 = hand_positions[1]
        palm_distance = np.hypot(x2 - x1, y2 - y1)
        # Si las palmas están cerca (umbral ajustable)
        if palm_distance < 27:
            game_started = True
            start_time = time.time()
    else:
        cv2.putText(frame, "Junta tus manos para comenzar", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
```

- Si el juego no ha comenzado, se espera a que el jugador junte sus manos.
- Se detecta si hay dos manos y se calcula la distancia entre las dos primeras posiciones detectadas.
- Si la distancia es menor que un umbral (27 píxeles), se considera que las manos están juntas y el juego comienza.
- Si no, se muestra el mensaje "Junta tus manos para comenzar".

---

#### Lógica principal del juego cuando está en marcha

```python
if game_started and not game_over:
    current_time = time.time()
    elapsed_time = int(current_time - start_time)
    # Mostrar temporizador
    cv2.putText(frame, f"Tiempo: {elapsed_time}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    # Aumentar dificultad cada cierto tiempo
    # ...
```

- Se calcula el tiempo transcurrido desde que inició el juego.
- Se muestra el temporizador en pantalla.
- Se incrementa la dificultad cada **difficulty_increase_interval** segundos:
  - Aumenta la velocidad de caída de manzanas y piedras.
  - Disminuye el intervalo de aparición de manzanas.
  - Aumenta el número máximo de manzanas y piedras en pantalla.

#### Generación de nuevas manzanas y piedras

```python
# Generar nuevas manzanas si es necesario
if len(apples) < max_apples_on_screen and current_time - last_apple_spawn_time > apple_spawn_interval:
    apple_x = random.randint(25, w - 25)
    apples.append(GameObject(apple_x, -25, apple_fall_speed, obj_type='apple'))
    last_apple_spawn_time = current_time
```

- Si hay menos manzanas en pantalla que el máximo permitido y ha pasado suficiente tiempo desde la última aparición, se genera una nueva manzana en una posición horizontal aleatoria.

Lo mismo se hace para las piedras, pero comienzan a aparecer después de 20 segundos de juego.

---

#### Actualización y dibujo de manzanas

```python
for apple in apples[:]:
    status = apple.update_position()
    # Dibujar manzana
    frame = overlay_image_alpha(frame, apple_img, (apple.x - 25, apple.y - 25))
    # Lógica de colisiones y estado de la manzana
    # ...
```

- Se itera sobre la lista de manzanas y se actualiza su posición.
- Se dibuja la manzana en su posición actual.
- Se verifica si la manzana ha tocado el suelo, en cuyo caso se resta una vida y se elimina la manzana.
- Se verifica si la manzana colisiona con una mano:
  - Si es así y la boca está abierta, se genera una trayectoria parabólica hacia la boca.
  - El estado de la manzana cambia a 'to_mouth'.

#### Manzanas dirigiéndose hacia la boca

- Si la manzana está en estado 'to_mouth', se actualiza su posición siguiendo la trayectoria generada.
- Si la manzana ha llegado a la boca:
  - Si la boca está abierta, la manzana se consume.
  - Si la boca está cerrada, se resta una vida.

---

#### Lógica similar para las piedras

- Las piedras se manejan de manera similar a las manzanas, pero:

  - No restan vidas si caen al suelo.
  - Si una piedra llega a la boca, resta una vida.
  - Las piedras comienzan a aparecer después de cierto tiempo.

---

#### Dibujo de las vidas (corazones)

```python
for i in range(lives):
    frame = overlay_image_alpha(frame, life_img, (w - (i + 1) * 40, 10))
```

- Se dibujan tantos corazones como vidas restantes en la esquina superior derecha.

---

#### Manejo del fin del juego

```python
if game_over:
    cv2.putText(frame, "Juego Terminado", (w // 2 - 200, h // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 255), 3)
    cv2.putText(frame, f"Tiempo total: {total_time}s", (w // 2 - 150, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    cv2.putText(frame, "Presiona 'R' para reiniciar", (w // 2 - 200, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
```

- Si el juego ha terminado (vidas agotadas), se muestra un mensaje de fin de juego, el tiempo total jugado y la opción de reiniciar presionando 'R'.

---

#### Mostrar el frame y manejar la entrada de teclado

```python
cv2.imshow('Minijuego: Come la Manzana', frame)

# Esperar por tecla de salida
tec = cv2.waitKey(1)
if tec & 0xFF == 27:  # Esc para salir
    break
elif tec & 0xFF == ord('r') and game_over:
    # Reiniciar juego
    # ...
```

- Se muestra el frame actualizado en una ventana llamada 'Minijuego: Come la Manzana'.
- Se espera por una entrada de teclado:
  - Si se presiona 'Esc' (código ASCII 27), se sale del juego.
  - Si se presiona 'R' y el juego ha terminado, se reinicia el juego reseteando todas las variables a sus valores iniciales.

---

#### Cierre de la aplicación

```python
# Cerrar cámara y ventanas
cap.release()
cv2.destroyAllWindows()
```

- Al salir del bucle principal, se libera la cámara y se cierran todas las ventanas de OpenCV.

---

## Referencias y bibliografía

- OpenCV Documentation: [docs.opencv.org](https://docs.opencv.org/)
- Matplotlib Documentation: [matplotlib.org](https://matplotlib.org/stable/contents.html)
- Imutils Documentation: [github.com/jrosebr1/imutils](https://github.com/jrosebr1/imutils)
- MTCNN Documentation: [github.com/ipazc/mtcnn](https://github.com/ipazc/mtcnn)
- TensorFlow Documentation: [tensorflow.org](https://www.tensorflow.org/api_docs)
- DeepFace Documentation: [github.com/serengil/deepface](https://github.com/serengil/deepface)
- Dlib Documentation: [dlib.net](http://dlib.net/)
- Scikit-learn Documentation: [scikit-learn.org](https://scikit-learn.org/stable/documentation.html)
- Scikit-image Documentation: [scikit-image.org](https://scikit-image.org/docs/stable/)
