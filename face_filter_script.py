import cv2
import numpy as np
import time
from retinaface import RetinaFace
from deepface import DeepFace  # Added DeepFace import

# Debug mode
debug = True  # Set to False to disable debug mode

# Load filter images with alpha channel
glasses = cv2.imread('gafas.png', cv2.IMREAD_UNCHANGED)
mustache = cv2.imread('bigote.png', cv2.IMREAD_UNCHANGED)
hat = cv2.imread('sombrero.png', cv2.IMREAD_UNCHANGED)

# Verify if filter images are loaded correctly
if glasses is None or mustache is None or hat is None:
    print("Error loading filter images.")
    exit()

# Connect to the webcam
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

# Initialize variables for face detection frequency
detection_interval = 3  # Detect faces every 3 frames
emotion_interval = 5  # Analyze emotions every 5 frames
faces = None
emotion_results = {}

# Flags to indicate which filters are active
use_glasses = True
use_mustache = True
use_hat = True

def overlay_image(frame, overlay, position):
    """Overlay an image with transparency over the frame."""
    x, y = position
    h, w = overlay.shape[:2]

    # Ensure the coordinates are within the frame
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return

    # Get region of interest on the frame
    roi = frame[y:y+h, x:x+w]

    # Separate the color and alpha channels
    overlay_img = overlay[:, :, :3]
    overlay_mask = overlay[:, :, 3]

    # Convert mask to 3 channels and normalize
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR) / 255.0
    background_mask = 1.0 - overlay_mask

    # Blend the overlay with the ROI
    roi = (overlay_img * overlay_mask + roi * background_mask).astype(np.uint8)

    # Put the blended image back into the frame
    frame[y:y+h, x:x+w] = roi

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Increment frame count
    frame_count += 1

    # Calculate FPS every 10 frames
    if frame_count % 10 == 0:
        end_time = time.time()
        fps = 10 / (end_time - start_time)
        start_time = end_time

    if debug:
        # Display FPS on the frame
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Detect faces every 'detection_interval' frames
    if frame_count % detection_interval == 0:
        faces = RetinaFace.detect_faces(frame)

    # Analyze emotions every 'emotion_interval' frames
    if frame_count % emotion_interval == 0:
        analyze_emotions = True
    else:
        analyze_emotions = False

    face_count = 0
    if isinstance(faces, dict):
        face_count = len(faces)
        for face_id, face in faces.items():
            facial_area = face['facial_area']
            landmarks = face['landmarks']
            x1, y1, x2, y2 = facial_area
            face_roi = frame[y1:y2, x1:x2]

            # Perform emotion analysis
            if analyze_emotions:
                try:
                    analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                        
                    emotion = analysis['dominant_emotion']
                    
                except Exception as e:
                    print(f"Emotion analysis failed: {e}")
                    emotion_results[face_id] = 'Unknown'
            else:
                emotion = emotion_results.get(face_id, 'Analyzing...')

            if debug:
                # Display the emotion label on the frame
                cv2.putText(frame, f'Emotion: {emotion}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Draw bounding rectangle
                face_width = x2 - x1
                face_height = y2 - y1
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (255, 0, 0), 2)
                cv2.putText(frame, f'Face: ({x1},{y1}) ({x2},{y2}) W:{face_width} H:{face_height}',
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                # Display landmarks with names and coordinates
                for point_name, point in landmarks.items():
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    cv2.putText(frame, f'{point_name} ({x},{y})', (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Calculate positions and sizes for filters
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            nose = landmarks['nose']
            mouth_left = landmarks['mouth_left']
            mouth_right = landmarks['mouth_right']

            eye_center = ((left_eye[0] + right_eye[0]) / 2,
                          (left_eye[1] + right_eye[1]) / 2)
            eye_width = int(abs(right_eye[0] - left_eye[0]) * 2)
            eye_height = int(eye_width * glasses.shape[0] / glasses.shape[1])
            eye_x = int(eye_center[0] - eye_width / 2)
            eye_y = int(eye_center[1] - eye_height / 2)

            # Adjust position and size of mustache
            mouth_center = ((mouth_left[0] + mouth_right[0]) / 2,
                            (mouth_left[1] + mouth_right[1]) / 2)
            mustache_width = int(abs(mouth_right[0] - mouth_left[0]) * 1.5)
            mustache_height = int(
                mustache_width * mustache.shape[0] / mustache.shape[1])
            mustache_x = int(mouth_center[0] - mustache_width / 2)
            mustache_y = int(mouth_center[1] - mustache_height / 1.4)

            # Calculate position and size of hat
            head_width = int(eye_width * 1.5)
            head_height = int(
                head_width * hat.shape[0] / hat.shape[1])
            head_x = int(eye_center[0] - head_width / 2)
            head_y = int(eye_y - head_height + eye_height // 2)

            # Resize and overlay filters based on user selection
            if use_glasses:
                glasses_resized = cv2.resize(
                    glasses, (eye_width, eye_height), interpolation=cv2.INTER_AREA)
                overlay_image(frame, glasses_resized, (eye_x, eye_y))

            if use_mustache:
                mustache_resized = cv2.resize(
                    mustache, (mustache_width, mustache_height), interpolation=cv2.INTER_AREA)
                overlay_image(frame, mustache_resized, (mustache_x, mustache_y))

            if use_hat:
                hat_resized = cv2.resize(
                    hat, (head_width, head_height), interpolation=cv2.INTER_AREA)
                overlay_image(frame, hat_resized, (head_x, head_y))

    if debug:
        # Display number of faces detected
        cv2.putText(frame, f'Faces detected: {face_count}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the result
    cv2.imshow('Fun Filter', frame)

    # Handle key events
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        # Exit with 'Esc' key
        break
    elif key == ord('d'):
        # Toggle debug mode with 'd' key
        debug = not debug
    elif key == ord('g'):
        # Toggle glasses filter with 'g' key
        use_glasses = not use_glasses
    elif key == ord('m'):
        # Toggle mustache filter with 'm' key
        use_mustache = not use_mustache
    elif key == ord('h'):
        # Toggle hat filter with 'h' key
        use_hat = not use_hat

cap.release()
cv2.destroyAllWindows()
