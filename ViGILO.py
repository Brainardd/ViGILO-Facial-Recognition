import cv2
import os
import dlib
import sys
import time
import tkinter as tk
from scipy.spatial import distance as dist
import numpy as np
from keras.models import load_model

# System Encoding Configuration
# ensures that the output encoding is set to UTF-8, which is important for displaying characters correctly
sys.stdout.reconfigure(encoding='utf-8')

# Constants
# defines constants used throughout the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(SCRIPT_DIR, "shape_predictor_68_face_landmarks.dat")
DROWSINESS_THRESHOLD = 0.225
DROWSINESS_DURATION = 7
NOTIFICATION_DELAY = 7

# Global Variables
# initializes the Dlib predictor and face detector, and sets up global variables to track notification and rest times
predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_detector = dlib.get_frontal_face_detector()
last_notification_time = 0
last_rest_time = None

# Load Emotion Recognition Model
# loads the pre-trained emotion recognition model and defines the emotion labels
EMOTION_MODEL_PATH = os.path.join(SCRIPT_DIR, "emotion_model.h5")
emotion_model = load_model(EMOTION_MODEL_PATH)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Face Preprocessing Function
# prepares the face image for emotion prediction by converting it to grayscale, resizing, normalizing, and reshaping it
def preprocess_face(face):
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray_face, (48, 48))
    normalized_face = resized_face / 255.0
    reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
    return reshaped_face

# Emotion Prediction Function
# predicts the emotion from the preprocessed face image and returns the corresponding label
def predict_emotion(face):
    preprocessed_face = preprocess_face(face)
    emotion_prediction = emotion_model.predict(preprocessed_face)
    max_index = np.argmax(emotion_prediction[0])
    emotion_label = emotion_labels[max_index]
    return emotion_label

# Popup Notification Function
# creates and displays a popup notification using tkinter when drowsiness is detected
def show_popup():
    root = tk.Tk()
    root.title("ViGILO Drowsiness Alert")

    # Calculate popup window position
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    popup_width = 300
    popup_height = 150
    x_position = (screen_width - popup_width) // 2
    y_position = (screen_height - popup_height) // 2

    # Set popup window geometry
    root.geometry(f"{popup_width}x{popup_height}+{x_position}+{y_position}")

    # Display notification
    label_text = "You seem drowsy!"
    label = tk.Label(root, text=label_text, font=("Arial", 14))
    label.pack(pady=10)

    def close_popup():
        root.destroy()

    # Okay button
    okay_button = tk.Button(root, text="Okay", command=close_popup)
    okay_button.pack(side=tk.LEFT, padx=20)

    # Take a Rest button
    rest_button = tk.Button(root, text="Take A Rest", command=close_popup)
    rest_button.pack(side=tk.RIGHT, padx=20)

    root.mainloop()

# Eye Aspect Ratio Calculation
# calculates the Eye Aspect Ratio (EAR) which is used to detect drowsiness
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Drowsiness Detection Function
# detects drowsiness based on the EAR and displays a popup notification if drowsiness is detected.
def detect_drowsiness(shape, frame, x, y):
    global last_notification_time, last_rest_time

    # extract the coordinates of the left and right eyes from the facial landmarks
    left_eye = shape[36:42]
    right_eye = shape[42:48]

    # calculate the Eye Aspect Ratio (EAR) for both eyes
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    
    # average the EAR of both eyes to get a single EAR value
    ear = (left_ear + right_ear) / 2.0

    # check if the EAR is below the drowsiness threshold
    if ear < DROWSINESS_THRESHOLD:
        current_time = time.time() # get the current time
        if last_rest_time is None:  # if first detection, initialize the rest time
            last_rest_time = current_time
        elif current_time - last_rest_time >= DROWSINESS_DURATION: # check if the drowsiness duration is exceeded
            if current_time - last_notification_time >= NOTIFICATION_DELAY: # check if the notification delay is exceeded
                show_popup() # show the drowsiness alert popup
                
    else:
        # reset the rest and notification times if no drowsiness is detected
        last_rest_time = None
        last_notification_time = 0

    # display the EAR on frame
    cv2.putText(frame, f'EAR: {ear:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Landmark Drawing Function
# draws landmarks and connecting lines on the face for visualization
def draw_landmarks(frame, shape):
    for (x, y) in shape:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
    ''' ( **I COMMENT SO IF ITS BETTER TO HAVE LINES FOR THE FACIAL RECOGNITION, I'LL UNCOMMENT**)
    # lines between landmarks for visualization
    for i in range(1, 17):
        cv2.line(frame, shape[i], shape[i - 1], (0, 255, 0), 1)
    for i in range(28, 30):
        cv2.line(frame, shape[i], shape[i - 1], (0, 255, 0), 1)
    for i in range(31, 36):
        cv2.line(frame, shape[i], shape[i - 1], (0, 255, 0), 1)
    for i in range(37, 42):
        cv2.line(frame, shape[i], shape[i - 1], (0, 255, 0), 1)
    for i in range(43, 48):
        cv2.line(frame, shape[i], shape[i - 1], (0, 255, 0), 1)
    for i in range(49, 60):
        cv2.line(frame, shape[i], shape[i - 1], (0, 255, 0), 1)
    for i in range(61, 68):
        cv2.line(frame, shape[i], shape[i - 1], (0, 255, 0), 1)
    '''

# Face Processing Function
# processes a detected face, draws landmarks, predicts the emotion, and dont detect drowsiness if emotion is "Happy"   
def process_region(frame, face):
    shape = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), face)
    shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

    # detects face
    x, y = face.left(), face.top()
    w, h = face.width(), face.height()
    draw_landmarks(frame, shape)

    # detects emotion
    padding = 20
    x1, y1 = max(x - padding, 0), max(y - padding, 0)
    x2, y2 = min(x + w + padding, frame.shape[1]), min(y + h + padding, frame.shape[0])
    face_region = frame[y1:y2, x1:x2]

    if face_region.size > 0:
        emotion = predict_emotion(face_region)

        # skip drowsiness detection if emotion is "Happy"
        if emotion.lower() != 'happy':
            detect_drowsiness(shape, frame, x, y)

        cv2.putText(frame, f'{emotion}', (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('ViGILO', frame)


# Main Function
# captures video from main webcam, processes each frame to detect faces, handles drowsiness and emotion detection
# if 'q' pressed = system close
def main():
    global last_rest_time
    cap = cv2.VideoCapture(0)  # Use the correct camera index
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return
    window_name = 'ViGILO'

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray_frame)

            if len(faces) > 0:
                face = faces[0]
                process_region(frame, face)
            else:
                cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

# execute program
if __name__ == "__main__":
    main()

