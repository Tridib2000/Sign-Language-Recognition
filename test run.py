import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from keras.models import model_from_json

# Load the model's architecture
with open(r'E:\research paper for finland\model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

# Load the model's weights
model.load_weights(r'E:\research paper for finland\model_weights.h5')

# Load the labels (ensure the labels are correctly formatted in labels.csv)
labels_csv_path = r'E:\research paper for finland\labels.csv'
labels_df = pd.read_csv(labels_csv_path, header=None)
class_labels = labels_df.iloc[:, 0].tolist()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Open video capture (for webcam live feed)
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(image_rgb)

    # Check if hand(s) are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand skeleton on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the hand landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Preprocess the landmarks for input to the model
            input_data = np.array([landmarks]).reshape(-1, 63, 1)

            # Make a prediction using the model
            prediction = model.predict(input_data)
            predicted_class = class_labels[np.argmax(prediction)]
            predicted_class_str = str(predicted_class)  # Convert predicted class to string

            # Display the predicted class on the frame
            cv2.putText(frame, predicted_class_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Sign Recognition', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
