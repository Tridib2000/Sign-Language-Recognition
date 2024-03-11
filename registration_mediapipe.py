import cv2
import mediapipe as mp
import os
import csv
import re

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
a='fox'
# Set the directory path to scan images
image_directory = rf'E:\..........\data\{a}'

# Set the directory path to save the CSV file
output_directory = r'E:\.........\train'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Create a CSV file path
csv_file = os.path.join(output_directory, f'{a}.csv')

# Process images and save hand landmarks data
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for filename in os.listdir(image_directory):
        if re.search(r'\.jpg$', filename, re.IGNORECASE):
            # Load image
            image_path = os.path.join(image_directory, filename)
            image = cv2.imread(image_path)

            # Convert the image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe Hands
            results = hands.process(image_rgb)

            # Check if hand(s) are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract the 21 hand landmarks
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append(landmark.x)
                        landmarks.append(landmark.y)
                        landmarks.append(landmark.z)

                    # Write the hand landmarks data to the CSV file
                    writer.writerow(landmarks)

# Release all resources
hands.close()

print(f"Hand landmarks data saved to {csv_file}")
