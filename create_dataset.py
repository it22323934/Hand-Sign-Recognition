import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hands model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Data directory
DATA_DIR = './data'
data = []
labels = []

# Loop through each subdirectory (class) in the DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    if os.path.isdir(os.path.join(DATA_DIR, dir_)):  # Check if it's a directory
        # Loop through each image in the class directory
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            img_full_path = os.path.join(DATA_DIR, dir_, img_path)

            # Read the image
            img = cv2.imread(img_full_path)
            if img is None:
                print(f"Error loading image: {img_full_path}")
                continue

            # Convert the image to RGB (MediaPipe requires RGB input)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the image with MediaPipe Hands
            results = hands.process(img_rgb)

            # Prepare the data_aux list to hold landmark coordinates
            data_aux = []

            if results.multi_hand_landmarks:
                # Extract x, y, and z coordinates of each hand landmark
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        z = hand_landmarks.landmark[i].z  # Including z-coordinate
                        data_aux.append(x)
                        data_aux.append(y)
                        data_aux.append(z)  # Append z-coordinate for completeness

                # Add data and corresponding label (class name)
                data.append(data_aux)
                labels.append(dir_)

            else:
                print(f"No hand detected in image: {img_full_path}")

# Close the hands model
hands.close()

# Save the data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Data processing complete. {len(data)} samples saved to 'data.pickle'.")
