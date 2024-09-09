import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

while True:
    data_aux = []
    x_=[]
    y_=[]
    
    # Capture frame from webcam
    ret, frame = cap.read()
    
    H,W, _ = frame.shape
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
        # Collect data for both hands if present
        for hand_landmarks in results.multi_hand_landmarks:   
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z  # Add z coordinate
                data_aux.extend([x, y, z])
                x_.append(x)
                y_.append(y)
        
        # Pad data with zeros if only one hand is detected
        if len(data_aux) == 63:  # One hand detected
            data_aux.extend([0] * 63)  # Pad with zeros to match the 126 features
        
        x1=int(min(x_) * W)
        y1=int(min(y_) * H)
        x2=int(max(x_) * W)
        y2=int(max(y_) * H)
        
        # Convert data_aux to numpy array and predict
        prediction = model.predict([np.asarray(data_aux)])
        
        # Corrected line to access dictionary
        prediction_character = labels_dict[int(prediction[0])]
        print(f"Prediction: {prediction_character}")
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),4)
        cv2.putText(frame, prediction_character, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
        
    # Display the webcam feed
    cv2.imshow('frame', frame)
    # Break on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
