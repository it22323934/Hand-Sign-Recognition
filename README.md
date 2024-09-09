# Hand Sign Recognition Using Mediapipe and Machine Learning

This project implements a hand sign recognition system using a webcam and a pre-trained machine learning model. The system uses Mediapipe for hand landmark detection and a RandomForestClassifier model to predict hand signs.

## Features
- Real-time hand landmark detection using Mediapipe.
- Hand sign recognition with a pre-trained RandomForestClassifier model.
- Displays hand landmarks on the webcam feed.
- Supports multiple hand sign classes.
  
## Requirements

To run the project, you'll need the following dependencies:

### Python Libraries
- OpenCV (`cv2`) for capturing and processing webcam images.
- Mediapipe (`mediapipe`) for detecting hand landmarks.
- Scikit-learn (`sklearn`) for using the trained RandomForestClassifier model.
- Numpy (`numpy`) for processing data in arrays.
- Pickle (`pickle`) for loading the pre-trained model.

### Install Dependencies
You can install the required libraries using `pip`:

```bash
pip install opencv-python mediapipe scikit-learn numpy

## How to Use

### Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repo_url>
```

### Download the Trained Model

Ensure that you have the trained model stored as `model.p` in the root directory of the project.

### Run the Hand Sign Recognition

Run the `inference_classifier.py` script to start the hand sign recognition:

```bash
python inference_classifier.py
```

### Using the Webcam

- The script will access your webcam to capture frames in real-time.
- Hand landmarks will be detected and visualized on the webcam feed.
- The system will predict and print the corresponding hand sign based on the hand detected.

### Exit the Program

Press `q` on your keyboard to exit the webcam feed and close the program.

## File Structure

```bash
/data/                     # Directory containing training data images
/model.p                   # Trained model using RandomForestClassifier
/inference_classifier.py    # Main Python script for real-time hand sign recognition
/README.md                 # This file
```

## Model Details

- The hand sign recognition model is trained using a `RandomForestClassifier` from the Scikit-learn library.
- The training data consists of images of hands performing different signs, and MediaPipe is used to extract hand landmarks from these images.
- The model expects **126 features** as input (x, y, z coordinates for 21 hand landmarks on both hands). If only **one hand** is detected, the input is padded with zeros to match the expected 126 features.

## Labels

The current hand sign recognition model supports the following labels:

- **A** for the sign representing the letter "A".
- **B** for the sign representing the letter "B".
- **L** for the sign representing the letter "L".

These labels can be extended by training the model with additional hand sign data.

## Troubleshooting

### `ValueError: X has 63 features, but RandomForestClassifier is expecting 126 features`

- This error occurs when only one hand is detected. The code pads the data with zeros to ensure that the input size is 126 features, corresponding to both hands.

### Webcam Not Detected:

- Make sure that your webcam is properly connected and accessible. You can check your camera device using OpenCV by running the following simple capture script:

```python
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## Future Improvements

- Extend the model to support more hand signs.
- Optimize the detection for faster real-time processing.
- Add a graphical interface to display predictions on the webcam feed.
- Improve hand detection accuracy using dynamic image modes (non-static).

## License

This project is open-source and available under the MIT License.

