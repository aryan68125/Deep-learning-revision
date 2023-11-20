# Assuming you have a pre-trained emotion detection model (replace 'path_to_model.h5' with your model path)
from tensorflow.keras.models import load_model

#TODO Load the pre trained model
emotion_model = load_model('path_to_model.h5')

import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load a pre-trained face detection model (Haarcascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y+h, x:x+w]

        # Resize the face image to the required input size for your emotion detection model
        face_resized = cv2.resize(face_roi, (48, 48))

        # Preprocess the face image for the emotion detection model
        face_array = image.img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = preprocess_input(face_array)

        # Predict emotions using your emotion detection model
        emotion_predictions = emotion_model.predict(face_array)

        # Get the predicted emotion label
        emotion_label = np.argmax(emotion_predictions)

        # Draw the bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
