# Import required libraries
import cv2
from deepface import DeepFace

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (for detection)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert grayscale to RGB (for DeepFace analysis)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis using DeepFace
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Extract dominant emotion and confidence score
        dominant_emotion = result[0]['dominant_emotion']
        emotion_confidence = result[0]['emotion'][dominant_emotion]

        # Prepare label with emotion and confidence
        label = f"{dominant_emotion.capitalize()} ({emotion_confidence:.2f}%)"

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display emotion label above the rectangle
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show the frame with emotion labels
    cv2.imshow('Real-Time Emotion Detection', frame)

    # Exit the program when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
