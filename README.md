🎭 Real-Time Emotion Detection Using AI

This project detects human emotions in real time using Python, OpenCV, and DeepFace.
It captures live video from a webcam, detects faces, and classifies emotions such as Happy, Sad, Angry, Surprised, Fearful, Disgusted, and Neutral using deep learning models.

🧠 Overview

Facial expressions are one of the most powerful forms of non-verbal communication.
This project leverages Artificial Intelligence (AI) and Computer Vision (CV) to automatically recognize human emotions. By integrating OpenCV for face detection and DeepFace for emotion recognition, it enhances Human–Computer Interaction (HCI) and demonstrates how machines can understand human emotions.

🎯 Objectives

Detect human faces from live webcam video using OpenCV’s Haar Cascade Classifier.

Analyze detected faces using DeepFace for emotion classification.

Recognize seven basic emotions: Happy, Sad, Angry, Fearful, Disgusted, Surprised, and Neutral.

Display the predicted emotion in real time on the webcam video feed.

Explore real-world applications of emotion detection in education, healthcare, marketing, and security.

⚙️ System Requirements
🖥️ Hardware Requirements

Processor: Intel i3 or higher

RAM: Minimum 4 GB

Hard Disk: 500 MB free space

Webcam: Integrated or external (640x480 resolution)

Graphics: NVIDIA GPU (optional for faster performance)

💻 Software Requirements

Operating System: Windows 10/11, Linux, or macOS

Programming Language: Python 3.8 or higher

IDE/Editor: Visual Studio Code or PyCharm

Required Libraries:

opencv-python
deepface
tensorflow
keras
numpy

🚀 Installation and Execution
Step 1: Clone the Repository

Open your terminal or command prompt and run:

git clone https://github.com/Mujammil777/Emotion-Detection-Using-AI.git
cd Emotion-Detection-Using-AI

Step 2: Install Required Libraries

Install all dependencies using pip:

pip install opencv-python deepface tensorflow keras numpy

Step 3: Run the Program

Run the Python script to start real-time emotion detection:

python emotion_detection.py


Once you run the program, your webcam will open, and the system will start detecting your face and predicting your emotion in real time.

🧩 Source Code Example
import cv2
from deepface import DeepFace

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = rgb_frame[y:y + h, x:x + w]
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        emotion_confidence = result[0]['emotion'][dominant_emotion]
        label = f"{dominant_emotion.capitalize()} ({emotion_confidence:.2f}%)"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Real-Time Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

📸 Output Screens
Stage	Description
System Initialization	Webcam starts capturing live video feed.
Face Detection	Faces are detected and outlined with rectangles.
Emotion Recognition	Detected emotions appear as labels on each face.
Multiple Faces	Multiple people’s emotions are recognized simultaneously.
Program Termination	Press q to quit safely and close the webcam window.
💡 Features

Real-time emotion detection with live webcam feed.

Automatic face detection using Haar Cascade Classifier.

Deep learning-based emotion recognition using pre-trained CNN models (VGG-Face, ResNet, Facenet).

High accuracy and fast performance.

Multi-face emotion recognition.

Simple and user-friendly interface.

🌍 Applications

Education: Measure student engagement in online classes.

Healthcare: Track patient stress, anxiety, or depression.

Marketing: Analyze customer emotions during product testing.

Security: Detect suspicious or aggressive behavior.

Entertainment: Create emotion-aware games or applications.

Customer Service: Build empathetic chatbots and assistants.

🧭 Future Scope

Integration with IoT devices: Control lights or music based on user mood.

Web and Mobile Apps: Develop portable versions for real-time emotion detection.

Speech & Multi-Modal Recognition: Combine face + voice emotion analysis.

Improved Accuracy: Train with larger datasets for cultural and lighting variations.

Mental Health Monitoring: Track long-term emotional patterns for therapy.

Dashboard Integration: Visualize emotions with analytics charts.

📚 References

OpenCV Documentation

DeepFace GitHub Repository

TensorFlow Documentation

Keras API

Ian Goodfellow, Yoshua Bengio, and Aaron Courville, Deep Learning, MIT Press, 2016.

Richard Szeliski, Computer Vision: Algorithms and Applications, Springer, 2022.

👨‍💻 Author

Shaik Mujammil
B.Tech (CSE), K.S.R.M College of Engineering, Kadapa
GitHub: Mujammil777

Project: Real-Time Emotion Detection Using AI

✅ This project demonstrates how Artificial Intelligence can understand human emotions and enhance real-time human-computer interaction.
