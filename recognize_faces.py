import cv2
import os
import numpy as np
import json
import sys
import time

# Define model path
model_path = os.path.abspath("lbph_model.xml")  # Get absolute path

# Load LBPH Face Recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Check if model file exists
if not os.path.exists(model_path):
    print(json.dumps({"error": f"Trained model not found at {model_path}"}))
    sys.exit(1)

# Load trained model
face_recognizer.read(model_path)

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow
if not cap.isOpened():
    print(json.dumps({"error": "Could not open webcam"}))
    sys.exit(1)

print(f"📂 Model loaded from: {model_path}")
print("🔍 Opening camera...")

try:
    # Ensure the first frame is captured before starting the timer
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to capture first frame")

    cv2.imshow("Face Recognition", frame)  # Show the first frame
    cv2.waitKey(500)  # Short delay to display the frame

    start_time = time.time()  # Start timing after the first frame
    timeout = 5  # Run for 5 seconds
    recognized_faces = []

    print(f"⏳ Recognizing faces for {timeout} seconds...")

    while True:
        ret, frame = cap.read()
        if not ret:
            raise Exception("Failed to capture frame")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]

            try:
                label, confidence = face_recognizer.predict(face_crop)
                if confidence < 50:  # Lower confidence means better match
                    recognized_faces.append({"name": f"Person_{label}", "confidence": round(confidence, 2)})
                    print(f"✅ Recognized: Person_{label} (Confidence: {confidence:.2f})")

                    # **IMMEDIATE STOP**
                    cap.release()
                    cv2.destroyAllWindows()
                    print(json.dumps({"recognized_faces": recognized_faces}))
                    sys.exit(0)

            except Exception as e:
                print(json.dumps({"error": f"Prediction error: {str(e)}"}))

        # Draw rectangle around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Stop after timeout
        if time.time() - start_time > timeout:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(json.dumps({"error": str(e)}))



finally:
    # **Forcefully release the camera**
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

# If no faces were recognized
if not recognized_faces:
    recognized_faces.append({"message": "No faces recognized"})

# Return result as JSON
print(json.dumps({"recognized_faces": recognized_faces}))
