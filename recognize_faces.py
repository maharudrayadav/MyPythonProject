import cv2
import os
import numpy as np
import json
import sys
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.get("/recognize/{username}")
def recognize_face(username: str):
    """Recognizes a face using LBPH model stored in /model/{username}/"""
    
    # Define user-specific model path
    model_path = f"model/{username}/lbph_model_{username}.xml"
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Trained model not found for user: {username}")

    # Load LBPH Face Recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model_path)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open webcam")

    print(f"📂 Model loaded from: {model_path}")
    print("🔍 Opening camera...")

    start_time = time.time()
    timeout = 5  # Run for 5 seconds
    recognized_faces = []

    try:
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret:
                raise HTTPException(status_code=500, detail="Failed to capture frame")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_crop = gray[y:y + h, x:x + w]

                try:
                    label, confidence = face_recognizer.predict(face_crop)
                    if confidence < 50:  # Lower confidence means better match
                        recognized_faces.append({"name": f"{username}", "confidence": round(confidence, 2)})
                        cap.release()
                        cv2.destroyAllWindows()
                        return {"recognized_faces": recognized_faces}

                except Exception as e:
                    return {"error": f"Prediction error: {str(e)}"}

        return {"message": "No faces recognized"}

    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
