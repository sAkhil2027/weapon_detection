from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime

app = FastAPI()

# Load model
model = YOLO("weapon_detector.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Core processing (unchanged)
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image"}

        results = model(img)[0]
        
        # Check if any detection meets your 0.6 confidence threshold
        is_detected = any(float(box.conf) >= 0.6 for box in results.boxes)
        
        # Optimized direct output
        return {
            "weapon_detected": is_detected,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S")
        }

    except Exception as e:
        return {"error": str(e)}