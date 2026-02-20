from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from datetime import datetime

# 🔥 Allow Ultralytics model class for new PyTorch versions
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception:
    pass

app = FastAPI(
    title="Weapon Detection API",
    description="Upload an image to detect weapons using YOLO model.",
    version="2.0.0"
)

# Load model at startup
model = YOLO("weapon_detector.pt")

@app.get("/")
def home():
    return {"status": "Weapon Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Convert bytes to image
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image file"}

        # Run inference
        results = model(img)[0]

        detections = []
        class_names = model.names

        for box in results.boxes:
            confidence = float(box.conf)

            # Confidence threshold
            if confidence < 0.6:
                continue

            class_id = int(box.cls)
            bbox = box.xyxy.tolist()

            detections.append({
                "label": class_names[class_id],
                "confidence": round(confidence, 3),
                "bbox": bbox
            })

        # Timestamp only if weapon detected
        detected_time = None
        if len(detections) > 0:
            detected_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "total_detections": len(detections),
            "weapon_detected": len(detections) > 0,
            "detected_at": detected_time,
            "detections": detections
        }

    except Exception as e:
        return {"error": str(e)}