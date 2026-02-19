from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np

# Create FastAPI app with metadata (improves Swagger UI)
app = FastAPI(
    title="Weapon Detection API",
    description="Upload an image to detect weapons using YOLO model.",
    version="1.0.0"
)

# Load model ONCE at startup
model = YOLO("weapon_detector.pt")

@app.get("/")
def home():
    return {"status": "Weapon Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        contents = await file.read()

        # Convert to numpy array
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image file"}

        # Run inference
        results = model(img)[0]

        detections = []
        class_names = model.names  # class label mapping

        for box in results.boxes:
            confidence = float(box.conf)

            # Filter weak detections
            if confidence < 0.6:
                continue

            class_id = int(box.cls)
            bbox = box.xyxy.tolist()

            detections.append({
                "label": class_names[class_id],
                "confidence": round(confidence, 3),
                "bbox": bbox
            })

        return {
            "total_detections": len(detections),
            "detections": detections
        }

    except Exception as e:
        return {"error": str(e)}
