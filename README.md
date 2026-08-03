# 🔫 Weapon Detection API using YOLOv8 & FastAPI

A high-performance REST API for real-time weapon detection built using **YOLOv8**, **FastAPI**, and **OpenCV**. The API accepts an image, performs object detection using a custom-trained YOLO model, and returns whether a weapon is detected along with the current date and time.

This project is ideal for AI-powered surveillance systems, smart security cameras, and automated threat detection applications.

---

# 🚀 Features

- 🔫 Detects weapons in uploaded images
- ⚡ High-speed inference with YOLOv8
- 🌐 REST API built using FastAPI
- 🖼️ Supports image uploads
- 📊 Confidence threshold filtering (0.6)
- 📅 Returns detection timestamp
- 🛡️ Simple JSON response for easy integration
- 📦 Ready for deployment

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| FastAPI | REST API Framework |
| YOLOv8 (Ultralytics) | Object Detection Model |
| OpenCV | Image Processing |
| NumPy | Image Array Conversion |
| Uvicorn | ASGI Server |

---

# 📁 Project Structure

```
.
├── main.py                  # FastAPI application
├── weapon_detector.pt       # Trained YOLOv8 model
├── requirements.txt
├── README.md
└── sample_images/
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/weapon-detection-api.git

cd weapon-detection-api
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the API

```bash
uvicorn main:app --reload
```

By default, the API will be available at:

```
http://127.0.0.1:8000
```

---

# 📚 API Documentation

FastAPI automatically generates interactive documentation.

Swagger UI

```
http://127.0.0.1:8000/docs
```

ReDoc

```
http://127.0.0.1:8000/redoc
```

---

# 📤 API Endpoint

## POST `/predict`

Uploads an image and checks whether a weapon is present.

### Request

Content-Type:

```
multipart/form-data
```

Parameter

| Name | Type | Required |
|------|------|----------|
| file | Image | Yes |

---

# ✅ Success Response

```json
{
    "weapon_detected": true,
    "date": "2026-08-03",
    "time": "14:35:28"
}
```

---

# ❌ Invalid Image Response

```json
{
    "error": "Invalid image"
}
```

---

# ⚠️ Error Response

```json
{
    "error": "Detailed exception message"
}
```

---

# 🧠 How It Works

## Step 1

Client uploads an image.

↓

## Step 2

FastAPI receives the uploaded file.

↓

## Step 3

Image is converted into a NumPy array.

↓

## Step 4

OpenCV decodes the image.

↓

## Step 5

YOLOv8 performs object detection.

↓

## Step 6

Each detected object is checked against the confidence threshold (0.6).

↓

## Step 7

API returns whether a weapon was detected along with the current date and time.

---

# 🏗 System Architecture

```
           Client
              │
              ▼
       Upload Image
              │
              ▼
          FastAPI API
              │
              ▼
      Read Uploaded File
              │
              ▼
 NumPy Buffer Conversion
              │
              ▼
 OpenCV Image Decoding
              │
              ▼
      YOLOv8 Inference
              │
              ▼
 Confidence Threshold Check
              │
              ▼
        JSON Response
```

---

# 🔍 Detection Logic

The model performs inference on the uploaded image and evaluates the confidence score for each detected object.

A weapon is considered detected if **any detection has a confidence score greater than or equal to 0.60**.

Example:

```python
is_detected = any(
    float(box.conf) >= 0.6
    for box in results.boxes
)
```

---

# 🧩 Core Components

## FastAPI

Provides a lightweight, high-performance REST API.

```python
app = FastAPI()
```

---

## YOLOv8 Model

Loads the custom-trained weapon detection model.

```python
model = YOLO("weapon_detector.pt")
```

---

## OpenCV

Decodes uploaded image bytes into an image matrix.

```python
img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
```

---

## NumPy

Converts uploaded bytes into an array before OpenCV decoding.

```python
npimg = np.frombuffer(contents, np.uint8)
```

---

# 📊 Sample Workflow

```
Upload Image
      │
      ▼
Receive Bytes
      │
      ▼
Convert to NumPy
      │
      ▼
Decode using OpenCV
      │
      ▼
YOLOv8 Detection
      │
      ▼
Confidence Filtering
      │
      ▼
Return JSON Response
```

---

# 🧪 Testing the API

Using cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-F "file=@image.jpg"
```

---

Using Python

```python
import requests

url = "http://127.0.0.1:8000/predict"

with open("image.jpg", "rb") as f:
    response = requests.post(
        url,
        files={"file": f}
    )

print(response.json())
```

---

# 📦 Dependencies

```
fastapi
uvicorn
ultralytics
opencv-python
numpy
python-multipart
```

Install manually:

```bash
pip install fastapi uvicorn ultralytics opencv-python numpy python-multipart
```

---

# 🌟 Future Improvements

- 🎥 Real-time webcam detection
- 📹 Video stream processing
- ☁️ Cloud deployment (AWS, Azure, GCP)
- 🖥️ Streamlit or React dashboard
- 📸 Save annotated detection images
- 🚨 Email/SMS alerts on weapon detection
- 🔔 Push notifications
- 📊 Detection analytics dashboard
- 🐳 Docker support
- 🔐 Authentication & API keys

---

# 💡 Use Cases

- Smart CCTV surveillance
- School and college security
- Airport security systems
- Railway station monitoring
- Office and workplace surveillance
- Public event security
- Automated threat detection
- AI-powered safety solutions

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature/new-feature
```

3. Commit your changes

```bash
git commit -m "Added new feature"
```

4. Push to GitHub

```bash
git push origin feature/new-feature
```

5. Open a Pull Request.

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Akhil Vikram Singh**

If you found this project useful, consider giving it a ⭐ on GitHub!
