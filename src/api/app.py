from pathlib import Path
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from ultralytics import YOLO


project_dir = Path(__file__).resolve().parents[2]

model_path = (
    project_dir
    / "models"
    / "neu_defect_detector_256"
    / "weights"
    / "best.onnx"
)


app = FastAPI(
    title="Industrial Defect Detection API",
    version="1.0.0",
    description="ONNX-based industrial surface defect detection API.",
)


if not model_path.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {model_path}"
    )

model = YOLO(str(model_path), task="detect")


@app.get("/")
def home():
    return {
        "message": "Industrial Defect Detection API is running."
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "ONNX",
        "image_size": 256
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(
        0.25,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections"
    )
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode the uploaded image."
        )

    start_time = time.perf_counter()

    results = model.predict(
        source=image,
        imgsz=256,
        conf=confidence,
        verbose=False
    )

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    result = results[0]
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        detection_confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_id": class_id,
            "class_name": result.names[class_id],
            "confidence": round(
                detection_confidence,
                4
            ),
            "bounding_box": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            }
        })

    return {
        "success": True,
        "filename": file.filename,
        "model": "best.onnx",
        "confidence_threshold": confidence,
        "inference_time_ms": round(
            inference_time_ms,
            2
        ),
        "image_size": {
            "width": image.shape[1],
            "height": image.shape[0]
        },
        "detection_count": len(detections),
        "detections": detections
    }