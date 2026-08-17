from pathlib import Path
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox


class ImageSize(BaseModel):
    width: int
    height: int


class PredictResponse(BaseModel):
    success: bool
    filename: str
    model: str
    confidence_threshold: float
    inference_time_ms: float
    image_size: ImageSize
    detection_count: int
    prediction_image: str
    prediction_url: str
    detections: list[Detection]


project_dir = Path(__file__).resolve().parents[2]

model_path = (
    project_dir
    / "models"
    / "neu_defect_detector_256"
    / "weights"
    / "best.onnx"
)

output_dir = (
    project_dir
    / "results"
    / "api_predictions"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


MAX_FILE_SIZE = 5 * 1024 * 1024

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
}


app = FastAPI(
    title="Industrial Defect Detection API",
    version="1.0.0",
    description="ONNX-based industrial surface defect detection API.",
)


if not model_path.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {model_path}"
    )

model = YOLO(
    str(model_path),
    task="detect"
)


warmup_image = np.zeros(
    (256, 256, 3),
    dtype=np.uint8
)

model.predict(
    source=warmup_image,
    imgsz=256,
    conf=0.25,
    verbose=False
)


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


@app.post(
    "/predict",
    response_model=PredictResponse
)
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(
        0.25,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections"
    )
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Please upload JPG or PNG."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image file is too large. Maximum allowed size is 5 MB."
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

    try:
        results = model.predict(
            source=image,
            imgsz=256,
            conf=confidence,
            verbose=False
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {exc}"
        ) from exc

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

    annotated_image = result.plot()

    original_name = Path(
        file.filename or "image.jpg"
    ).stem

    output_path = (
        output_dir
        / f"{original_name}_prediction.jpg"
    )

    cv2.imwrite(
        str(output_path),
        annotated_image
    )

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
        "prediction_image": str(output_path),
        "prediction_url": (
            f"/prediction/{output_path.name}"
        ),
        "detections": detections
    }


@app.get("/prediction/{filename}")
def get_prediction(filename: str):
    file_path = output_dir / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Prediction image not found."
        )

    return FileResponse(
        path=file_path,
        media_type="image/jpeg"
    )