from pathlib import Path
import json
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CONTENT_TYPE_LATEST,
    generate_latest,
)
from onnxruntime import InferenceSession


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


class PLCDetection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class PLCPayload(BaseModel):
    defect_detected: bool
    defect_count: int
    detections: list[PLCDetection]


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
    plc_payload: PLCPayload


class PredictionHistoryItem(BaseModel):
    timestamp: str
    filename: str
    model: str
    confidence_threshold: float
    inference_time_ms: float
    image_size: ImageSize
    detection_count: int
    prediction_image: str
    prediction_url: str
    detections: list[Detection]
    plc_payload: PLCPayload | None = None


class PredictionHistoryResponse(BaseModel):
    count: int
    predictions: list[PredictionHistoryItem]


project_dir = Path(__file__).resolve().parents[2]

model_path = (
    project_dir
    / "models"
    / "neu_defect_detector_256"
    / "weights"
    / "best.onnx"
)

output_dir = project_dir / "results" / "api_predictions"
output_dir.mkdir(parents=True, exist_ok=True)

history_file = output_dir / "prediction_history.jsonl"

MAX_FILE_SIZE = 5 * 1024 * 1024
IMAGE_SIZE = 256

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
}

DEFAULT_CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


prediction_requests = Counter(
    "defect_detection_requests_total",
    "Total number of prediction requests",
)

successful_predictions = Counter(
    "defect_detection_successful_predictions_total",
    "Total number of successful predictions",
)

failed_predictions = Counter(
    "defect_detection_failed_predictions_total",
    "Total number of failed prediction requests",
)

inference_latency = Histogram(
    "defect_detection_inference_latency_seconds",
    "YOLO inference latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

service_uptime = Gauge(
    "defect_detection_service_uptime_seconds",
    "Time since the API service started",
)

service_start_time = time.time()


app = FastAPI(
    title="Industrial Defect Detection API",
    version="1.0.0",
    description="ONNX-based industrial surface defect detection API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if not model_path.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {model_path}"
    )


model = InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"],
)

model_input = model.get_inputs()[0]
model_input_name = model_input.name

model_output = model.get_outputs()[0]
model_output_name = model_output.name


def get_class_names():
    try:
        metadata = model.get_modelmeta().custom_metadata_map
        names = metadata.get("names")

        if names:
            try:
                parsed_names = json.loads(names)

                if isinstance(parsed_names, dict):
                    return [
                        parsed_names.get(
                            str(index),
                            parsed_names.get(index, str(index)),
                        )
                        for index in range(len(parsed_names))
                    ]

                if isinstance(parsed_names, list):
                    return parsed_names

            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    except Exception:
        pass

    return DEFAULT_CLASS_NAMES


CLASS_NAMES = get_class_names()


def letterbox_image(image):
    original_height, original_width = image.shape[:2]

    scale = min(
        IMAGE_SIZE / original_width,
        IMAGE_SIZE / original_height,
    )

    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))

    resized_image = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        114,
        dtype=np.uint8,
    )

    pad_x = (IMAGE_SIZE - resized_width) // 2
    pad_y = (IMAGE_SIZE - resized_height) // 2

    canvas[
        pad_y:pad_y + resized_height,
        pad_x:pad_x + resized_width
    ] = resized_image

    return canvas, scale, pad_x, pad_y


def preprocess_image(image):
    processed_image, scale, pad_x, pad_y = letterbox_image(image)

    processed_image = cv2.cvtColor(
        processed_image,
        cv2.COLOR_BGR2RGB,
    )

    processed_image = (
        processed_image.astype(np.float32) / 255.0
    )

    processed_image = np.transpose(
        processed_image,
        (2, 0, 1),
    )

    processed_image = np.expand_dims(
        processed_image,
        axis=0,
    )

    return processed_image, scale, pad_x, pad_y


def calculate_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection_width = np.maximum(0, x2 - x1)
    intersection_height = np.maximum(0, y2 - y1)

    intersection = intersection_width * intersection_height

    box_area = (
        max(0, box[2] - box[0])
        * max(0, box[3] - box[1])
    )

    boxes_area = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )

    union = box_area + boxes_area - intersection

    return intersection / (union + 1e-6)


def non_max_suppression(
    boxes,
    scores,
    class_ids,
    iou_threshold=0.45,
):
    if len(boxes) == 0:
        return []

    keep_indices = []

    for class_id in np.unique(class_ids):
        class_indices = np.where(
            class_ids == class_id
        )[0]

        class_boxes = boxes[class_indices]
        class_scores = scores[class_indices]

        order = np.argsort(class_scores)[::-1]

        while len(order) > 0:
            current = order[0]

            keep_indices.append(
                class_indices[current]
            )

            if len(order) == 1:
                break

            remaining = order[1:]

            ious = calculate_iou(
                class_boxes[current],
                class_boxes[remaining],
            )

            order = remaining[
                ious < iou_threshold
            ]

    return keep_indices


def postprocess_predictions(
    output,
    original_image,
    scale,
    pad_x,
    pad_y,
    confidence_threshold,
):
    predictions = np.asarray(output)

    if predictions.ndim == 3:
        predictions = predictions[0]

    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    if predictions.shape[1] < 5:
        return []

    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(
        class_scores,
        axis=1,
    )

    scores = class_scores[
        np.arange(len(class_scores)),
        class_ids,
    ]

    confidence_mask = (
        scores >= confidence_threshold
    )

    boxes = boxes[confidence_mask]
    scores = scores[confidence_mask]
    class_ids = class_ids[confidence_mask]

    if len(boxes) == 0:
        return []

    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    original_height, original_width = original_image.shape[:2]

    x1 = np.clip(x1, 0, original_width - 1)
    y1 = np.clip(y1, 0, original_height - 1)
    x2 = np.clip(x2, 0, original_width - 1)
    y2 = np.clip(y2, 0, original_height - 1)

    boxes = np.column_stack(
        (x1, y1, x2, y2)
    )

    keep_indices = non_max_suppression(
        boxes,
        scores,
        class_ids,
    )

    detections = []

    for index in keep_indices:
        class_id = int(class_ids[index])

        if class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
        else:
            class_name = f"class_{class_id}"

        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(
                float(scores[index]),
                4,
            ),
            "bounding_box": {
                "x1": round(float(boxes[index][0]), 2),
                "y1": round(float(boxes[index][1]), 2),
                "x2": round(float(boxes[index][2]), 2),
                "y2": round(float(boxes[index][3]), 2),
            },
        })

    return detections


def draw_detections(image, detections):
    annotated_image = image.copy()

    for detection in detections:
        box = detection["bounding_box"]

        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )

        label_width, label_height = label_size

        label_y1 = max(
            0,
            y1 - label_height - baseline - 5,
        )

        label_y2 = (
            label_y1
            + label_height
            + baseline
            + 5
        )

        label_x2 = x1 + label_width + 5

        cv2.rectangle(
            annotated_image,
            (x1, label_y1),
            (label_x2, label_y2),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            annotated_image,
            label,
            (x1 + 2, label_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated_image


def build_plc_payload(detections):
    plc_detections = []

    for detection in detections:
        box = detection["bounding_box"]

        plc_detections.append({
            "class_id": detection["class_id"],
            "class_name": detection["class_name"],
            "confidence": detection["confidence"],
            "x1": box["x1"],
            "y1": box["y1"],
            "x2": box["x2"],
            "y2": box["y2"],
        })

    return {
        "defect_detected": len(plc_detections) > 0,
        "defect_count": len(plc_detections),
        "detections": plc_detections,
    }


warmup_image = np.zeros(
    (IMAGE_SIZE, IMAGE_SIZE, 3),
    dtype=np.uint8,
)

warmup_tensor, _, _, _ = preprocess_image(
    warmup_image
)

model.run(
    [model_output_name],
    {
        model_input_name: warmup_tensor
    },
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
        "model": "best.onnx",
        "model_format": "ONNX",
        "model_exists": model_path.exists(),
        "image_size": IMAGE_SIZE,
        "supported_formats": ["jpg", "jpeg", "png"],
        "max_file_size_mb": 5,
    }


@app.get("/metrics")
def metrics():
    service_uptime.set(
        time.time() - service_start_time
    )

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    confidence: float = Query(
        0.25,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections",
    ),
):
    prediction_requests.inc()

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        failed_predictions.inc()

        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Please upload JPG or PNG.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        failed_predictions.inc()

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        failed_predictions.inc()

        raise HTTPException(
            status_code=413,
            detail="Image file is too large. Maximum allowed size is 5 MB.",
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        failed_predictions.inc()

        raise HTTPException(
            status_code=400,
            detail="Unable to decode the uploaded image.",
        )

    try:
        input_tensor, scale, pad_x, pad_y = preprocess_image(
            image
        )
    except Exception as exc:
        failed_predictions.inc()

        raise HTTPException(
            status_code=500,
            detail=f"Image preprocessing failed: {exc}",
        ) from exc

    start_time = time.perf_counter()

    try:
        outputs = model.run(
            [model_output_name],
            {
                model_input_name: input_tensor
            },
        )
    except Exception as exc:
        failed_predictions.inc()

        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {exc}",
        ) from exc

    inference_time = time.perf_counter() - start_time

    inference_latency.observe(inference_time)

    inference_time_ms = inference_time * 1000

    try:
        detections = postprocess_predictions(
            outputs[0],
            image,
            scale,
            pad_x,
            pad_y,
            confidence,
        )
    except Exception as exc:
        failed_predictions.inc()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction postprocessing failed: {exc}",
        ) from exc

    annotated_image = draw_detections(
        image,
        detections,
    )

    plc_payload = build_plc_payload(
        detections
    )

    original_name = Path(
        file.filename or "image.jpg"
    ).stem

    output_path = (
        output_dir
        / f"{original_name}_prediction.jpg"
    )

    success = cv2.imwrite(
        str(output_path),
        annotated_image,
    )

    if not success:
        failed_predictions.inc()

        raise HTTPException(
            status_code=500,
            detail="Failed to save prediction image.",
        )

    prediction_url = (
        str(request.base_url).rstrip("/")
        + f"/prediction/{output_path.name}"
    )

    prediction_record = {
        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "filename": file.filename,
        "model": "best.onnx",
        "confidence_threshold": confidence,
        "inference_time_ms": round(
            inference_time_ms,
            2,
        ),
        "image_size": {
            "width": image.shape[1],
            "height": image.shape[0],
        },
        "detection_count": len(detections),
        "prediction_image": str(output_path),
        "prediction_url": prediction_url,
        "detections": detections,
        "plc_payload": plc_payload,
    }

    try:
        with history_file.open(
            "a",
            encoding="utf-8",
        ) as file_handle:
            file_handle.write(
                json.dumps(prediction_record)
                + "\n"
            )
    except OSError as exc:
        failed_predictions.inc()

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save prediction history: {exc}",
        ) from exc

    successful_predictions.inc()

    return {
        "success": True,
        "filename": file.filename,
        "model": "best.onnx",
        "confidence_threshold": confidence,
        "inference_time_ms": round(
            inference_time_ms,
            2,
        ),
        "image_size": {
            "width": image.shape[1],
            "height": image.shape[0],
        },
        "detection_count": len(detections),
        "prediction_image": str(output_path),
        "prediction_url": prediction_url,
        "detections": detections,
        "plc_payload": plc_payload,
    }


@app.get(
    "/plc/latest",
    response_model=PLCPayload,
)
def get_latest_plc_payload():
    if not history_file.exists():
        return {
            "defect_detected": False,
            "defect_count": 0,
            "detections": [],
        }

    latest_record = None

    try:
        with history_file.open(
            "r",
            encoding="utf-8",
        ) as file_handle:
            for line in file_handle:
                line = line.strip()

                if line:
                    latest_record = json.loads(line)

    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read latest prediction: {exc}",
        ) from exc

    if latest_record is None:
        return {
            "defect_detected": False,
            "defect_count": 0,
            "detections": [],
        }

    if latest_record.get("plc_payload") is not None:
        return latest_record["plc_payload"]

    return build_plc_payload(
        latest_record.get("detections", [])
    )


@app.get(
    "/predictions",
    response_model=PredictionHistoryResponse,
)
def get_prediction_history():
    if not history_file.exists():
        return {
            "count": 0,
            "predictions": [],
        }

    predictions = []

    try:
        with history_file.open(
            "r",
            encoding="utf-8",
        ) as file_handle:
            for line in file_handle:
                line = line.strip()

                if line:
                    predictions.append(
                        json.loads(line)
                    )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read prediction history: {exc}",
        ) from exc

    return {
        "count": len(predictions),
        "predictions": predictions,
    }


@app.get("/prediction/{filename}")
def get_prediction(filename: str):
    file_path = output_dir / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Prediction image not found.",
        )

    return FileResponse(
        path=file_path,
        media_type="image/jpeg",
    )