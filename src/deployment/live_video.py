from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort


PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "neu_defect_detector_256"
    / "weights"
    / "best.onnx"
)

IMAGE_SIZE = 256
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def preprocess(frame):
    image = cv2.resize(
        frame,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    return np.expand_dims(image, axis=0)


def detect(output, frame):
    predictions = np.asarray(output[0])

    if predictions.ndim == 3:
        predictions = predictions[0]

    if predictions.shape == (1344, 10):
        predictions = predictions.T

    if predictions.shape != (10, 1344):
        raise ValueError(
            f"Unexpected ONNX output shape: {predictions.shape}"
        )

    boxes = predictions[:4]
    class_scores = predictions[4:]

    scores = class_scores.max(axis=0)
    class_ids = class_scores.argmax(axis=0)

    height, width = frame.shape[:2]

    scale_x = width / IMAGE_SIZE
    scale_y = height / IMAGE_SIZE

    candidate_boxes = []
    candidate_scores = []
    candidate_classes = []

    for i, score in enumerate(scores):

        score = float(score)

        if score < CONFIDENCE_THRESHOLD:
            continue

        x_center, y_center, box_width, box_height = boxes[:, i]

        x1 = max(
            0,
            min(
                IMAGE_SIZE - 1,
                float(x_center - box_width / 2)
            )
        )

        y1 = max(
            0,
            min(
                IMAGE_SIZE - 1,
                float(y_center - box_height / 2)
            )
        )

        x2 = max(
            0,
            min(
                IMAGE_SIZE - 1,
                float(x_center + box_width / 2)
            )
        )

        y2 = max(
            0,
            min(
                IMAGE_SIZE - 1,
                float(y_center + box_height / 2)
            )
        )

        candidate_boxes.append([
            int(x1),
            int(y1),
            int(x2 - x1),
            int(y2 - y1),
        ])

        candidate_scores.append(score)
        candidate_classes.append(int(class_ids[i]))

    if not candidate_boxes:
        return frame, 0

    indices = cv2.dnn.NMSBoxes(
        candidate_boxes,
        candidate_scores,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD,
    )

    if len(indices) == 0:
        return frame, 0

    indices = np.asarray(indices).reshape(-1)

    for i in indices:

        class_id = candidate_classes[i]
        confidence = candidate_scores[i]

        x, y, box_width, box_height = candidate_boxes[i]

        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + box_width) * scale_x)
        y2 = int((y + box_height) * scale_y)

        label = (
            f"{CLASS_NAMES[class_id]} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return frame, len(indices)


def main():

    if not MODEL_PATH.exists():
        print("ONNX model not found.")
        return

    print("Loading ONNX model...")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    print("ONNX model loaded successfully.")
    print("Starting webcam...")
    print("Press Q to exit.")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Unable to open webcam.")
        return

    previous_time = time.perf_counter()

    try:

        while True:

            success, frame = camera.read()

            if not success:
                break

            input_image = preprocess(frame)

            start_time = time.perf_counter()

            output = session.run(
                None,
                {input_name: input_image}
            )

            inference_time = (
                time.perf_counter() - start_time
            ) * 1000

            frame, detection_count = detect(
                output,
                frame
            )

            current_time = time.perf_counter()

            fps = 1 / (
                current_time - previous_time
            )

            previous_time = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Inference: {inference_time:.2f} ms",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Detections: {detection_count}",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Industrial Defect Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Live detection stopped.")


if __name__ == "__main__":
    main()