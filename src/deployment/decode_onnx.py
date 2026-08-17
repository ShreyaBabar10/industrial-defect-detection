from pathlib import Path

import cv2
import numpy as np


project_dir = Path(__file__).resolve().parents[2]

prediction_path = (
    project_dir
    / "results"
    / "onnx_predictions"
    / "crazing_241_prediction.npy"
)

image_path = (
    project_dir
    / "Data"
    / "processed"
    / "NEU-DET-YOLO"
    / "images"
    / "val"
    / "crazing_241.jpg"
)

output_dir = project_dir / "results" / "onnx_predictions"

class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def decode_predictions():

    if not prediction_path.exists():
        print("Prediction file not found.")
        return

    if not image_path.exists():
        print("Test image not found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = np.load(prediction_path)[0]

    boxes = predictions[:4]
    class_scores = predictions[4:]

    scores = class_scores.max(axis=0)
    class_ids = class_scores.argmax(axis=0)

    confidence_threshold = 0.25
    nms_threshold = 0.45

    image = cv2.imread(str(image_path))

    if image is None:
        print("Unable to read test image.")
        return

    scale_x = image.shape[1] / 256
    scale_y = image.shape[0] / 256

    candidate_boxes = []
    candidate_scores = []
    candidate_classes = []

    for index, score in enumerate(scores):

        if score < confidence_threshold:
            continue

        x_center, y_center, width, height = boxes[:, index]

        x1 = float(x_center - width / 2)
        y1 = float(y_center - height / 2)
        x2 = float(x_center + width / 2)
        y2 = float(y_center + height / 2)

        x1 = max(0, min(255, x1))
        y1 = max(0, min(255, y1))
        x2 = max(0, min(255, x2))
        y2 = max(0, min(255, y2))

        candidate_boxes.append(
            [
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1),
            ]
        )
        candidate_scores.append(float(score))
        candidate_classes.append(int(class_ids[index]))

    print(f"Candidates above {confidence_threshold}: {len(candidate_boxes)}")

    if not candidate_boxes:
        print("No detections found.")
        return

    keep_indices = cv2.dnn.NMSBoxes(
        candidate_boxes,
        candidate_scores,
        confidence_threshold,
        nms_threshold,
    )

    if len(keep_indices) == 0:
        print("No detections remained after NMS.")
        return

    keep_indices = np.array(keep_indices).reshape(-1)

    print(f"Detections after NMS: {len(keep_indices)}")
    print()

    for index in keep_indices:

        class_id = candidate_classes[index]
        score = candidate_scores[index]

        x, y, width, height = candidate_boxes[index]

        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + width) * scale_x)
        y2 = int((y + height) * scale_y)

        label = f"{class_names[class_id]} {score:.2f}"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        print(
            f"{class_names[class_id]}: "
            f"{score:.3f} "
            f"box=[{x1}, {y1}, {x2}, {y2}]"
        )

    output_path = output_dir / "crazing_241_prediction.jpg"

    cv2.imwrite(str(output_path), image)

    print()
    print("Prediction visualization saved to:")
    print(output_path)


if __name__ == "__main__":
    decode_predictions()