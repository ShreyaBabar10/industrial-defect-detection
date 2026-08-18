from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import cv2


project_dir = Path(__file__).resolve().parents[2]

model_path = (
    project_dir
    / "models"
    / "neu_defect_detector"
    / "weights"
    / "best.pt"
)

image_dir = (
    project_dir
    / "Data"
    / "processed"
    / "NEU-DET-YOLO"
    / "images"
    / "val"
)

label_dir = (
    project_dir
    / "Data"
    / "processed"
    / "NEU-DET-YOLO"
    / "labels"
    / "val"
)

output_dir = (
    project_dir
    / "results"
    / "error_analysis"
)

visual_dir = output_dir / "visuals"

output_dir.mkdir(parents=True, exist_ok=True)
visual_dir.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}

IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.25


def load_yolo_boxes(label_file):

    boxes = []

    if not label_file.exists():
        return boxes

    with open(label_file, "r", encoding="utf-8") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            cx = float(values[1])
            cy = float(values[2])
            width = float(values[3])
            height = float(values[4])

            boxes.append(
                {
                    "class_id": class_id,
                    "box": [cx, cy, width, height],
                }
            )

    return boxes


def yolo_to_xyxy(box):

    cx, cy, width, height = box

    x1 = cx - width / 2
    y1 = cy - height / 2
    x2 = cx + width / 2
    y2 = cy + height / 2

    return [x1, y1, x2, y2]


def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width * intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def analyze_image(model, image_path):

    label_path = (
        label_dir
        / f"{image_path.stem}.txt"
    )

    ground_truths = load_yolo_boxes(label_path)

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    height, width = image.shape[:2]

    results = model.predict(
        source=str(image_path),
        imgsz=224,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]

    predictions = []

    for box in result.boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        xyxy = box.xyxy[0].tolist()

        normalized_box = [
            xyxy[0] / width,
            xyxy[1] / height,
            xyxy[2] / width,
            xyxy[3] / height,
        ]

        predictions.append(
            {
                "class_id": class_id,
                "confidence": confidence,
                "box": normalized_box,
            }
        )

    matched_ground_truths = set()

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for prediction in predictions:

        best_iou = 0.0
        best_index = None

        for index, ground_truth in enumerate(ground_truths):

            if index in matched_ground_truths:
                continue

            if (
                prediction["class_id"]
                != ground_truth["class_id"]
            ):
                continue

            ground_truth_box = yolo_to_xyxy(
                ground_truth["box"]
            )

            iou = calculate_iou(
                prediction["box"],
                ground_truth_box
            )

            if iou > best_iou:

                best_iou = iou
                best_index = index

        if (
            best_index is not None
            and best_iou >= IOU_THRESHOLD
        ):

            tp[prediction["class_id"]] += 1
            matched_ground_truths.add(best_index)

        else:

            fp[prediction["class_id"]] += 1

    for index, ground_truth in enumerate(ground_truths):

        if index not in matched_ground_truths:

            fn[ground_truth["class_id"]] += 1

    return tp, fp, fn


def main():

    if not model_path.exists():

        print(
            f"Model not found: {model_path}"
        )

        return

    if not image_dir.exists():

        print(
            f"Validation image directory not found: {image_dir}"
        )

        return

    model = YOLO(str(model_path))

    totals_tp = defaultdict(int)
    totals_fp = defaultdict(int)
    totals_fn = defaultdict(int)

    images = sorted(
        image_dir.glob("*.jpg")
    )

    print(
        f"Validation images found: {len(images)}"
    )

    for image_path in images:

        result = analyze_image(
            model,
            image_path
        )

        if result is None:
            continue

        tp, fp, fn = result

        for class_id, value in tp.items():
            totals_tp[class_id] += value

        for class_id, value in fp.items():
            totals_fp[class_id] += value

        for class_id, value in fn.items():
            totals_fn[class_id] += value

    report_file = (
        output_dir
        / "fp_fn_analysis.txt"
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Industrial Defect Detection - "
            "FP/FN Error Analysis\n"
        )

        file.write(
            "==================================================\n\n"
        )

        file.write(
            f"IoU threshold: {IOU_THRESHOLD}\n"
        )

        file.write(
            f"Confidence threshold: "
            f"{CONFIDENCE_THRESHOLD}\n\n"
        )

        file.write(
            f"{'Class':<22}"
            f"{'TP':>8}"
            f"{'FP':>8}"
            f"{'FN':>8}\n"
        )

        file.write("-" * 46 + "\n")

        for class_id, class_name in CLASS_NAMES.items():

            class_tp = totals_tp[class_id]
            class_fp = totals_fp[class_id]
            class_fn = totals_fn[class_id]

            total_tp += class_tp
            total_fp += class_fp
            total_fn += class_fn

            file.write(
                f"{class_name:<22}"
                f"{class_tp:>8}"
                f"{class_fp:>8}"
                f"{class_fn:>8}\n"
            )

        file.write("\n")
        file.write("-" * 46 + "\n")

        file.write(
            f"{'TOTAL':<22}"
            f"{total_tp:>8}"
            f"{total_fp:>8}"
            f"{total_fn:>8}\n"
        )

    print()
    print("===== FP/FN ERROR ANALYSIS =====")

    print(
        f"{'Class':<22}"
        f"{'TP':>8}"
        f"{'FP':>8}"
        f"{'FN':>8}"
    )

    print("-" * 46)

    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"{class_name:<22}"
            f"{totals_tp[class_id]:>8}"
            f"{totals_fp[class_id]:>8}"
            f"{totals_fn[class_id]:>8}"
        )

    print("-" * 46)

    print(
        f"{'TOTAL':<22}"
        f"{total_tp:>8}"
        f"{total_fp:>8}"
        f"{total_fn:>8}"
    )

    print()
    print(
        f"Report saved to: {report_file}"
    )


if __name__ == "__main__":
    main()