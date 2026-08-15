from pathlib import Path
import cv2
import albumentations as A


# Project paths
project_dir = Path(__file__).resolve().parents[2]

input_images = (
    project_dir
    / "Data"
    / "processed"
    / "NEU-DET-YOLO"
    / "images"
    / "train"
)

input_labels = (
    project_dir
    / "Data"
    / "processed"
    / "NEU-DET-YOLO"
    / "labels"
    / "train"
)

output_images = (
    project_dir
    / "Data"
    / "augmented"
    / "images"
)

output_labels = (
    project_dir
    / "Data"
    / "augmented"
    / "labels"
)


# Augmentation pipeline
augmentation = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),
        A.GaussNoise(p=0.2),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"]
    )
)


def read_yolo_labels(label_file):
    boxes = []
    class_labels = []

    with open(label_file, "r") as file:
        for line in file:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])
            box = [float(value) for value in values[1:]]

            class_labels.append(class_id)
            boxes.append(box)

    return boxes, class_labels


def save_labels(label_file, boxes, class_labels):
    with open(label_file, "w") as file:
        for class_id, box in zip(class_labels, boxes):
            values = [class_id] + list(box)

            file.write(
                " ".join(f"{value:.6f}" for value in values)
                + "\n"
            )


def augment_dataset():

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted(input_images.glob("*.jpg"))

    print(f"Training images found: {len(image_files)}")

    created = 0

    for image_file in image_files:

        label_file = input_labels / f"{image_file.stem}.txt"

        if not label_file.exists():
            print("Label not found:", image_file.name)
            continue

        image = cv2.imread(str(image_file))

        if image is None:
            print("Could not read:", image_file.name)
            continue

        boxes, class_labels = read_yolo_labels(label_file)

        result = augmentation(
            image=image,
            bboxes=boxes,
            class_labels=class_labels
        )

        augmented_image = result["image"]
        augmented_boxes = result["bboxes"]
        augmented_classes = result["class_labels"]

        output_image_file = (
            output_images / f"{image_file.stem}_aug.jpg"
        )

        output_label_file = (
            output_labels / f"{image_file.stem}_aug.txt"
        )

        cv2.imwrite(
            str(output_image_file),
            augmented_image
        )

        save_labels(
            output_label_file,
            augmented_boxes,
            augmented_classes
        )

        created += 1

    print(f"Augmented images created: {created}")


if __name__ == "__main__":
    augment_dataset()