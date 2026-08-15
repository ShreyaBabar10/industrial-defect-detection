from pathlib import Path
import random
import cv2


project_dir = Path(__file__).resolve().parents[2]

image_dir = project_dir / "Data" / "augmented" / "images"
label_dir = project_dir / "Data" / "augmented" / "labels"

output_dir = project_dir / "Data" / "inspection"
output_dir.mkdir(parents=True, exist_ok=True)


class_names = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches"
}


def draw_boxes(image, label_file):

    height, width = image.shape[:2]

    with open(label_file, "r") as file:
        lines = file.readlines()

    for line in lines:

        values = line.strip().split()

        if len(values) != 5:
            continue

        class_id = int(float(values[0]))

        center_x = float(values[1]) * width
        center_y = float(values[2]) * height
        box_width = float(values[3]) * width
        box_height = float(values[4]) * height

        x1 = int(center_x - box_width / 2)
        y1 = int(center_y - box_height / 2)
        x2 = int(center_x + box_width / 2)
        y2 = int(center_y + box_height / 2)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        label = class_names.get(class_id, "unknown")

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    return image


def main():

    images = list(image_dir.glob("*.jpg"))

    if not images:
        print("No augmented images found.")
        return

    samples = random.sample(
        images,
        min(10, len(images))
    )

    for image_file in samples:

        label_file = label_dir / f"{image_file.stem}.txt"

        if not label_file.exists():
            continue

        image = cv2.imread(str(image_file))

        if image is None:
            continue

        image = draw_boxes(image, label_file)

        output_file = output_dir / image_file.name

        cv2.imwrite(
            str(output_file),
            image
        )

        print("Saved:", output_file.name)

    print()
    print("Inspection completed.")
    print("Open the Data/inspection folder to review the samples.")


if __name__ == "__main__":
    main()