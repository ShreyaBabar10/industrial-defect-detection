from pathlib import Path


# Project directory
project_dir = Path(__file__).resolve().parents[2]

# Augmented dataset
image_dir = project_dir / "Data" / "augmented" / "images"
label_dir = project_dir / "Data" / "augmented" / "labels"


def validate_label(label_file):
    """Check whether a YOLO label file is valid."""

    with open(label_file, "r") as file:
        lines = file.readlines()

    if not lines:
        return False

    for line in lines:
        values = line.strip().split()

        # YOLO label should contain:
        # class_id x_center y_center width height
        if len(values) != 5:
            return False

        try:
            class_id = float(values[0])
            coordinates = [float(value) for value in values[1:]]
        except ValueError:
            return False

        # Class ID should be a whole number from 0 to 5
        if not class_id.is_integer():
            return False

        class_id = int(class_id)

        if class_id < 0 or class_id > 5:
            return False

        # YOLO coordinates must be normalized between 0 and 1
        for value in coordinates:
            if value < 0 or value > 1:
                return False

    return True


def main():

    images = list(image_dir.glob("*.jpg"))
    labels = list(label_dir.glob("*.txt"))

    valid_labels = 0
    invalid_labels = 0
    missing_labels = 0

    for image in images:

        label_file = label_dir / f"{image.stem}.txt"

        if not label_file.exists():
            missing_labels += 1
            continue

        if validate_label(label_file):
            valid_labels += 1
        else:
            invalid_labels += 1

    print("Augmented images:", len(images))
    print("Augmented labels:", len(labels))
    print("Valid labels:", valid_labels)
    print("Invalid labels:", invalid_labels)
    print("Missing labels:", missing_labels)


if __name__ == "__main__":
    main()