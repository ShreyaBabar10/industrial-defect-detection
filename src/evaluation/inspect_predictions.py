from pathlib import Path
from ultralytics import YOLO


# Project root
project_dir = Path(__file__).resolve().parents[2]

# Trained model
model_path = (
    project_dir
    / "models"
    / "neu_defect_detector"
    / "weights"
    / "best.pt"
)

# Validation images
image_dir = (
    project_dir
    / "Data"
    / "processed"
    / "NEU-DET-YOLO"
    / "images"
    / "val"
)

# Prediction output
output_dir = project_dir / "results" / "error_analysis"


def inspect_predictions():

    if not model_path.exists():
        print("Trained model not found.")
        return

    if not image_dir.exists():
        print("Validation image folder not found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    class_names = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches"
    ]

    total_processed = 0

    for class_name in class_names:

        images = sorted(image_dir.glob(f"{class_name}_*.jpg"))
        selected_images = images[:5]

        print(f"{class_name}: {len(selected_images)} images")

        for image in selected_images:

            model.predict(
                source=str(image),
                imgsz=224,
                conf=0.25,
                save=True,
                project=str(output_dir),
                name="predictions",
                exist_ok=True,
                verbose=False
            )

            total_processed += 1

    print()
    print(f"Total images inspected: {total_processed}")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    inspect_predictions()