from pathlib import Path
from ultralytics import YOLO


# Project root
project_dir = Path(__file__).resolve().parents[2]

# Dataset configuration
data_config = project_dir / "configs" / "data.yaml"

# Trained YOLO model
model_path = (
    project_dir
    / "models"
    / "neu_defect_detector"
    / "weights"
    / "best.pt"
)


def evaluate_model():

    if not model_path.exists():
        print("Trained model not found.")
        print(f"Expected model: {model_path}")
        return

    print("Loading trained model...")
    print(f"Model: {model_path}")

    model = YOLO(str(model_path))

    print()
    print("Running validation...")
    print()

    results = model.val(
        data=str(data_config),
        imgsz=224,
        split="val"
    )

    print()
    print("===== Baseline Model Evaluation =====")
    print(f"mAP@50:    {results.box.map50:.4f}")
    print(f"mAP@50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")
    print()
    print("Evaluation completed.")


if __name__ == "__main__":
    evaluate_model()