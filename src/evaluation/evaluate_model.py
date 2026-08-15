from pathlib import Path
from ultralytics import YOLO


project_dir = Path(__file__).resolve().parents[2]

data_config = project_dir / "configs" / "data.yaml"
model_path = project_dir / "models" / "best.pt"


def evaluate_model():

    if not model_path.exists():
        print("Trained model not found.")
        print("YOLO model training is required before evaluation.")
        return

    print("Loading trained model...")
    model = YOLO(str(model_path))

    print("Running validation...")

    results = model.val(
        data=str(data_config),
        imgsz=200,
        split="val"
    )

    print()
    print("===== Model Evaluation =====")
    print(f"mAP@50:    {results.box.map50:.4f}")
    print(f"mAP@50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")


if __name__ == "__main__":
    evaluate_model()