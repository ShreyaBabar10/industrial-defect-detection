from pathlib import Path
from ultralytics import YOLO


project_dir = Path(__file__).resolve().parents[2]

model_path = (
    project_dir
    / "models"
    / "neu_defect_detector_256"
    / "weights"
    / "best.pt"
)


def export_model():

    if not model_path.exists():
        print("Trained model not found.")
        print(f"Expected model: {model_path}")
        return

    print("Loading trained YOLO model...")
    print(f"Model: {model_path}")

    model = YOLO(str(model_path))

    print()
    print("Exporting model to ONNX...")

    exported_path = model.export(
        format="onnx",
        imgsz=256,
        opset=12,
        simplify=True
    )

    print()
    print("ONNX export completed.")
    print(f"Exported model: {exported_path}")


if __name__ == "__main__":
    export_model()