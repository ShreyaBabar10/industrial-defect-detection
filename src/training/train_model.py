from pathlib import Path
from ultralytics import YOLO


# Project root
project_dir = Path(__file__).resolve().parents[2]

# Dataset configuration
data_config = project_dir / "configs" / "data.yaml"

# Model output directory
model_output = project_dir / "models"


def train_model():

    model_output.mkdir(parents=True, exist_ok=True)

    print("Starting YOLO training...")
    print(f"Dataset configuration: {data_config}")

    # Start with a lightweight pretrained YOLO model.
    # This keeps the first training run practical on a local machine.
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data_config),
        epochs=30,
        imgsz=200,
        batch=16,
        project=str(model_output),
        name="neu_defect_detector",
        pretrained=True,
        patience=8,
        workers=2,
        verbose=True
    )

    print()
    print("Training completed.")
    print(f"Results saved in: {model_output / 'neu_defect_detector'}")


if __name__ == "__main__":
    train_model()