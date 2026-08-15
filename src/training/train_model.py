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

    print("Starting improved YOLO training...")
    print(f"Dataset configuration: {data_config}")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_config),
        epochs=40,
        imgsz=224,
        batch=16,
        project=str(model_output),
        name="neu_defect_detector_improved",
        pretrained=True,
        patience=10,
        workers=2,
        degrees=5,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=1.0,
        verbose=True
    )

    print()
    print("Improved training completed.")
    print(
        f"Results saved in: "
        f"{model_output / 'neu_defect_detector_improved'}"
    )


if __name__ == "__main__":
    train_model()