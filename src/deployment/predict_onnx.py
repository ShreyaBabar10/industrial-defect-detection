from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


# Project root
project_dir = Path(__file__).resolve().parents[2]

# ONNX model
model_path = (
    project_dir
    / "models"
    / "neu_defect_detector_256"
    / "weights"
    / "best.onnx"
)

# Test image
image_path = (
    project_dir
    / "Data"
    / "processed"
    / "NEU-DET-YOLO"
    / "images"
    / "val"
    / "crazing_241.jpg"
)

# Prediction output
output_dir = project_dir / "results" / "onnx_predictions"


def predict_image():

    if not model_path.exists():
        print("ONNX model not found.")
        print(f"Expected model: {model_path}")
        return

    if not image_path.exists():
        print("Test image not found.")
        print(f"Expected image: {image_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ONNX model...")
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"]
    )

    input_info = session.get_inputs()[0]
    input_name = input_info.name

    image = cv2.imread(str(image_path))

    if image is None:
        print("Unable to read test image.")
        return

    original_height, original_width = image.shape[:2]

    resized = cv2.resize(image, (256, 256))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    input_image = resized.astype(np.float32) / 255.0
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = np.expand_dims(input_image, axis=0)

    print("Running ONNX inference...")

    output = session.run(
        None,
        {input_name: input_image}
    )

    print()
    print(f"Input image: {image_path.name}")
    print(f"Original size: {original_width} x {original_height}")
    print(f"Model input: {input_image.shape}")
    print(f"Output shape: {output[0].shape}")

    output_path = output_dir / "crazing_241_prediction.npy"
    np.save(output_path, output[0])

    print()
    print("ONNX inference completed.")
    print(f"Raw prediction saved to: {output_path}")


if __name__ == "__main__":
    predict_image()