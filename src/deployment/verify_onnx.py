from pathlib import Path
import onnxruntime as ort


project_dir = Path(__file__).resolve().parents[2]

model_path = (
    project_dir
    / "models"
    / "neu_defect_detector_256"
    / "weights"
    / "best.onnx"
)


def verify_model():

    if not model_path.exists():
        print("ONNX model not found.")
        print(f"Expected model: {model_path}")
        return

    print("Loading ONNX model...")
    print(f"Model: {model_path}")

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"]
    )

    print()
    print("ONNX model loaded successfully.")

    for input_info in session.get_inputs():
        print(f"Input: {input_info.name}")
        print(f"Shape: {input_info.shape}")
        print(f"Type: {input_info.type}")

    print()
    print("ONNX verification completed.")


if __name__ == "__main__":
    verify_model()