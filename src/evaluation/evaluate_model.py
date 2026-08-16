from pathlib import Path
from ultralytics import YOLO


# Project root
project_dir = Path(__file__).resolve().parents[2]

# Dataset configuration
data_config = project_dir / "configs" / "data.yaml"

# Models to compare
models = {
    "baseline_224": (
        project_dir
        / "models"
        / "neu_defect_detector"
        / "weights"
        / "best.pt"
    ),
    "higher_resolution_256": (
        project_dir
        / "models"
        / "neu_defect_detector_256"
        / "weights"
        / "best.pt"
    ),
}


def evaluate_model(model_name, model_path, image_size):
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None

    print()
    print(f"Evaluating {model_name}...")
    print(f"Model: {model_path}")
    print(f"Image size: {image_size}")

    model = YOLO(str(model_path))

    results = model.val(
        data=str(data_config),
        imgsz=image_size,
        split="val"
    )

    metrics = {
        "mAP@50": results.box.map50,
        "mAP@50-95": results.box.map,
        "Precision": results.box.mp,
        "Recall": results.box.mr,
    }

    print()
    print(f"===== {model_name} =====")
    print(f"mAP@50:    {metrics['mAP@50']:.4f}")
    print(f"mAP@50-95: {metrics['mAP@50-95']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")

    return metrics


def save_comparison(results):
    output_dir = project_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "day5_resolution_comparison.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("YOLO Resolution Comparison\n")
        file.write("==========================\n\n")

        for model_name, metrics in results.items():
            file.write(f"{model_name}\n")
            file.write("-" * len(model_name) + "\n")
            file.write(f"mAP@50:    {metrics['mAP@50']:.4f}\n")
            file.write(f"mAP@50-95: {metrics['mAP@50-95']:.4f}\n")
            file.write(f"Precision: {metrics['Precision']:.4f}\n")
            file.write(f"Recall:    {metrics['Recall']:.4f}\n\n")

    print()
    print(f"Comparison saved to: {output_file}")


def main():
    results = {}

    baseline_results = evaluate_model(
        "Baseline 224px",
        models["baseline_224"],
        224
    )

    if baseline_results:
        results["Baseline 224px"] = baseline_results

    high_resolution_results = evaluate_model(
        "Higher Resolution 256px",
        models["higher_resolution_256"],
        256
    )

    if high_resolution_results:
        results["Higher Resolution 256px"] = high_resolution_results

    if results:
        save_comparison(results)


if __name__ == "__main__":
    main()