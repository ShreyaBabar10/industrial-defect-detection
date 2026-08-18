from pathlib import Path
import time

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

    # Accuracy evaluation
    results = model.val(
        data=str(data_config),
        imgsz=image_size,
        split="val",
        verbose=False
    )

    # Accuracy metrics
    metrics = {
        "mAP@50": results.box.map50,
        "mAP@50-95": results.box.map,
        "Precision": results.box.mp,
        "Recall": results.box.mr,
    }

    # Warm-up inference
    sample_image = (
        project_dir
        / "Data"
        / "processed"
        / "NEU-DET-YOLO"
        / "images"
        / "val"
    )

    image_files = sorted(sample_image.glob("*.jpg"))

    if not image_files:
        print("No validation images found for latency measurement.")
        return None

    sample = str(image_files[0])

    model.predict(
        source=sample,
        imgsz=image_size,
        conf=0.25,
        verbose=False
    )

    # Measure inference latency
    latency_runs = 10
    total_time = 0.0

    for _ in range(latency_runs):

        start_time = time.perf_counter()

        model.predict(
            source=sample,
            imgsz=image_size,
            conf=0.25,
            verbose=False
        )

        total_time += (
            time.perf_counter() - start_time
        )

    average_latency_ms = (
        total_time / latency_runs
    ) * 1000

    metrics["Latency (ms)"] = average_latency_ms

    metrics["Approx FPS"] = (
        1000 / average_latency_ms
    )

    print()
    print(f"===== {model_name} =====")
    print(f"mAP@50:       {metrics['mAP@50']:.4f}")
    print(f"mAP@50-95:    {metrics['mAP@50-95']:.4f}")
    print(f"Precision:    {metrics['Precision']:.4f}")
    print(f"Recall:       {metrics['Recall']:.4f}")
    print(
        f"Latency:      "
        f"{metrics['Latency (ms)']:.2f} ms"
    )
    print(
        f"Approx FPS:   "
        f"{metrics['Approx FPS']:.2f}"
    )

    return metrics


def save_comparison(results):

    output_dir = project_dir / "results"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "day5_resolution_comparison.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "YOLO Resolution Accuracy-Speed Comparison\n"
        )

        file.write(
            "==========================================\n\n"
        )

        for model_name, metrics in results.items():

            file.write(
                f"{model_name}\n"
            )

            file.write(
                "-" * len(model_name)
                + "\n"
            )

            file.write(
                f"mAP@50:       "
                f"{metrics['mAP@50']:.4f}\n"
            )

            file.write(
                f"mAP@50-95:    "
                f"{metrics['mAP@50-95']:.4f}\n"
            )

            file.write(
                f"Precision:    "
                f"{metrics['Precision']:.4f}\n"
            )

            file.write(
                f"Recall:       "
                f"{metrics['Recall']:.4f}\n"
            )

            file.write(
                f"Latency (ms): "
                f"{metrics['Latency (ms)']:.2f}\n"
            )

            file.write(
                f"Approx FPS:   "
                f"{metrics['Approx FPS']:.2f}\n\n"
            )

    print()
    print(
        f"Comparison saved to: {output_file}"
    )


def main():

    results = {}

    baseline_results = evaluate_model(
        "Baseline 224px",
        models["baseline_224"],
        224
    )

    if baseline_results:
        results["Baseline 224px"] = (
            baseline_results
        )

    high_resolution_results = evaluate_model(
        "Higher Resolution 256px",
        models["higher_resolution_256"],
        256
    )

    if high_resolution_results:
        results["Higher Resolution 256px"] = (
            high_resolution_results
        )

    if results:
        save_comparison(results)


if __name__ == "__main__":
    main()