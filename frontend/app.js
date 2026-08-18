const API_BASE_URL = "https://industrial-defect-detection-vsd2.onrender.com";

const imageInput = document.getElementById("imageInput");
const detectButton = document.getElementById("detectButton");

const confidence = document.getElementById("confidence");
const confidenceValue = document.getElementById("confidenceValue");

const inputPreview = document.getElementById("inputPreview");
const resultImage = document.getElementById("resultImage");

const previewSection = document.getElementById("previewSection");

const detectionStatus = document.getElementById("detectionStatus");
const detectionCount = document.getElementById("detectionCount");
const inferenceTime = document.getElementById("inferenceTime");
const modelName = document.getElementById("modelName");

const detectionsContainer =
    document.getElementById("detectionsContainer");

const healthStatus =
    document.getElementById("healthStatus");

const errorSection =
    document.getElementById("errorSection");

const errorMessage =
    document.getElementById("errorMessage");

let selectedFile = null;


confidence.addEventListener("input", () => {
    confidenceValue.textContent =
        Number(confidence.value).toFixed(2);
});


imageInput.addEventListener("change", () => {

    selectedFile = imageInput.files[0];

    if (!selectedFile) {
        detectButton.disabled = true;
        return;
    }

    const imageURL =
        URL.createObjectURL(selectedFile);

    inputPreview.src = imageURL;

    previewSection.classList.remove("hidden");

    detectButton.disabled = false;

    hideError();
});


detectButton.addEventListener(
    "click",
    detectDefects
);


async function checkHealth() {

    try {

        const response =
            await fetch(`${API_BASE_URL}/health`);

        if (!response.ok) {
            throw new Error("API is not healthy.");
        }

        const data =
            await response.json();

        healthStatus.textContent =
            `API Online • ${data.model}`;

    } catch (error) {

        healthStatus.textContent =
            "API Offline";
    }
}


async function detectDefects() {

    if (!selectedFile) {
        return;
    }

    detectButton.disabled = true;
    detectButton.textContent =
        "Running inference...";

    hideError();

    const formData =
        new FormData();

    formData.append(
        "file",
        selectedFile
    );

    const threshold =
        confidence.value;

    try {

        const response =
            await fetch(
                `${API_BASE_URL}/predict?confidence=${threshold}`,
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Prediction request failed."
            );
        }

        displayResults(data);

    } catch (error) {

        showError(
            error.message
        );

    } finally {

        detectButton.disabled = false;

        detectButton.textContent =
            "Detect Defects";
    }
}


function displayResults(data) {

    /*
     * Backend now returns an absolute prediction URL:
     * http://127.0.0.1:8000/prediction/filename.jpg
     *
     * Older responses may contain only:
     * /prediction/filename.jpg
     *
     * This handles both cases.
     */
    const imageURL =
        data.prediction_url.startsWith("http")
            ? data.prediction_url
            : `${API_BASE_URL}${data.prediction_url}`;

    resultImage.src = imageURL;

    resultImage.alt =
        "Detection result";

    resultImage.onerror = () => {

        showError(
            "Prediction image could not be loaded."
        );
    };

    detectionCount.textContent =
        data.detection_count;

    inferenceTime.textContent =
        `${Number(data.inference_time_ms).toFixed(2)} ms`;

    modelName.textContent =
        data.model;

    if (data.detection_count > 0) {

        detectionStatus.textContent =
            "DEFECT DETECTED";

    } else {

        detectionStatus.textContent =
            "NO DEFECT";
    }

    renderDetections(
        data.detections
    );

    previewSection.classList.remove(
        "hidden"
    );
}


function renderDetections(detections) {

    detectionsContainer.innerHTML = "";

    if (
        !detections ||
        detections.length === 0
    ) {

        detectionsContainer.textContent =
            "No defects detected.";

        return;
    }

    detections.forEach(
        (detection, index) => {

            const row =
                document.createElement("div");

            row.className =
                "detection-row";

            const box =
                detection.bounding_box;

            row.innerHTML = `
                <div class="detection-title">
                    ${index + 1}. ${detection.class_name}
                </div>

                <div class="detection-info">
                    Confidence:
                    ${(detection.confidence * 100).toFixed(2)}%
                    <br>

                    Bounding Box:
                    (${box.x1}, ${box.y1})
                    →
                    (${box.x2}, ${box.y2})
                </div>
            `;

            detectionsContainer.appendChild(
                row
            );
        }
    );
}


function showError(message) {

    errorMessage.textContent =
        message;

    errorSection.classList.remove(
        "hidden"
    );
}


function hideError() {

    errorSection.classList.add(
        "hidden"
    );

    errorMessage.textContent =
        "";
}


checkHealth();