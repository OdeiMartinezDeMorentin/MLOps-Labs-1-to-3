"""Gradio GUI of the image classifier (frontend).

It is a microservice of its own: it does not import the logic of the project,
it only calls the /predict endpoint of the REST API (backend) through HTTP.
"""

import os
from pathlib import Path

import gradio as gr
import requests

# URL of the FastAPI REST API (environment variable of the GUI service)
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

# Generous timeout to allow Modal container cold starts
TIMEOUT = float(os.environ.get("API_TIMEOUT", "60"))


def predict(image_path: str | None) -> str:
    """Send an image file to the API /predict endpoint and return the class label."""
    if not image_path:
        return "Error: please upload an image"

    path = Path(image_path)
    if not path.is_file():
        return f"Error: image file not found ({image_path})"

    try:
        with path.open("rb") as img_file:
            files = {"file": (path.name, img_file, "image/jpeg")}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=TIMEOUT)
    except requests.exceptions.Timeout:
        return "Error: the API did not answer in time (it may be starting, try again)"
    except requests.exceptions.RequestException as exc:
        return f"Error: could not connect to the API ({exc})"

    if response.ok:
        data = response.json()
        predicted_class = data["class"]
        confidence = data.get("confidence")
        if confidence is not None:
            return f"{predicted_class} (confidence: {confidence:.2f})"
        return str(predicted_class)

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return f"Error {response.status_code}: {detail}"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=gr.Textbox(label="Predicted Class"),
    title="Image Classifier created with FastAPI and Gradio",
    description=(
        "Interactive image classifier using the /predict endpoint of the API. "
        "The first request may take a while if the API is starting."
    ),
)

if __name__ == "__main__":  # pragma: no cover
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
