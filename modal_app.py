"""Hosting of the backend on Modal.

Modal runs the Docker image of the API published in Docker Hub by the CI/CD pipeline
and serves the FastAPI app as a web endpoint (ASGI) with a public HTTPS URL:
    https://<modal-workspace>--image-classifier-api-serve.modal.run
(`modal deploy` prints it). The URL does not change between deployments, so it is
stored as the BACKEND_API_URL secret of the GitHub repository and used by the GUI.
"""

import os
import sys

import modal

# Image of the API in Docker Hub. The CI/CD pipeline sets DOCKER_IMAGE with the tag of
# the commit (<user>/image-classifier-api:<commit SHA>).
DOCKER_IMAGE = os.environ.get(
    "DOCKER_IMAGE", "docker.io/odeimardemor/image-classifier-api:latest"
)

image = modal.Image.from_registry(DOCKER_IMAGE)

app = modal.App("image-classifier-api")


# - No GPU in Lab 2 (random model does not need it). In Lab 3 we will add gpu="T4".
# - scaledown_window: seconds without requests before stopping the container.
@app.function(image=image, scaledown_window=120)
@modal.asgi_app()
def serve():
    """Return the FastAPI app included in the image; Modal serves it at a public URL."""
    sys.path.insert(0, "/app")
    # pylint: disable=import-outside-toplevel
    from api.api import app as fastapi_app

    return fastapi_app
