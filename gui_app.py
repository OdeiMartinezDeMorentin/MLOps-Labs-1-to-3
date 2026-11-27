"""Hosting of the GUI (Gradio) on Modal.

The GUI is a service of its own, separated from the backend (modal_app.py): Modal
builds a small image with the dependencies of gui/requirements.txt and the gui/
folder, and serves the Gradio app at a public HTTPS URL:
    https://<modal-workspace>--image-classifier-gui-ui.modal.run
(`modal deploy` prints it).
"""

import os
import sys

import modal

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_requirements("gui/requirements.txt")
    .add_local_dir("gui", remote_path="/root/gui")
)

app = modal.App("image-classifier-gui")

backend = modal.Secret.from_dict({"API_URL": os.environ.get("API_URL", "")})


# - max_containers=1: Gradio keeps session state in memory.
# - concurrent: that container answers several users at the same time.
@app.function(image=image, secrets=[backend], max_containers=1, scaledown_window=120)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    """Return the Gradio app mounted on a FastAPI app; Modal serves it."""
    sys.path.insert(0, "/root")
    # pylint: disable=import-outside-toplevel
    import gradio as gr
    from fastapi import FastAPI

    from gui.app import demo

    return gr.mount_gradio_app(FastAPI(), demo, path="/")
