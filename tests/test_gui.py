"""Unit tests for the Gradio GUI (gui/app.py)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from gui import app as gui


def fake_response(status_code: int, payload: dict | None, text: str = "") -> MagicMock:
    """Build a fake requests.Response."""
    response = MagicMock()
    response.ok = status_code < 400
    response.status_code = status_code
    response.json.return_value = payload
    response.text = text
    return response


def test_gui_returns_predicted_class(tmp_path: Path):
    """Verify GUI uploads image to /predict and displays predicted class."""
    img_path = tmp_path / "sample.jpg"
    img_path.write_bytes(b"fake-image-bytes")

    payload = {"filename": "sample.jpg", "class": "cat", "confidence": 0.91}
    with patch.object(
        gui.requests, "post", return_value=fake_response(200, payload)
    ) as post:
        result = gui.predict(str(img_path))

    assert result == "cat (confidence: 0.91)"
    assert post.call_args.args[0].endswith("/predict")
    assert "file" in post.call_args.kwargs["files"]


def test_gui_returns_class_without_confidence(tmp_path: Path):
    """Verify GUI handles responses containing only the class field."""
    img_path = tmp_path / "sample.jpg"
    img_path.write_bytes(b"fake-image-bytes")

    with patch.object(
        gui.requests, "post", return_value=fake_response(200, {"class": "dog"})
    ):
        assert gui.predict(str(img_path)) == "dog"


def test_gui_shows_api_error(tmp_path: Path):
    """Verify API JSON error details are displayed to the user."""
    img_path = tmp_path / "sample.jpg"
    img_path.write_bytes(b"fake-image-bytes")

    error = fake_response(400, {"detail": "Could not process image"})
    with patch.object(gui.requests, "post", return_value=error):
        assert gui.predict(str(img_path)) == "Error 400: Could not process image"


def test_gui_shows_error_without_json(tmp_path: Path):
    """Verify non-JSON API errors fall back to response text."""
    img_path = tmp_path / "sample.jpg"
    img_path.write_bytes(b"fake-image-bytes")

    error = fake_response(500, None, text="Internal Server Error")
    error.json.side_effect = ValueError("no JSON")
    with patch.object(gui.requests, "post", return_value=error):
        assert gui.predict(str(img_path)) == "Error 500: Internal Server Error"


def test_gui_handles_timeout(tmp_path: Path):
    """Verify timeout (e.g., cold start) returns a readable message."""
    img_path = tmp_path / "sample.jpg"
    img_path.write_bytes(b"fake-image-bytes")

    with patch.object(gui.requests, "post", side_effect=requests.exceptions.Timeout()):
        assert "did not answer in time" in gui.predict(str(img_path))


def test_gui_handles_connection_error(tmp_path: Path):
    """Verify connection errors return a readable message."""
    img_path = tmp_path / "sample.jpg"
    img_path.write_bytes(b"fake-image-bytes")

    with patch.object(
        gui.requests, "post", side_effect=requests.exceptions.ConnectionError("down")
    ):
        assert "could not connect" in gui.predict(str(img_path))


def test_gui_requires_image():
    """Verify API is not called if no image is provided."""
    with patch.object(gui.requests, "post") as post:
        assert gui.predict(None) == "Error: please upload an image"
    post.assert_not_called()


def test_gui_missing_file_path():
    """Verify API is not called if the image path does not exist."""
    with patch.object(gui.requests, "post") as post:
        assert "image file not found" in gui.predict("/nonexistent/image.jpg")
    post.assert_not_called()
