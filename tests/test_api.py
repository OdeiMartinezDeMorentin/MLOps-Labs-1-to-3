"""Unit and integration tests for the FastAPI application."""

# pylint: disable=redefined-outer-name

import io
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.api import app
from mylib.image_processing import CLASSES, image_to_bytes


@pytest.fixture
def client() -> TestClient:
    """Provide a FastAPI TestClient instance."""
    return TestClient(app)


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Create raw JPEG bytes for testing file uploads."""
    img = Image.new("RGB", (100, 80), color="red")
    return image_to_bytes(img, image_format="JPEG")


def test_home_endpoint(client: TestClient):
    """Verify GET / returns HTTP 200 with HTML content."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Welcome to the Image Processing" in response.text
    assert "/docs" in response.text


def test_health_endpoint(client: TestClient):
    """Verify GET /health returns HTTP 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classes_endpoint(client: TestClient):
    """Verify GET /classes returns list of classes."""
    response = client.get("/classes")
    assert response.status_code == 200
    assert response.json() == CLASSES


def test_predict_endpoint_success(client: TestClient, sample_jpeg_bytes: bytes):
    """Verify POST /predict successfully classifies an image."""
    files = {"file": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample.jpg"
    assert data["class"] in CLASSES
    assert 0.70 <= data["confidence"] <= 0.99


def test_predict_endpoint_corrupted_file(client: TestClient):
    """Verify POST /predict returns HTTP 400 for corrupted image upload."""
    files = {"file": ("corrupt.jpg", b"not-a-valid-image", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Could not process image" in response.json()["detail"]


def test_predict_endpoint_missing_file(client: TestClient):
    """Verify POST /predict returns HTTP 422 when file is omitted."""
    response = client.post("/predict")
    assert response.status_code == 422


def test_resize_endpoint_success(client: TestClient, sample_jpeg_bytes: bytes):
    """Verify POST /resize returns resized image binary."""
    files = {"file": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")}
    data = {"width": "50", "height": "30"}
    response = client.post("/resize", files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-width"] == "50"
    assert response.headers["x-height"] == "30"
    resized = Image.open(io.BytesIO(response.content))
    assert resized.size == (50, 30)


@pytest.mark.parametrize(
    "width,height",
    [
        ("0", "50"),
        ("50", "0"),
        ("-10", "50"),
        ("50", "-10"),
    ],
)
def test_resize_endpoint_invalid_dimensions(
    client: TestClient, sample_jpeg_bytes: bytes, width: str, height: str
):
    """Verify POST /resize returns HTTP 400 for non-positive dimensions."""
    files = {"file": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")}
    data = {"width": width, "height": height}
    response = client.post("/resize", files=files, data=data)
    assert response.status_code == 400
    assert "positive integers" in response.json()["detail"]


def test_resize_endpoint_corrupted_file(client: TestClient):
    """Verify POST /resize returns HTTP 400 for corrupted image upload."""
    files = {"file": ("sample.jpg", b"bad-data", "image/jpeg")}
    data = {"width": "50", "height": "50"}
    response = client.post("/resize", files=files, data=data)
    assert response.status_code == 400
    assert "Could not resize image" in response.json()["detail"]


def test_resize_endpoint_invalid_types(client: TestClient, sample_jpeg_bytes: bytes):
    """Verify POST /resize returns HTTP 422 when dimensions are not integers."""
    files = {"file": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")}
    data = {"width": "invalid", "height": "50"}
    response = client.post("/resize", files=files, data=data)
    assert response.status_code == 422


def test_grayscale_endpoint_success(client: TestClient, sample_jpeg_bytes: bytes):
    """Verify POST /grayscale converts image to grayscale."""
    files = {"file": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")}
    response = client.post("/grayscale", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    gray = Image.open(io.BytesIO(response.content))
    assert gray.size == (100, 80)


def test_grayscale_endpoint_corrupted_file(client: TestClient):
    """Verify POST /grayscale returns HTTP 400 for corrupted image."""
    files = {"file": ("sample.jpg", b"corrupted", "image/jpeg")}
    response = client.post("/grayscale", files=files)
    assert response.status_code == 400
    assert "Could not convert image to grayscale" in response.json()["detail"]


def test_rotate_endpoint_success(client: TestClient, sample_jpeg_bytes: bytes):
    """Verify POST /rotate rotates image and returns binary."""
    files = {"file": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")}
    data = {"angle": "90"}
    response = client.post("/rotate", files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    rotated = Image.open(io.BytesIO(response.content))
    assert rotated.size == (80, 100)


def test_rotate_endpoint_corrupted_file(client: TestClient):
    """Verify POST /rotate returns HTTP 400 for corrupted image."""
    files = {"file": ("sample.jpg", b"corrupted", "image/jpeg")}
    data = {"angle": "45"}
    response = client.post("/rotate", files=files, data=data)
    assert response.status_code == 400
    assert "Could not rotate image" in response.json()["detail"]


def test_rotate_endpoint_invalid_angle(client: TestClient, sample_jpeg_bytes: bytes):
    """Verify POST /rotate returns HTTP 422 for non-float angle."""
    files = {"file": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")}
    data = {"angle": "not-a-number"}
    response = client.post("/rotate", files=files, data=data)
    assert response.status_code == 422

