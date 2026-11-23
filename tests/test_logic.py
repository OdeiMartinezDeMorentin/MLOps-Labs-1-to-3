"""Unit tests for the image processing and prediction logic module."""

# pylint: disable=redefined-outer-name

import io
from PIL import Image
import pytest

from mylib.image_processing import (
    CLASSES,
    convert_to_grayscale,
    image_to_bytes,
    load_image,
    predict_image,
    resize_image,
    rotate_image,
)


@pytest.fixture
def sample_image() -> Image.Image:
    """Provide a standard RGB sample image for tests."""
    return Image.new("RGB", (100, 80), color="blue")


def test_classes_list():
    """Verify predefined classes list is valid and contains standard labels."""
    assert len(CLASSES) > 0
    assert "cat" in CLASSES
    assert "dog" in CLASSES


def test_predict_image(sample_image):
    """Test image prediction returns expected structure and valid class."""
    result = predict_image(sample_image)
    assert isinstance(result, dict)
    assert "class" in result
    assert "confidence" in result
    assert result["class"] in CLASSES
    assert 0.70 <= result["confidence"] <= 0.99


def test_predict_image_invalid_type():
    """Verify predict_image raises TypeError when input is not a PIL Image."""
    with pytest.raises(TypeError, match="Expected a PIL Image instance"):
        predict_image("not an image")


def test_resize_image_valid(sample_image):
    """Verify resize_image correctly resizes to target dimensions."""
    resized = resize_image(sample_image, 50, 40)
    assert resized.size == (50, 40)


@pytest.mark.parametrize(
    "width,height",
    [
        (0, 50),
        (50, 0),
        (-10, 50),
        (50, -10),
        (-20, -20),
    ],
)
def test_resize_image_invalid_dimensions(sample_image, width, height):
    """Verify resize_image raises ValueError for non-positive dimensions."""
    with pytest.raises(ValueError, match="must be positive integers"):
        resize_image(sample_image, width, height)


def test_resize_image_invalid_types(sample_image):
    """Verify resize_image raises TypeError for non-integer dimensions or invalid image."""
    with pytest.raises(TypeError, match="Width and height must be integers"):
        resize_image(sample_image, "50", 40)

    with pytest.raises(TypeError, match="Expected a PIL Image instance"):
        resize_image("invalid", 50, 40)


def test_convert_to_grayscale(sample_image):
    """Verify grayscale conversion transforms image mode to 'L'."""
    gray = convert_to_grayscale(sample_image)
    assert gray.mode == "L"
    assert gray.size == sample_image.size


def test_convert_to_grayscale_invalid_type():
    """Verify convert_to_grayscale raises TypeError on invalid input."""
    with pytest.raises(TypeError, match="Expected a PIL Image instance"):
        convert_to_grayscale(123)


def test_rotate_image(sample_image):
    """Verify image rotation produces expected rotated image."""
    rotated = rotate_image(sample_image, 90)
    assert rotated.size == (80, 100)


def test_rotate_image_invalid_types(sample_image):
    """Verify rotate_image raises TypeError on invalid types."""
    with pytest.raises(TypeError, match="Expected a PIL Image instance"):
        rotate_image("not_an_image", 45)

    with pytest.raises(TypeError, match="Angle must be a number"):
        rotate_image(sample_image, "invalid_angle")


def test_load_image_from_bytes(sample_image):
    """Verify load_image loads image correctly from raw bytes."""
    raw_bytes = image_to_bytes(sample_image, "PNG")
    loaded = load_image(raw_bytes)
    assert loaded.size == sample_image.size
    assert loaded.mode == "RGB"


def test_load_image_from_bytesio(sample_image):
    """Verify load_image loads image correctly from BytesIO buffer."""
    buf = io.BytesIO(image_to_bytes(sample_image, "PNG"))
    loaded = load_image(buf)
    assert loaded.size == sample_image.size
    assert loaded.mode == "RGB"


def test_load_image_from_path(tmp_path, sample_image):
    """Verify load_image loads image correctly from a file path."""
    file_path = tmp_path / "test.jpg"
    sample_image.save(file_path, format="JPEG")
    loaded = load_image(str(file_path))
    assert loaded.size == sample_image.size
    assert loaded.mode == "RGB"


def test_load_image_invalid_type():
    """Verify load_image raises TypeError for unsupported sources."""
    with pytest.raises(TypeError, match="Unsupported source type"):
        load_image(12345)


def test_image_to_bytes(sample_image):
    """Verify image_to_bytes outputs non-empty bytes."""
    data = image_to_bytes(sample_image, "JPEG")
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_image_to_bytes_rgba_to_jpeg():
    """Verify image_to_bytes converts RGBA to RGB when saving as JPEG."""
    rgba_image = Image.new("RGBA", (20, 20), color=(255, 0, 0, 128))
    data = image_to_bytes(rgba_image, "JPEG")
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_image_to_bytes_invalid_type():
    """Verify image_to_bytes raises TypeError when not given a PIL Image."""
    with pytest.raises(TypeError, match="Expected a PIL Image instance"):
        image_to_bytes("not an image")
