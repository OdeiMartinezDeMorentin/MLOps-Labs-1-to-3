"""Image processing and classification logic."""

import io
from pathlib import Path
import random
from typing import Any, Union
from PIL import Image

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_image(source: Union[str, Path, bytes, io.BytesIO]) -> Image.Image:
    """Load an image from a file path, raw bytes, or BytesIO buffer."""
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, io.BytesIO):
        source.seek(0)
        img = Image.open(source)
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")
    return img.convert("RGB")


def image_to_bytes(image: Image.Image, image_format: str = "JPEG") -> bytes:
    """Convert a PIL Image to bytes in memory."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image instance.")
    buffer = io.BytesIO()
    # JPEG does not support RGBA or P modes; convert to RGB if necessary
    if image_format.upper() in ("JPEG", "JPG") and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buffer, format=image_format)
    buffer.seek(0)
    return buffer.getvalue()


def predict_image(image: Image.Image) -> dict[str, Any]:
    """Predict the class of a given image randomly from predefined classes.

    In this lab stage, class prediction is simulated.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image instance.")
    predicted_class = random.choice(CLASSES)
    confidence = round(random.uniform(0.70, 0.99), 2)
    return {"class": predicted_class, "confidence": confidence}


def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize an image to the given width and height.

    Raises:
        ValueError: If width or height are not strictly positive integers.
        TypeError: If input is not a PIL Image or dimensions are not integers.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image instance.")
    if not isinstance(width, int) or not isinstance(height, int):
        raise TypeError("Width and height must be integers.")
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Width and height must be positive integers (got width={width}, height={height})."
        )
    return image.resize((width, height))


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """Convert an image to grayscale."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image instance.")
    return image.convert("L")


def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """Rotate an image by the given angle in degrees."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image instance.")
    if not isinstance(angle, (int, float)):
        raise TypeError("Angle must be a number.")
    return image.rotate(float(angle), expand=True)

