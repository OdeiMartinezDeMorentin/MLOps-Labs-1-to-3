"""FastAPI application for image processing and classification microservice."""

from pathlib import Path
from typing import Any, List
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from mylib.image_processing import (
    CLASSES,
    convert_to_grayscale,
    image_to_bytes,
    load_image,
    predict_image,
    resize_image,
    rotate_image,
)

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(
    title="Image Processing & Classification API",
    description="REST API to classify and process images using Pillow.",
    version="0.1.0",
)


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """Home endpoint rendering the HTML welcome page."""
    return templates.TemplateResponse(request=request, name="home.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for smoke testing and service monitoring."""
    return {"status": "ok"}


@app.get("/classes")
def list_classes() -> List[str]:
    """Return all available image prediction classes."""
    return CLASSES


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:
    """Predict the class of an uploaded image."""
    try:
        contents = await file.read()
        image = load_image(contents)
        result = predict_image(image)
        return {
            "filename": file.filename,
            "class": result["class"],
            "confidence": result["confidence"],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not process image: {exc}",
        ) from exc


@app.post("/resize")
async def resize_endpoint(
    file: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...),
) -> Response:
    """Resize an uploaded image to the requested width and height."""
    if width <= 0 or height <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Width and height must be positive integers (got width={width}, height={height})."
            ),
        )
    try:
        contents = await file.read()
        image = load_image(contents)
        resized_image = resize_image(image, width, height)
        img_bytes = image_to_bytes(resized_image, image_format="JPEG")
        return Response(
            content=img_bytes,
            media_type="image/jpeg",
            headers={"X-Width": str(width), "X-Height": str(height)},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not resize image: {exc}",
        ) from exc


@app.post("/grayscale")
async def grayscale_endpoint(file: UploadFile = File(...)) -> Response:
    """Convert an uploaded image to grayscale."""
    try:
        contents = await file.read()
        image = load_image(contents)
        gray_image = convert_to_grayscale(image)
        img_bytes = image_to_bytes(gray_image, image_format="JPEG")
        return Response(content=img_bytes, media_type="image/jpeg")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not convert image to grayscale: {exc}",
        ) from exc


@app.post("/rotate")
async def rotate_endpoint(
    file: UploadFile = File(...),
    angle: float = Form(...),
) -> Response:
    """Rotate an uploaded image by angle in degrees."""
    try:
        contents = await file.read()
        image = load_image(contents)
        rotated_image = rotate_image(image, angle)
        img_bytes = image_to_bytes(rotated_image, image_format="JPEG")
        return Response(content=img_bytes, media_type="image/jpeg")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not rotate image: {exc}",
        ) from exc


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)

