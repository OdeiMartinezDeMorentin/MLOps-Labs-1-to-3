"""Command Line Interface (CLI) for image processing and prediction."""

import json
from pathlib import Path
from typing import Optional
import click

from mylib.image_processing import (
    CLASSES,
    convert_to_grayscale,
    load_image,
    predict_image,
    resize_image,
    rotate_image,
)


@click.group()
@click.version_option(version="0.1.0", prog_name="mlops-image-cli")
def cli() -> None:
    """MLOps Command Line Interface for image processing and classification."""


@cli.command("classes")
def list_classes_cmd() -> None:
    """List all available prediction classes."""
    click.echo(json.dumps(CLASSES))


@cli.command("predict")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
def predict_cmd(image_path: str) -> None:
    """Predict the class of an image given its file path."""
    try:
        img = load_image(image_path)
        result = predict_image(img)
        click.echo(f"Class: {result['class']} (confidence: {result['confidence']:.2f})")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        click.echo(f"Error: {exc}", err=True)


@cli.command("resize")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("width", type=int)
@click.argument("height", type=int)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to save the resized image. Defaults to resized_<original_name>.",
)
def resize_cmd(image_path: str, width: int, height: int, output: Optional[str]) -> None:
    """Resize an image to specified WIDTH and HEIGHT."""
    try:
        img = load_image(image_path)
        resized_img = resize_image(img, width, height)
        out_path = (
            Path(output)
            if output
            else Path(image_path).with_name(f"resized_{Path(image_path).name}")
        )
        resized_img.save(out_path)
        click.echo(f"Image resized to {width}x{height} and saved to {out_path}")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        click.echo(f"Error: {exc}", err=True)


@cli.command("grayscale")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to save the grayscale image.",
)
def grayscale_cmd(image_path: str, output: Optional[str]) -> None:
    """Convert an image to grayscale."""
    try:
        img = load_image(image_path)
        gray_img = convert_to_grayscale(img)
        out_path = (
            Path(output)
            if output
            else Path(image_path).with_name(f"grayscale_{Path(image_path).name}")
        )
        gray_img.save(out_path)
        click.echo(f"Grayscale image saved to {out_path}")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        click.echo(f"Error: {exc}", err=True)


@cli.command("rotate")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("angle", type=float)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to save the rotated image.",
)
def rotate_cmd(image_path: str, angle: float, output: Optional[str]) -> None:
    """Rotate an image by ANGLE degrees."""
    try:
        img = load_image(image_path)
        rotated_img = rotate_image(img, angle)
        out_path = (
            Path(output)
            if output
            else Path(image_path).with_name(f"rotated_{Path(image_path).name}")
        )
        rotated_img.save(out_path)
        click.echo(f"Rotated image by {angle} degrees saved to {out_path}")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        click.echo(f"Error: {exc}", err=True)


if __name__ == "__main__":  # pragma: no cover
    cli()
