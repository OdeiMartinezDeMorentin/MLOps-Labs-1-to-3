"""Unit tests for the CLI module."""

# pylint: disable=redefined-outer-name

import json
from pathlib import Path
from click.testing import CliRunner
from PIL import Image
import pytest

from cli.cli import cli
from mylib.image_processing import CLASSES


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Click CliRunner instance."""
    return CliRunner()


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    """Create a temporary test image file."""
    img_path = tmp_path / "sample.jpg"
    img = Image.new("RGB", (120, 80), color="green")
    img.save(img_path, format="JPEG")
    return img_path


def test_cli_help(runner: CliRunner):
    """Verify CLI help executes cleanly and lists commands."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "MLOps Command Line Interface" in result.output
    assert "predict" in result.output
    assert "resize" in result.output
    assert "grayscale" in result.output
    assert "rotate" in result.output


def test_classes_command(runner: CliRunner):
    """Verify 'classes' command returns the predefined classes list in JSON format."""
    result = runner.invoke(cli, ["classes"])
    assert result.exit_code == 0
    parsed = json.loads(result.output.strip())
    assert parsed == CLASSES


def test_predict_command_success(runner: CliRunner, image_file: Path):
    """Verify 'predict' command prints predicted class and confidence."""
    result = runner.invoke(cli, ["predict", str(image_file)])
    assert result.exit_code == 0
    assert "Class:" in result.output
    assert "confidence:" in result.output


def test_predict_command_nonexistent_file(runner: CliRunner):
    """Verify 'predict' command fails cleanly when file does not exist."""
    result = runner.invoke(cli, ["predict", "nonexistent_file.png"])
    assert result.exit_code != 0
    assert "does not exist" in result.output.lower() or "error" in result.output.lower()


def test_predict_command_corrupted_file(runner: CliRunner, tmp_path: Path):
    """Verify 'predict' handles corrupted/non-image files with an error message."""
    bad_file = tmp_path / "corrupted.jpg"
    bad_file.write_text("not real image binary data")
    result = runner.invoke(cli, ["predict", str(bad_file)])
    assert "Error:" in result.output


def test_resize_command_with_output(
    runner: CliRunner, image_file: Path, tmp_path: Path
):
    """Verify 'resize' command resizes image and saves to explicit output path."""
    out_file = tmp_path / "custom_resized.jpg"
    result = runner.invoke(
        cli, ["resize", str(image_file), "60", "40", "-o", str(out_file)]
    )
    assert result.exit_code == 0
    assert out_file.exists()
    resized = Image.open(out_file)
    assert resized.size == (60, 40)
    assert "Image resized to 60x40" in result.output


def test_resize_command_default_output(runner: CliRunner, image_file: Path):
    """Verify 'resize' command saves to default output name when -o is omitted."""
    result = runner.invoke(cli, ["resize", str(image_file), "50", "30"])
    assert result.exit_code == 0
    expected_out = image_file.with_name(f"resized_{image_file.name}")
    assert expected_out.exists()
    resized = Image.open(expected_out)
    assert resized.size == (50, 30)


def test_resize_command_invalid_dimensions(runner: CliRunner, image_file: Path):
    """Verify 'resize' displays an error message for invalid non-positive dimensions."""
    result = runner.invoke(cli, ["resize", str(image_file), "0", "40"])
    assert "Error:" in result.output
    assert "positive integers" in result.output


def test_resize_command_corrupted_file(runner: CliRunner, tmp_path: Path):
    """Verify 'resize' handles corrupted image file."""
    bad_file = tmp_path / "bad.jpg"
    bad_file.write_text("not an image")
    result = runner.invoke(cli, ["resize", str(bad_file), "50", "30"])
    assert "Error:" in result.output


def test_grayscale_command_with_output(
    runner: CliRunner, image_file: Path, tmp_path: Path
):
    """Verify 'grayscale' command converts image to grayscale and saves it."""
    out_file = tmp_path / "gray.jpg"
    result = runner.invoke(cli, ["grayscale", str(image_file), "-o", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    gray = Image.open(out_file)
    assert gray.mode == "L"


def test_grayscale_command_default_output(runner: CliRunner, image_file: Path):
    """Verify 'grayscale' command saves to default filename."""
    result = runner.invoke(cli, ["grayscale", str(image_file)])
    assert result.exit_code == 0
    expected_out = image_file.with_name(f"grayscale_{image_file.name}")
    assert expected_out.exists()


def test_grayscale_command_corrupted_file(runner: CliRunner, tmp_path: Path):
    """Verify 'grayscale' handles corrupted file."""
    bad_file = tmp_path / "bad.jpg"
    bad_file.write_text("not an image")
    result = runner.invoke(cli, ["grayscale", str(bad_file)])
    assert "Error:" in result.output


def test_rotate_command_with_output(
    runner: CliRunner, image_file: Path, tmp_path: Path
):
    """Verify 'rotate' command rotates image and saves to explicit output path."""
    out_file = tmp_path / "rotated.jpg"
    result = runner.invoke(cli, ["rotate", str(image_file), "90", "-o", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    rotated = Image.open(out_file)
    assert rotated.size == (80, 120)


def test_rotate_command_default_output(runner: CliRunner, image_file: Path):
    """Verify 'rotate' command saves to default filename."""
    result = runner.invoke(cli, ["rotate", str(image_file), "180"])
    assert result.exit_code == 0
    expected_out = image_file.with_name(f"rotated_{image_file.name}")
    assert expected_out.exists()


def test_rotate_command_corrupted_file(runner: CliRunner, tmp_path: Path):
    """Verify 'rotate' handles corrupted file."""
    bad_file = tmp_path / "bad.jpg"
    bad_file.write_text("not an image")
    result = runner.invoke(cli, ["rotate", str(bad_file), "45"])
    assert "Error:" in result.output
