# MLOps - Labs 1 to 3

[![CICD](https://github.com/OdeiMartinezDeMorentin/MLOps-Labs-1-to-3/actions/workflows/CICD.yml/badge.svg)](https://github.com/OdeiMartinezDeMorentin/MLOps-Labs-1-to-3/actions/workflows/CICD.yml)

Repository for Labs 1, 2, and 3 of the University Master's Degree in Machine Learning (MLOps course).

## Overview

Incremental MLOps image processing and classification project featuring:
- **Modular Logic (`mylib/`)**: Image preprocessing (resizing, grayscale, rotation with Pillow) and dummy class prediction.
- **CLI (`cli/`)**: Command-line interface built with Click.
- **REST API (`api/`)**: Microservice built with FastAPI, interactive Swagger docs (`/docs`), health check (`/health`), and Jinja2 home template (`/`).
- **Gradio GUI (`gui/`)**: Frontend microservice communicating with the REST API over HTTP (`POST /predict`).
- **Containerization (`Dockerfile`)**: Multi-stage Docker image built with `uv` for production deployment.
- **Cloud Hosting on Modal (`modal_app.py`, `gui_app.py`)**: Serverless hosting for both the FastAPI backend and Gradio frontend.
- **Testing & Quality (`tests/`)**: Unit and integration tests with `pytest`, coverage reporting with `pytest-cov`, formatting with `black`, and linting with `pylint`.
- **CI/CD Pipeline (`.github/workflows/CICD.yml`)**: Automated formatting/linting/testing, Docker build & smoke test, Docker Hub push, and Modal deployment on pushes to `main`.

---

## Setup & Installation

The project uses [uv](https://docs.astral.sh/uv/) for deterministic Python environment and dependency management with Python 3.13 pinned in `.python-version`.

```bash
make install
# or directly:
uv sync --locked
```

---

## Usage

### Command Line Interface (CLI)
```bash
uv run python -m cli.cli --help
uv run python -m cli.cli classes
uv run python -m cli.cli predict path/to/image.jpg
uv run python -m cli.cli resize path/to/image.jpg 200 150 -o resized.jpg
uv run python -m cli.cli grayscale path/to/image.jpg -o gray.jpg
uv run python -m cli.cli rotate path/to/image.jpg 90 -o rotated.jpg
```

### REST API (Local)
```bash
make run-api
# or:
uv run python -m api.api
```
- API Home: `http://localhost:8000/`
- Interactive API Docs (Swagger): `http://localhost:8000/docs`
- Health check: `GET http://localhost:8000/health`

### Gradio GUI (Local)
Start the API first (`make run-api`), then in another terminal:
```bash
make run-gui
# or:
API_URL=http://localhost:8000 uv run python gui/app.py
```
- GUI URL: `http://localhost:7860`

### Docker (Local)
```bash
make docker-build
make docker-run
```

---

## Code Quality & Testing

```bash
make format        # Auto-format code with black
make format-check  # Check formatting without modifying files (CI)
make lint          # Run pylint
make test          # Run pytest with >= 90% coverage check
make all           # Run install, format-check, lint, test
```

---

## License
MIT License. See [LICENSE](LICENSE) for details.
