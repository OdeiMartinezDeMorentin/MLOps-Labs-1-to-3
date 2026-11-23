# MLOps - Labs 1 to 3

[![CI](https://github.com/OdeiMartinezDeMorentin/MLOps-Labs-1-to-3/actions/workflows/CI.yml/badge.svg)](https://github.com/OdeiMartinezDeMorentin/MLOps-Labs-1-to-3/actions/workflows/CI.yml)

Repository for Labs 1, 2, and 3 of the University Master's Degree in Machine Learning (MLOps course).

## Lab 1: Continuous Integration using GitHub Actions

This first lab establishes an incremental image processing and classification service with:
- **Modular Logic (`mylib/`)**: Image preprocessing (resizing with Pillow) and dummy class prediction.
- **CLI (`cli/`)**: Command-line interface built with Click.
- **REST API (`api/`)**: Microservice built with FastAPI, including interactive Swagger docs (`/docs`), a health check (`/health`), and a Jinja2 home template (`/`).
- **Testing & Quality (`tests/`)**: Unit and integration tests with `pytest`, coverage reporting with `pytest-cov`, code formatting with `black`, and linting with `pylint`.
- **Automation & CI**: Local automation via `Makefile` and Continuous Integration pipeline via GitHub Actions (`.github/workflows/CI.yml`).

---

## Setup & Installation

The project uses [uv](https://docs.astral.sh/uv/) for fast, deterministic Python environment and dependency management with Python 3.13 pinned in `.python-version`.

### Prerequisites
- Python 3.13
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Install Dependencies
To install runtime and development dependencies matching `uv.lock`:
```bash
make install
# or directly:
uv sync --locked
```

---

## Usage

### Command Line Interface (CLI)
```bash
# Example help
uv run python -m cli.cli --help
```

### REST API
```bash
# Run API with auto-reload (default: http://localhost:8000)
make run-api
# or:
uv run uvicorn api.api:app --reload --port 8000
```
- API Home: `http://localhost:8000/`
- Interactive API Documentation (Swagger): `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

---

## Code Quality & Testing

All quality commands are automated via the `Makefile`:

```bash
# Auto-format code with black
make format

# Check formatting without modifying files (used in CI)
make format-check

# Run linter with pylint
make lint

# Run all tests with coverage check
make test

# Full pipeline locally (install, format-check, lint, test)
make all
```

---

## License
MIT License. See [LICENSE](LICENSE) for details.
