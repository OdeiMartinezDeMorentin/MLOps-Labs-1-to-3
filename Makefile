# We use Bash for complex commands and consistency
SHELL := /bin/bash
# We ensure that ~/.local/bin (where uv can be installed) is in the PATH
PATH := $(HOME)/.local/bin:$(PATH)

# Docker image name and local port (can be overridden: make docker-run PORT=9000)
IMAGE ?= image-classifier-api
PORT ?= 8000
# Source folders and files checked by black and pylint
SRC := mylib api cli gui modal_app.py gui_app.py

# Specify targets that are orders, not files
.PHONY: install format format-check lint test run-api run-gui docker-build docker-run \
	modal-serve modal-deploy modal-deploy-gui refactor all

# Installs uv (if needed) and the dependencies, failing if uv.lock is out of date
install:
	@if ! command -v uv &> /dev/null; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	uv sync --locked

# Formats the code (locally)
format:
	uv run black $(SRC) tests

# Checks the format without modifying files (CI): fails if black would change something
format-check:
	uv run black --check --diff $(SRC) tests

lint:
	uv run pylint --recursive=y --disable=R,C $(SRC)

test:
	uv run python -m pytest tests -vv --cov=mylib --cov=api --cov=cli --cov=gui \
		--cov-report=term-missing --cov-fail-under=90

# Runs the API locally with auto-reload: http://localhost:8000/docs
run-api:
	uv run uvicorn api.api:app --reload --port $(PORT)

# Runs the Gradio GUI locally against the local API (start it first with make run-api):
# http://localhost:7860
run-gui:
	API_URL=http://localhost:$(PORT) uv run python gui/app.py

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):8000 $(IMAGE)

# Temporary deployment on Modal with live logs (stopped with Ctrl+C)
modal-serve:
	uv run modal serve modal_app.py

# Deployment of the backend on Modal (it uses the image published in Docker Hub)
modal-deploy:
	uv run modal deploy modal_app.py

# Deployment of the GUI on Modal (API_URL must contain the URL of the backend)
modal-deploy-gui:
	uv run modal deploy gui_app.py

refactor: format lint

all: install format-check lint test
