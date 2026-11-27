# ---------------------------------------------------------------------------
# Stage 1 (builder): installs the runtime dependencies from uv.lock
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

# Compile .pyc files at install time (faster start-up) and copy instead of hard-linking
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv

WORKDIR /app

# Only the dependency files: this layer is cached while they do not change
COPY pyproject.toml uv.lock ./

# Export exact runtime dependencies from uv.lock (--no-dev excludes pytest, pylint, black, gradio, modal)
# and install them into the system Python prefix (/install) because Modal runs with system Python.
RUN uv export --frozen --no-dev --no-emit-project --no-hashes -o requirements.txt \
    && uv pip install --system --no-cache --prefix /install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 (runtime): only the installed packages and the source code
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY --from=builder /install /usr/local
COPY api ./api
COPY mylib ./mylib
COPY templates ./templates

# Note: the image runs as root because Modal adds its own layers to the system Python.
EXPOSE 8000

# Docker marks the container as unhealthy if /health does not answer
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT', '8000'), timeout=4)" || exit 1

# Used by `docker run` (locally and in the CI smoke test). Modal imports FastAPI directly via modal_app.py.
CMD ["sh", "-c", "exec uvicorn api.api:app --host 0.0.0.0 --port ${PORT}"]
