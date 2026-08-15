# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_SYSTEM_PYTHON=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Runtime libs (pdfplumber / pillow / health checks)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (used when lockfile install succeeds)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependency manifests first (better layer cache)
COPY pyproject.toml uv.lock requirements.txt ./

# Install dependencies FIRST without the source code to cache the heavy downloads
RUN set -eux; \
    if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Application source
COPY src ./src
COPY main.py gunicorn.conf.py ./

# Install the local project itself (and sync via uv if available)
RUN set -eux; \
    if [ -f uv.lock ] && uv sync --frozen --no-dev; then \
        echo "Installed with uv sync"; \
    else \
        echo "uv sync unavailable — falling back to pip + requirements.txt"; \
        pip install --no-cache-dir .; \
    fi; \
    gunicorn --version

RUN mkdir -p /app/logs /app/storage /app/data

EXPOSE 8000

# Production: Gunicorn + UvicornWorker (see gunicorn.conf.py)
CMD ["gunicorn", "hybrid_rag.api.app:app", "-c", "gunicorn.conf.py"]
