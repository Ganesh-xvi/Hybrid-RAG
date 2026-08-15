"""Gunicorn configuration for production (Docker / Linux).

Uses Uvicorn workers for FastAPI async support.
All values can be overridden via environment variables — see .env.example.
"""

from __future__ import annotations

import multiprocessing
import os


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_host = os.getenv("API_HOST", "0.0.0.0")
_port = os.getenv("API_PORT", "8000")
_default_workers = max(2, multiprocessing.cpu_count() * 2 + 1)

bind = os.getenv("GUNICORN_BIND", f"{_host}:{_port}")
workers = _env_int("GUNICORN_WORKERS", _default_workers)
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "uvicorn.workers.UvicornWorker")
threads = _env_int("GUNICORN_THREADS", 1)

timeout = _env_int("GUNICORN_TIMEOUT", 120)
graceful_timeout = _env_int("GUNICORN_GRACEFUL_TIMEOUT", 30)
keepalive = _env_int("GUNICORN_KEEPALIVE", 5)

accesslog = os.getenv("GUNICORN_ACCESS_LOG", "-")
errorlog = os.getenv("GUNICORN_ERROR_LOG", "-")
loglevel = os.getenv("GUNICORN_LOG_LEVEL", os.getenv("LOG_LEVEL", "info")).lower()
capture_output = _env_bool("GUNICORN_CAPTURE_OUTPUT", True)
preload_app = _env_bool("GUNICORN_PRELOAD", False)

# Required for reverse proxies / Docker port mapping
forwarded_allow_ips = os.getenv("GUNICORN_FORWARDED_ALLOW_IPS", "*")
proxy_allow_ips = os.getenv("GUNICORN_PROXY_ALLOW_IPS", "*")

# Process naming (shows in `ps` / container logs)
proc_name = os.getenv("GUNICORN_PROC_NAME", os.getenv("LOG_NAME", "hybrid_rag"))


def on_starting(server) -> None:
    server.log.info("Gunicorn starting | bind=%s workers=%s", bind, workers)


def when_ready(server) -> None:
    server.log.info("Gunicorn ready | pid=%s", os.getpid())


def worker_exit(server, worker) -> None:
    server.log.info("Worker exited | pid=%s", worker.pid)
