import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from hybrid_rag.config.settings import Settings, get_settings

_CONFIGURED = False


def setup_logging(settings: Settings | None = None) -> logging.Logger:
    global _CONFIGURED
    cfg = settings or get_settings()
    log = logging.getLogger(cfg.log_name)

    if _CONFIGURED and log.handlers:
        return log

    log.handlers.clear()
    log.propagate = False
    log.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    if cfg.log_to_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        log.addHandler(console)

    if cfg.log_file:
        log_path = Path(cfg.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

    _CONFIGURED = True
    return log


def get_logger() -> logging.Logger:
    return setup_logging()


logger = get_logger()
