from functools import wraps
import time
from collections.abc import Callable
from typing import Any, TypeVar

from hybrid_rag.config.settings import get_settings
from hybrid_rag.utils.logging import logger

F = TypeVar("F", bound=Callable[..., Any])


def retry(max_attempts: int | None = None, delay: float = 1.0) -> Callable[[F], F]:
    attempts = max_attempts or get_settings().dlq_max_retries

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "%s failed (attempt %s/%s): %s",
                        func.__name__,
                        attempt,
                        attempts,
                        exc,
                    )
                    if attempt < attempts:
                        time.sleep(delay * attempt)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
