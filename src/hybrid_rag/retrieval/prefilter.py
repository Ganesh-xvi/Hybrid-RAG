from hybrid_rag.config.settings import Settings, get_settings


def detect_company_filter(query: str, settings: Settings | None = None) -> str | None:
    cfg = settings or get_settings()
    lowered = query.lower()
    matches = [kw for kw in cfg.company_keywords if kw in lowered]
    if len(matches) == 1:
        return matches[0]
    return None
