def extract_company(filename: str) -> str:
    name = filename.lower()
    if "honeywell" in name:
        return "honeywell"
    if "cloudflare" in name:
        return "cloudflare"
    return "unknown"
