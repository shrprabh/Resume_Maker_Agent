"""Normalize secrets pasted from dashboards, shells, and .env files."""


def normalize_api_key(value: str | None, variable_name: str) -> str:
    if not value:
        return ""
    key = value.strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    prefix = f"{variable_name}="
    if key.upper().startswith(prefix):
        key = key[len(prefix) :].strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
        key = key[1:-1].strip()
    return key


def normalize_env_text(value: str | None) -> str:
    """Normalize non-secret text read through dotenv or Docker --env-file."""
    if not value:
        return ""
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text
