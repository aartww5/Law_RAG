import re


def normalize_question(text: str) -> str:
    return " ".join(text.strip().split())


def keyword_tokens(text: str) -> list[str]:
    normalized = normalize_question(text)
    ascii_tokens = re.findall(r"[a-zA-Z0-9]+", normalized.lower())
    chinese_chars = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
    return ascii_tokens + chinese_chars
