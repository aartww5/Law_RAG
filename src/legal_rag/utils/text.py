import re


WINDOW_BACKGROUND_PREFIX = "最近对话背景："
CURRENT_QUESTION_PREFIX = "当前问题："


def normalize_question(text: str) -> str:
    return " ".join(text.strip().split())


def keyword_tokens(text: str) -> list[str]:
    normalized = normalize_question(text)
    ascii_tokens = re.findall(r"[a-zA-Z0-9]+", normalized.lower())
    chinese_chars = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
    return ascii_tokens + chinese_chars


def split_structured_query(text: str) -> tuple[str, str]:
    normalized = normalize_question(text)
    if CURRENT_QUESTION_PREFIX not in normalized:
        return "", normalized

    background, current = normalized.rsplit(CURRENT_QUESTION_PREFIX, 1)
    if WINDOW_BACKGROUND_PREFIX in background:
        _, background = background.split(WINDOW_BACKGROUND_PREFIX, 1)
    return background.strip("；。 "), current.strip()
