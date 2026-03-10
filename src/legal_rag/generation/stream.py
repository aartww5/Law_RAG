from collections.abc import Iterator


def iter_text_chunks(text: str, chunk_size: int = 24) -> Iterator[str]:
    if not text:
        return

    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]
