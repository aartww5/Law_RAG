from collections.abc import Iterator
from pathlib import Path


def iter_law_files(laws_dir: str | Path) -> Iterator[Path]:
    path = Path(laws_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Laws directory does not exist: {path}")

    for txt_file in sorted(path.glob("*.txt")):
        yield txt_file


def iter_raw_law_lines(laws_dir: str | Path) -> Iterator[tuple[str, str, int, str]]:
    for txt_file in iter_law_files(laws_dir):
        law_name = txt_file.stem
        for line_no, line in enumerate(txt_file.read_text(encoding="utf-8").splitlines(), start=1):
            yield str(txt_file), law_name, line_no, line
