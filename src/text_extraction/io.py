"""Small JSONL helpers shared by both pipelines."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Iterable, Iterator


def iter_records(
    path: str | Path,
    *,
    input_format: str = "auto",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream JSONL or CSV objects without loading clinical text into memory."""
    source = Path(path)
    resolved_format = input_format
    if resolved_format == "auto":
        resolved_format = "csv" if source.suffix.lower() == ".csv" else "jsonl"
    if resolved_format not in {"csv", "jsonl"}:
        raise ValueError("input_format must be auto, csv, or jsonl")

    with source.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        rows: Iterable[dict[str, Any]]
        if resolved_format == "csv":
            rows = csv.DictReader(handle)
        else:
            rows = _iter_jsonl_handle(handle)
        for index, row in enumerate(rows):
            if limit is not None and index >= limit:
                break
            yield dict(row)


def _iter_jsonl_handle(handle: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Line {line_number} is not a JSON object")
        yield value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_records(path, input_format="jsonl"))


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
