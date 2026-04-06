from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def write_json_report(records: Iterable[dict], output_path: str | Path) -> Path:
    """Write a JSON report for a pipeline run."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return output


def write_csv_report(records: Iterable[dict], output_path: str | Path) -> Path:
    """Write a flat CSV report for a pipeline run."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    fieldnames = _collect_fieldnames(rows)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _flatten_value(row.get(key)) for key in fieldnames})
    return output


def _collect_fieldnames(rows: list[dict]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _flatten_value(value: object) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return "" if value is None else value
    return json.dumps(value, ensure_ascii=True)
