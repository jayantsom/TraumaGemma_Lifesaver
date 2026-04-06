from __future__ import annotations

import json
from pathlib import Path

import yaml

from .config import DEFAULT_PROMPTS


def load_prompts(prompts: list[str] | None = None, prompts_file: str | Path | None = None) -> list[str]:
    """Load prompts from CLI args and optional text/JSON/YAML file."""
    combined: list[str] = []
    if prompts:
        combined.extend(prompt.strip() for prompt in prompts if prompt and prompt.strip())
    if prompts_file:
        combined.extend(_load_prompt_file(Path(prompts_file)))

    seen: set[str] = set()
    unique: list[str] = []
    for prompt in combined:
        if prompt not in seen:
            unique.append(prompt)
            seen.add(prompt)
    return unique or DEFAULT_PROMPTS.copy()


def _load_prompt_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".json"):
        data = json.loads(path.read_text(encoding="utf-8"))
    elif suffixes.endswith((".yaml", ".yml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if isinstance(data, list):
        return [str(item).strip() for item in data if str(item).strip()]
    if isinstance(data, dict) and isinstance(data.get("prompts"), list):
        return [str(item).strip() for item in data["prompts"] if str(item).strip()]
    raise ValueError(f"Unsupported prompts file format: {path}")
