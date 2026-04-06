from __future__ import annotations

import re
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
NIFTI_EXTENSIONS = {".nii", ".nii.gz"}


def discover_images(input_dir: str | Path) -> list[Path]:
    """Return supported image files from a directory."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def discover_volumes(input_dir: str | Path) -> list[Path]:
    """Return supported NIfTI files from a directory."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    return sorted(path for path in root.iterdir() if path.is_file() and _is_nifti(path))


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def case_id_from_path(path: str | Path) -> str:
    file_path = Path(path)
    name = file_path.name
    if name.lower().endswith(".nii.gz"):
        return slugify(name[:-7])
    return slugify(file_path.stem)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "item"


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in NIFTI_EXTENSIONS)
