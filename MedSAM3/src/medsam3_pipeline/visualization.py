from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_mask(mask: np.ndarray, output_path: str | Path) -> Path:
    """Persist a binary mask as a PNG image."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(output)
    return output


def save_overlay(
    image: Image.Image,
    mask: np.ndarray,
    output_path: str | Path,
    *,
    alpha: float = 0.35,
    color: tuple[int, int, int] = (255, 80, 80),
) -> Path:
    """Save a lightweight color overlay for quick visual review."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    image_rgba = image.convert("RGBA")
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[mask > 0] = (*color, int(255 * alpha))

    composited = Image.alpha_composite(image_rgba, Image.fromarray(overlay, mode="RGBA"))
    composited.save(output)
    return output
