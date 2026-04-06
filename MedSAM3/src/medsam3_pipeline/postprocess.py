from __future__ import annotations

from typing import Any

import numpy as np


def combine_prompt_masks(result: dict[str, Any]) -> np.ndarray | None:
    """Merge all predicted masks for a prompt into one binary mask."""
    masks = result.get("masks")
    if masks is None or len(masks) == 0:
        return None
    merged = np.any(np.asarray(masks, dtype=bool), axis=0)
    return merged.astype(np.uint8) * 255


def serialize_prompt_result(result: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy-heavy MedSAM3 output into JSON-friendly metadata."""
    boxes = result.get("boxes")
    scores = result.get("scores")
    return {
        "prompt": result.get("prompt"),
        "num_detections": int(result.get("num_detections", 0)),
        "scores": [] if scores is None else [float(score) for score in scores],
        "boxes": [] if boxes is None else [[float(value) for value in box] for box in boxes],
    }
