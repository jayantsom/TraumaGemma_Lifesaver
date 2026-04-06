from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import SliceExtractionConfig
from .io_utils import case_id_from_path, ensure_dir
from .reporting import write_csv_report


def extract_slices(
    volume_paths: list[Path],
    config: SliceExtractionConfig,
) -> list[dict]:
    """Extract normalized grayscale PNG slices from one or more NIfTI volumes."""
    nib = _require_nibabel()
    run_dir = ensure_dir(config.run_dir)
    records: list[dict] = []

    for volume_path in volume_paths:
        volume = nib.load(str(volume_path))
        data = np.asarray(volume.get_fdata(), dtype=np.float32)
        if config.axis < 0 or config.axis >= data.ndim:
            raise ValueError(f"slice axis {config.axis} is out of range for volume with shape {data.shape}")
        case_id = case_id_from_path(volume_path)
        case_dir = ensure_dir(run_dir / case_id)
        selected_indices = _slice_indices(data.shape[config.axis], step=config.step, max_slices=config.max_slices)

        for slice_index in selected_indices:
            slice_array = np.take(data, indices=slice_index, axis=config.axis)
            normalized = _normalize_slice(slice_array)
            output_path = case_dir / f"{case_id}_axis{config.axis}_slice{slice_index:04d}.png"
            Image.fromarray(normalized, mode="L").save(output_path)
            records.append(
                {
                    "volume_path": str(volume_path.resolve()),
                    "case_id": case_id,
                    "slice_axis": config.axis,
                    "slice_index": slice_index,
                    "output_path": str(output_path.resolve()),
                }
            )

    (run_dir / "slice_map.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    write_csv_report(records, run_dir / "slice_map.csv")
    return records


def _require_nibabel():
    try:
        import nibabel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "NIfTI extraction requires nibabel. The current environment does not have it available."
        ) from exc
    return nibabel


def _slice_indices(size: int, *, step: int, max_slices: int | None) -> list[int]:
    if step <= 0:
        raise ValueError("step must be greater than zero")
    indices = list(range(0, size, step))
    if max_slices is not None:
        indices = indices[:max_slices]
    return indices


def _normalize_slice(slice_array: np.ndarray) -> np.ndarray:
    finite_values = slice_array[np.isfinite(slice_array)]
    if finite_values.size == 0:
        return np.zeros(slice_array.shape, dtype=np.uint8)

    low, high = np.percentile(finite_values, [1, 99])
    if high <= low:
        return np.zeros(slice_array.shape, dtype=np.uint8)

    clipped = np.clip(slice_array, low, high)
    normalized = (clipped - low) / (high - low)
    return np.round(normalized * 255).astype(np.uint8)
