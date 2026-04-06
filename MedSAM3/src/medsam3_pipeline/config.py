from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

DEFAULT_PROMPTS = ["liver", "spleen", "kidney", "brain"]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(slots=True)
class InferenceConfig:
    """Runtime settings for MedSAM3 batch inference."""

    config_path: Path
    output_dir: Path
    weights_path: Path | None = None
    threshold: float = 0.5
    resolution: int = 1008
    nms_iou: float = 0.5
    prompts: list[str] = field(default_factory=lambda: DEFAULT_PROMPTS.copy())
    run_name: str = field(default_factory=lambda: f"run_{_timestamp()}")

    @classmethod
    def from_values(
        cls,
        config_path: str | Path,
        output_dir: str | Path,
        prompts: Iterable[str],
        *,
        weights_path: str | Path | None = None,
        threshold: float = 0.5,
        resolution: int = 1008,
        nms_iou: float = 0.5,
        run_name: str | None = None,
    ) -> "InferenceConfig":
        return cls(
            config_path=Path(config_path),
            output_dir=Path(output_dir),
            weights_path=Path(weights_path) if weights_path else None,
            threshold=threshold,
            resolution=resolution,
            nms_iou=nms_iou,
            prompts=[prompt.strip() for prompt in prompts if prompt.strip()] or DEFAULT_PROMPTS.copy(),
            run_name=run_name or f"run_{_timestamp()}",
        )


@dataclass(slots=True)
class SliceExtractionConfig:
    """Settings for extracting 2D slices from NIfTI volumes."""

    output_dir: Path
    axis: int = 2
    step: int = 1
    max_slices: int | None = None
    run_name: str = field(default_factory=lambda: f"slices_{_timestamp()}")

    @property
    def run_dir(self) -> Path:
        return self.output_dir / "extracted_slices" / self.run_name

    @classmethod
    def from_values(
        cls,
        output_dir: str | Path,
        *,
        axis: int = 2,
        step: int = 1,
        max_slices: int | None = None,
        run_name: str | None = None,
    ) -> "SliceExtractionConfig":
        return cls(
            output_dir=Path(output_dir),
            axis=axis,
            step=step,
            max_slices=max_slices,
            run_name=run_name or f"slices_{_timestamp()}",
        )
