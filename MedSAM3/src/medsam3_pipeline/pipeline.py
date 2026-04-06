from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import InferenceConfig
from .io_utils import case_id_from_path, ensure_dir, slugify
from .logger import close_logging, setup_logging
from .postprocess import combine_prompt_masks, serialize_prompt_result
from .reporting import write_csv_report, write_json_report
from .visualization import save_mask, save_overlay


@dataclass(slots=True)
class PromptArtifactPaths:
    case_dir: Path
    prompt_dir: Path
    metadata_path: Path
    mask_path: Path
    overlay_path: Path
    detections_path: Path


class BatchInferencePipeline:
    """Thin batch wrapper around the existing MedSAM3 inference code."""

    def __init__(self, config: InferenceConfig, *, verbose: bool = False) -> None:
        self.config = config
        self.output_dir = ensure_dir(config.output_dir)
        self.reports_dir = ensure_dir(self.output_dir / "_reports")
        self.logs_dir = ensure_dir(self.output_dir / "_logs")
        self.logger = setup_logging(self.logs_dir, verbose=verbose)
        self._inferencer = None

    def run(self, image_paths: list[Path]) -> list[dict[str, Any]]:
        """Run MedSAM3 inference for a list of images."""
        if not self.config.prompts:
            raise ValueError("At least one prompt is required.")

        records: list[dict[str, Any]] = []
        inferencer = self._get_inferencer()
        self.logger.info("Starting inference for %s images across %s prompts", len(image_paths), len(self.config.prompts))

        for image_path in image_paths:
            self.logger.info("Processing %s", image_path)
            try:
                results = inferencer.predict(str(image_path), self.config.prompts)
                image = results["_image"]
                result_keys = sorted(key for key in results.keys() if key != "_image")
                for result in [results[index] for index in result_keys]:
                    record = self._save_prompt_outputs(image_path, image, result)
                    records.append(record)
            except Exception as exc:  # pragma: no cover
                self.logger.exception("Inference failed for %s", image_path)
                for prompt in self.config.prompts:
                    records.append(self._build_failure_record(image_path, prompt, str(exc)))

        self._write_run_reports(records)
        return records

    def _get_inferencer(self):
        if self._inferencer is None:
            self._inferencer = self._build_inferencer()
        return self._inferencer

    def _build_inferencer(self):
        from infer_sam import SAM3LoRAInference

        self.logger.info("Initializing SAM3LoRAInference")
        return SAM3LoRAInference(
            config_path=str(self.config.config_path),
            weights_path=str(self.config.weights_path) if self.config.weights_path else None,
            resolution=self.config.resolution,
            detection_threshold=self.config.threshold,
            nms_iou_threshold=self.config.nms_iou,
        )

    def _save_prompt_outputs(self, image_path: Path, image, result: dict[str, Any]) -> dict[str, Any]:
        prompt = str(result["prompt"])
        paths = self._artifact_paths(image_path, prompt)
        serialized = serialize_prompt_result(result)
        mask = combine_prompt_masks(result)

        output_path = ""
        mask_path = ""
        if mask is not None:
            mask_path = str(save_mask(mask, paths.mask_path).resolve())
            output_path = str(save_overlay(image, mask, paths.overlay_path).resolve())

        detections_payload = {
            "input_path": str(image_path.resolve()),
            "prompt": prompt,
            "detections": serialized,
        }
        paths.detections_path.write_text(json.dumps(detections_payload, indent=2), encoding="utf-8")

        record = {
            "input_path": str(image_path.resolve()),
            "prompt": prompt,
            "output_path": output_path,
            "mask_path": mask_path,
            "detections_path": str(paths.detections_path.resolve()),
            "config_path": str(self.config.config_path.resolve()),
            "weights_path": "" if self.config.weights_path is None else str(self.config.weights_path.resolve()),
            "threshold": self.config.threshold,
            "resolution": self.config.resolution,
            "nms_iou": self.config.nms_iou,
            "run_timestamp": _utc_timestamp(),
            "success": True,
            "error_message": "",
            "num_detections": serialized["num_detections"],
            "scores": serialized["scores"],
            "boxes": serialized["boxes"],
        }
        paths.metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    def _build_failure_record(self, image_path: Path, prompt: str, error_message: str) -> dict[str, Any]:
        paths = self._artifact_paths(image_path, prompt)
        record = {
            "input_path": str(image_path.resolve()),
            "prompt": prompt,
            "output_path": "",
            "mask_path": "",
            "detections_path": "",
            "config_path": str(self.config.config_path.resolve()),
            "weights_path": "" if self.config.weights_path is None else str(self.config.weights_path.resolve()),
            "threshold": self.config.threshold,
            "resolution": self.config.resolution,
            "nms_iou": self.config.nms_iou,
            "run_timestamp": _utc_timestamp(),
            "success": False,
            "error_message": error_message,
            "num_detections": 0,
            "scores": [],
            "boxes": [],
        }
        paths.metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    def _write_run_reports(self, records: list[dict[str, Any]]) -> None:
        write_json_report(records, self.reports_dir / f"{self.config.run_name}.json")
        write_csv_report(records, self.reports_dir / f"{self.config.run_name}.csv")
        summary = {
            "run_name": self.config.run_name,
            "output_dir": str(self.output_dir.resolve()),
            "config": asdict(self.config),
            "total_records": len(records),
            "successful_records": sum(1 for record in records if record["success"]),
            "failed_records": sum(1 for record in records if not record["success"]),
        }
        summary["config"]["config_path"] = str(self.config.config_path.resolve())
        summary["config"]["output_dir"] = str(self.config.output_dir.resolve())
        summary["config"]["weights_path"] = (
            "" if self.config.weights_path is None else str(self.config.weights_path.resolve())
        )
        (self.reports_dir / f"{self.config.run_name}_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    def _artifact_paths(self, image_path: Path, prompt: str) -> PromptArtifactPaths:
        case_dir = ensure_dir(self.output_dir / case_id_from_path(image_path))
        prompt_dir = ensure_dir(case_dir / slugify(prompt))
        return PromptArtifactPaths(
            case_dir=case_dir,
            prompt_dir=prompt_dir,
            metadata_path=prompt_dir / "metadata.json",
            mask_path=prompt_dir / "mask.png",
            overlay_path=prompt_dir / "overlay.png",
            detections_path=prompt_dir / "detections.json",
        )

    def close(self) -> None:
        """Release logger file handles for short-lived processes and tests."""
        close_logging(self.logger)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
