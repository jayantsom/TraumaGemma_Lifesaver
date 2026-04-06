from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import InferenceConfig, SliceExtractionConfig
from .io_utils import discover_images, discover_volumes
from .nifti import extract_slices
from .pipeline import BatchInferencePipeline
from .prompts import load_prompts


def build_parser() -> argparse.ArgumentParser:
    """Create the MedSAM3 pipeline CLI parser."""
    parser = argparse.ArgumentParser(description="Batch utilities for text-guided MedSAM3 inference.")
    parser.add_argument("--image", type=Path, help="Single input image or NIfTI volume.")
    parser.add_argument("--input-dir", type=Path, help="Directory of input images or NIfTI volumes.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for run outputs.")
    parser.add_argument("--prompt", nargs="+", default=None, help="One or more text prompts.")
    parser.add_argument("--prompts-file", type=Path, help="Text, JSON, or YAML file with prompts.")
    parser.add_argument("--config", type=Path, help="Path to MedSAM3 YAML config for inference.")
    parser.add_argument("--weights", type=Path, help="Optional LoRA weights path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold.")
    parser.add_argument("--resolution", type=int, default=1008, help="Inference resize resolution.")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--extract-slices", action="store_true", help="Run NIfTI slice extraction instead of inference.")
    parser.add_argument("--slice-axis", type=int, default=2, help="Axis to slice for NIfTI extraction.")
    parser.add_argument("--slice-step", type=int, default=1, help="Stride between extracted slices.")
    parser.add_argument("--max-slices", type=int, help="Maximum number of slices per volume.")
    parser.add_argument("--run-name", type=str, help="Optional custom run name.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.extract_slices:
        volume_paths = _resolve_volume_inputs(args.image, args.input_dir, parser)
        config = SliceExtractionConfig.from_values(
            args.output_dir,
            axis=args.slice_axis,
            step=args.slice_step,
            max_slices=args.max_slices,
            run_name=args.run_name,
        )
        extract_slices(volume_paths, config)
        return 0

    image_paths = _resolve_image_inputs(args.image, args.input_dir, parser)
    prompts = load_prompts(args.prompt, args.prompts_file)
    if args.config is None:
        parser.error("--config is required for inference.")

    config = InferenceConfig.from_values(
        config_path=args.config,
        output_dir=args.output_dir,
        prompts=prompts,
        weights_path=args.weights,
        threshold=args.threshold,
        resolution=args.resolution,
        nms_iou=args.nms_iou,
        run_name=args.run_name,
    )
    pipeline = BatchInferencePipeline(config, verbose=args.verbose)
    try:
        pipeline.run(image_paths)
    finally:
        pipeline.close()
    return 0


def _resolve_image_inputs(image: Path | None, input_dir: Path | None, parser: argparse.ArgumentParser) -> list[Path]:
    if image and input_dir:
        parser.error("Use either --image or --input-dir, not both.")
    if image:
        return [image]
    if input_dir:
        paths = discover_images(input_dir)
        if not paths:
            parser.error(f"No PNG/JPG images found in {input_dir}")
        return paths
    parser.error("One of --image or --input-dir is required.")
    raise AssertionError("unreachable")


def _resolve_volume_inputs(image: Path | None, input_dir: Path | None, parser: argparse.ArgumentParser) -> list[Path]:
    if image and input_dir:
        parser.error("Use either --image or --input-dir, not both.")
    if image:
        return [image]
    if input_dir:
        paths = discover_volumes(input_dir)
        if not paths:
            parser.error(f"No NIfTI files found in {input_dir}")
        return paths
    parser.error("One of --image or --input-dir is required for slice extraction.")
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
