from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml
from PIL import Image

from src.medsam3_pipeline.config import DEFAULT_PROMPTS, InferenceConfig
from src.medsam3_pipeline.io_utils import ensure_dir
from src.medsam3_pipeline.pipeline import BatchInferencePipeline


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "full_lora_config.yaml"
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
UPLOAD_DIR = DEFAULT_OUTPUTS / "_uploads"


def main() -> None:
    st.set_page_config(page_title="MedSAM3 Tester", layout="wide")
    st.title("MedSAM3 Segmentation Tester")
    st.caption("Experiment UI for text-guided segmentation on PNG and JPG images.")

    with st.sidebar:
        st.header("Run Settings")
        config_path = Path(
            st.text_input("Config path", value=str(DEFAULT_CONFIG))
        )
        weights_value = st.text_input("Weights path (optional)", value="")
        output_dir = Path(
            st.text_input("Output directory", value=str(DEFAULT_OUTPUTS))
        )
        threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        nms_iou = st.slider("NMS IoU", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        resolution = st.number_input("Resolution", min_value=256, max_value=2048, value=1008, step=16)
        selected_prompts = st.multiselect(
            "Prompts",
            options=DEFAULT_PROMPTS,
            default=DEFAULT_PROMPTS,
        )
        extra_prompts = st.text_input("Additional prompts", value="", help="Comma-separated custom prompts.")

    upload = st.file_uploader("Upload a PNG or JPG image", type=["png", "jpg", "jpeg"])
    run_button = st.button("Run Inference", type="primary", disabled=upload is None)

    if not run_button or upload is None:
        st.info("Choose an image, adjust settings if needed, and click Run Inference.")
        return

    prompts = _collect_prompts(selected_prompts, extra_prompts)
    if not prompts:
        st.error("Select at least one prompt.")
        return

    if not config_path.exists():
        st.error(f"Config file not found: {config_path}")
        return

    weights_path = Path(weights_value).expanduser() if weights_value.strip() else None
    if weights_path and not weights_path.exists():
        st.error(f"Weights file not found: {weights_path}")
        return

    resolved_weights = _resolve_weights_path(config_path, weights_path)
    if resolved_weights is None:
        st.error(
            "No LoRA weights were found. "
            "Set a valid weights path in the sidebar, or update the config output path to point at the trained weights file."
        )
        st.code(str(_config_default_weights_path(config_path)))
        return

    image_path = _save_upload(upload)
    st.image(Image.open(image_path), caption=image_path.name, use_container_width=True)

    config = InferenceConfig.from_values(
        config_path=config_path,
        output_dir=output_dir,
        prompts=prompts,
        weights_path=resolved_weights,
        threshold=float(threshold),
        resolution=int(resolution),
        nms_iou=float(nms_iou),
        run_name=f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    with st.spinner("Running MedSAM3 inference..."):
        pipeline = BatchInferencePipeline(config, verbose=True)
        try:
            records = pipeline.run([image_path])
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.exception(exc)
            return
        finally:
            pipeline.close()

    _render_results(records)


def _collect_prompts(selected_prompts: list[str], extra_prompts: str) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for prompt in list(selected_prompts) + [part.strip() for part in extra_prompts.split(",") if part.strip()]:
        if prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
    return prompts


def _save_upload(upload) -> Path:
    ensure_dir(UPLOAD_DIR)
    image_path = UPLOAD_DIR / upload.name
    image_path.write_bytes(upload.getvalue())
    return image_path


def _resolve_weights_path(config_path: Path, weights_path: Path | None) -> Path | None:
    if weights_path is not None:
        return weights_path if weights_path.exists() else None

    candidate = _config_default_weights_path(config_path)
    return candidate if candidate.exists() else None


def _config_default_weights_path(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    output_dir = config.get("output", {}).get("output_dir", "outputs/sam3_lora_full")
    return (config_path.parent.parent / output_dir / "best_lora_weights.pt").resolve()


def _render_results(records: list[dict]) -> None:
    st.subheader("Results")

    if not records:
        st.warning("The pipeline returned no records.")
        return

    for record in records:
        with st.container(border=True):
            cols = st.columns([1.2, 1.2, 1.6])
            with cols[0]:
                st.markdown(f"**Prompt**: `{record['prompt']}`")
                st.markdown(f"**Success**: `{record['success']}`")
                st.markdown(f"**Detections**: `{record['num_detections']}`")
                if not record["success"]:
                    st.error(record["error_message"])
            with cols[1]:
                if record["output_path"]:
                    st.image(record["output_path"], caption="Overlay", use_container_width=True)
                else:
                    st.caption("No overlay saved")
                if record["mask_path"]:
                    st.image(record["mask_path"], caption="Mask", use_container_width=True)
            with cols[2]:
                st.json(_record_for_display(record))
                if record["detections_path"]:
                    detections = json.loads(Path(record["detections_path"]).read_text(encoding="utf-8"))
                    with st.expander("Detections JSON"):
                        st.json(detections)


def _record_for_display(record: dict) -> dict:
    return {
        "input_path": record["input_path"],
        "prompt": record["prompt"],
        "output_path": record["output_path"],
        "mask_path": record["mask_path"],
        "threshold": record["threshold"],
        "resolution": record["resolution"],
        "nms_iou": record["nms_iou"],
        "num_detections": record["num_detections"],
        "scores": record["scores"],
        "success": record["success"],
        "error_message": record["error_message"],
    }


if __name__ == "__main__":
    main()
