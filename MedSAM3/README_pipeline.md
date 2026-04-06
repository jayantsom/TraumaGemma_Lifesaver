# MedSAM3 Batch Pipeline

This package adds a small batch-oriented wrapper around the existing `infer_sam.py` flow. It is meant for experimentation and evaluation, not clinical use.

## What It Does

- Runs text-guided MedSAM3 inference on one image or a folder of images
- Saves outputs by case and by prompt
- Writes per-prompt metadata plus consolidated JSON and CSV run reports
- Extracts PNG slices from `.nii` and `.nii.gz` volumes for later MedSAM3 runs
- Uses `liver`, `spleen`, `kidney`, and `brain` by default when prompts are not provided

## Layout

- `src/medsam3_pipeline/cli.py`: CLI entrypoint
- `src/medsam3_pipeline/pipeline.py`: batch inference runner
- `src/medsam3_pipeline/nifti.py`: NIfTI slice extraction
- `src/medsam3_pipeline/reporting.py`: JSON and CSV reporting
- `src/medsam3_pipeline/visualization.py`: mask and overlay helpers
- `tests/`: non-model smoke tests
- `examples/prompts.txt`: sample prompts file

## Commands

Run CLI help:

```powershell
.venv\Scripts\python.exe -m src.medsam3_pipeline.cli --help
```

Single-image inference:

```powershell
.venv\Scripts\python.exe -m src.medsam3_pipeline.cli `
  --config configs\full_lora_config.yaml `
  --image path\to\image.png `
  --output-dir outputs
```

Batch inference from a folder:

```powershell
.venv\Scripts\python.exe -m src.medsam3_pipeline.cli `
  --config configs\full_lora_config.yaml `
  --input-dir path\to\png_folder `
  --output-dir outputs `
  --prompts-file examples\prompts.txt `
  --threshold 0.5 `
  --resolution 1008 `
  --nms-iou 0.5
```

Batch inference with explicit weights:

```powershell
.venv\Scripts\python.exe -m src.medsam3_pipeline.cli `
  --config configs\full_lora_config.yaml `
  --input-dir path\to\png_folder `
  --output-dir outputs `
  --prompts-file examples\prompts.txt `
  --weights path\to\best_lora_weights.pt
```

Extract PNG slices from one NIfTI volume:

```powershell
.venv\Scripts\python.exe -m src.medsam3_pipeline.cli `
  --extract-slices `
  --image path\to\volume.nii.gz `
  --output-dir outputs `
  --slice-axis 2 `
  --slice-step 4 `
  --max-slices 32
```

Extract PNG slices from a folder of volumes:

```powershell
.venv\Scripts\python.exe -m src.medsam3_pipeline.cli `
  --extract-slices `
  --input-dir path\to\nifti_folder `
  --output-dir outputs `
  --slice-axis 1 `
  --slice-step 2
```

Run smoke tests:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_prompts.py tests\test_io_utils.py tests\test_reporting.py tests\test_pipeline_paths.py tests\test_nifti.py
```

Run the Streamlit tester:

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

If Streamlit is not installed yet:

```powershell
.venv\Scripts\python.exe -m pip install -e .[ui]
```

## Output Structure

Inference results are written directly under `outputs/{case_id}/{prompt_slug}/`.

For each case and prompt:

- `mask.png`: merged binary mask for that prompt
- `overlay.png`: quick visual overlay on the source image
- `detections.json`: raw prompt-level detections summary
- `metadata.json`: prompt-level execution metadata

Run-level reports are written under `outputs/_reports/`:

- `{run_name}.json`
- `{run_name}.csv`
- `{run_name}_summary.json`

Logs are written to `outputs/_logs/run.log`.

Slice extraction runs are written under `outputs/extracted_slices/{run_name}/` with:

- per-volume PNG slices
- `slice_map.json`
- `slice_map.csv`

## Notes

- The pipeline reuses `SAM3LoRAInference` from `infer_sam.py` directly.
- If `--weights` is omitted, MedSAM3 keeps its current auto-detection behavior from the YAML config.
- NIfTI extraction requires `nibabel` to be available in the environment.
- The Streamlit app is meant for interactive testing, not for batch experimentation at scale.
