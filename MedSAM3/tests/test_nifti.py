from pathlib import Path

import numpy as np
import pytest

from src.medsam3_pipeline.config import SliceExtractionConfig
from src.medsam3_pipeline.nifti import extract_slices


nibabel = pytest.importorskip("nibabel")


def test_extract_slices_writes_pngs_and_mapping(tmp_path: Path) -> None:
    volume = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
    image = nibabel.Nifti1Image(volume, affine=np.eye(4))
    volume_path = tmp_path / "case001.nii.gz"
    nibabel.save(image, volume_path)

    config = SliceExtractionConfig.from_values(tmp_path / "outputs", axis=2, step=1, max_slices=2, run_name="demo")
    records = extract_slices([volume_path], config)

    assert len(records) == 2
    assert Path(records[0]["output_path"]).exists()
    assert (config.run_dir / "slice_map.json").exists()
    assert (config.run_dir / "slice_map.csv").exists()
