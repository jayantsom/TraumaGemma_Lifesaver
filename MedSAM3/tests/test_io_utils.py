from src.medsam3_pipeline.io_utils import case_id_from_path, discover_images


def test_discover_images_filters_supported_suffixes(tmp_path) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("ignore", encoding="utf-8")

    paths = discover_images(tmp_path)

    assert [path.name for path in paths] == ["a.png", "b.jpg"]


def test_case_id_handles_nii_gz() -> None:
    assert case_id_from_path("example volume.nii.gz") == "example_volume"
