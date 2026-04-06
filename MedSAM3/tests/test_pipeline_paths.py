from pathlib import Path

from src.medsam3_pipeline.config import InferenceConfig
from src.medsam3_pipeline.pipeline import BatchInferencePipeline


def test_artifact_paths_group_by_case_and_prompt(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("output: {}\n", encoding="utf-8")
    config = InferenceConfig.from_values(
        config_path=config_path,
        output_dir=tmp_path / "outputs",
        prompts=["left kidney"],
        run_name="demo",
    )
    pipeline = BatchInferencePipeline(config)

    paths = pipeline._artifact_paths(Path("Case 01.png"), "left kidney")

    assert paths.case_dir.name == "Case_01"
    assert paths.case_dir.parent == config.output_dir
    assert paths.prompt_dir.name == "left_kidney"
    assert paths.metadata_path.name == "metadata.json"
    pipeline.close()
