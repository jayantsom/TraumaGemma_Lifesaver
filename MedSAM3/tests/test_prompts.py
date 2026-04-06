from pathlib import Path

from src.medsam3_pipeline.config import DEFAULT_PROMPTS
from src.medsam3_pipeline.prompts import load_prompts


def test_load_prompts_deduplicates_cli_and_file(tmp_path: Path) -> None:
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("liver\nkidney\n# note\nliver\n", encoding="utf-8")

    prompts = load_prompts(["liver", "spleen"], prompts_file)

    assert prompts == ["liver", "spleen", "kidney"]


def test_load_prompts_falls_back_to_defaults() -> None:
    assert load_prompts() == DEFAULT_PROMPTS
