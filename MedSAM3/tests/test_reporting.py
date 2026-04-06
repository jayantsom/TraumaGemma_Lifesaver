import json

from src.medsam3_pipeline.reporting import write_csv_report, write_json_report


def test_write_reports(tmp_path) -> None:
    records = [
        {"prompt": "liver", "success": True, "num_detections": 1},
        {"prompt": "kidney", "success": False, "num_detections": 0},
    ]

    json_path = write_json_report(records, tmp_path / "report.json")
    csv_path = write_csv_report(records, tmp_path / "report.csv")

    assert json.loads(json_path.read_text(encoding="utf-8")) == records
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "prompt,success,num_detections" in csv_text
    assert "kidney,False,0" in csv_text
