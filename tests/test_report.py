"""Reporting must never hide missing seeds, changed settings, or cost provenance."""
import json
from pathlib import Path

import pytest
import yaml

from homeroom.config import config_snapshot
from homeroom.report import ReportError, aggregate, load_results


def _experiment(tmp_path):
    cfg = {
        "experiment": {"name": "study", "full_name": "study", "seeds": [0, 1, 2],
                       "results_dir": str(tmp_path)},
        "data": {"dataset": "toy", "k_shot": 3},
        "teacher": {"provider": "queue", "model": "teacher"},
        "student": {"kind": "tfidf"}, "budget": {"usd_per_arm": 1},
        "classroom": {}, "bulk": {"match_classroom_spend": True},
        "arms": ["real_fewshot", "classroom", "bulk"],
    }
    out = tmp_path / "study"
    out.mkdir()
    for seed in cfg["experiment"]["seeds"]:
        for arm in cfg["arms"]:
            row = {
                "experiment": "study", "arm": arm, "seed": seed, "k_shot": 3,
                "test": {"accuracy": 0.7 + 0.01*seed + (0.02 if arm == "classroom" else 0),
                         "macro_f1": 0.7, "ece": 0.1},
                "n_train": 12, "config": config_snapshot(cfg),
                "cost": {"spent_usd": 0 if arm == "real_fewshot" else 0.5,
                         "limit_usd": 1, "real_spent_usd": 0.5},
            }
            if arm == "bulk":
                row["spend_cap_usd"] = 0.5
            (out / f"{arm}_seed{seed}.json").write_text(json.dumps(row))
    return cfg, out


def test_report_pairs_seeds_and_labels_queue_estimates(tmp_path):
    cfg, out = _experiment(tmp_path)
    # A stale extra run must never inflate coverage or alter the mean.
    (out / "classroom_seed99.json").write_text("{}")
    rows, hashes = load_results(cfg)
    report = aggregate(cfg, rows, hashes)
    assert len(rows) == len(hashes) == 9
    assert report["cost_basis"] == "estimated"  # real_spent_usd does not make a queue run billed.
    assert report["accuracy_difference_pp"]["mean"] == pytest.approx(2)
    assert report["groups"]["classroom"]["accuracy"]["mean"] == pytest.approx(0.73)
    assert report["groups"]["classroom"]["accuracy"]["std"] == pytest.approx((2/3)**0.5 * 0.01)
    assert [p["seed"] for p in report["pairs"]] == [0, 1, 2]


def test_report_rejects_missing_seed(tmp_path):
    cfg, out = _experiment(tmp_path)
    (out / "bulk_seed2.json").unlink()
    with pytest.raises(ReportError, match="bulk_seed2.json"):
        load_results(cfg)


@pytest.mark.parametrize("change, error", [
    ("teacher", "settings differ"),
    ("identity", "identity"),
    ("nan", "out of range"),
    ("cap", "not matched"),
    ("overspend", "exceeds"),
])
def test_report_rejects_incomparable_or_invalid_results(tmp_path, change, error):
    cfg, out = _experiment(tmp_path)
    path = out / "bulk_seed0.json"
    r = json.loads(path.read_text())
    if change == "teacher":
        r["config"]["teacher"]["model"] = "another-teacher"
    elif change == "identity":
        r["seed"] = 7
    elif change == "nan":
        r["test"]["accuracy"] = float("nan")
    elif change == "cap":
        r["spend_cap_usd"] = 0.9
    else:
        r["cost"]["spent_usd"] = 2
    path.write_text(json.dumps(r))
    with pytest.raises(ReportError, match=error):
        load_results(cfg)


def test_offline_demo_generates_complete_report(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from homeroom.run import main
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    config_path = tmp_path / "configs" / "toy_demo.yaml"
    config_path.write_text((root / "configs" / "toy_demo.yaml").read_text())
    main(["demo"])
    out = tmp_path / "results" / "toy_demo_dryrun"
    r = json.loads((out / "report.json").read_text())
    assert r["cost_basis"] == "simulated" and r["seeds"] == [0, 1, 2]
    assert all(g["accuracy"]["n"] == 3 for g in r["groups"].values())
    assert "Mock outputs" in (out / "report.md").read_text(encoding="utf-8")
    assert (out / "accuracy_vs_spend.png").read_bytes().startswith(b"\x89PNG")
    svg = (out / "accuracy_vs_spend.svg").read_text(encoding="utf-8")
    assert all(line == line.rstrip() for line in svg.splitlines())
    assert svg.endswith("\n") and not svg.endswith("\n\n")
    import xml.etree.ElementTree as ET
    assert ET.fromstring(svg).tag == "{http://www.w3.org/2000/svg}svg"
    # Re-running uses saved results rather than duplicating seed data.
    hashes = r["source_sha256"]
    main(["demo"])
    assert json.loads((out / "report.json").read_text())["source_sha256"] == hashes
    assert (out / "accuracy_vs_spend.svg").read_text(encoding="utf-8") == svg


def test_demo_refuses_a_paid_teacher_before_running(tmp_path):
    from homeroom.run import main
    cfg = {"data": {"dataset": "toy"}, "teacher": {"provider": "openai_compatible"}}
    path = tmp_path / "paid.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="mock teacher"):
        main(["demo", "--config", str(path)])
