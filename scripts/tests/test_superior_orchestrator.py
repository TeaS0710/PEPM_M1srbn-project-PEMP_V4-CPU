import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.superior.superior_orchestrator import (
    AnalysisHooksConfig,
    ExpConfig,
    SchedulerConfig,
    DEFAULT_WEIGHTS,
    generate_run_plan,
    load_exp_config,
    read_runs_tsv,
    run_analysis_hooks,
    write_runs_tsv,
    plt,
)


def make_minimal_config(tmp_path: Path) -> Path:
    cfg = {
        "exp_id": "unit_exp",
        "description": "Minimal exp for tests",
        "base": {
            "profile": "demo_profile",
            "stage": "pipeline",
            "fixed": {"CORPUS_ID": "web1"},
            "overrides": {"ideology.view": "global"},
        },
        "axes": [
            {
                "name": "dataset",
                "type": "choice",
                "values": [
                    {"label": "a", "overrides": {"data.corpus_ids": ["web1"]}},
                    {"label": "b", "overrides": {"data.corpus_ids": ["web2"]}},
                ],
            }
        ],
        "grid": {"mode": "cartesian"},
        "run": {"repeats": 1, "seed_strategy": "fixed", "base_seed": 7},
        "scheduler": {"parallel": 1, "max_weight": 2, "resource_classes": {"sklearn": "light"}},
    }
    cfg_path = tmp_path / "exp.yml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_path


def _dummy_exp_config(hooks: list[dict]) -> ExpConfig:
    return ExpConfig(
        exp_id="unit_exp",
        description="",
        base_profile="demo_profile",
        base_stage="pipeline",
        base_make_vars={},
        base_overrides={},
        axes=[],
        grid_mode="cartesian",
        repeats=1,
        seed_strategy="fixed",
        base_seed=0,
        scheduler=SchedulerConfig(
            parallel=1,
            max_ram_gb=None,
            resource_classes={},
            weights=DEFAULT_WEIGHTS,
            max_weight=4,
        ),
        analysis_hooks=AnalysisHooksConfig(after_experiment=hooks),
    )


def test_load_exp_config(tmp_path: Path):
    cfg_path = make_minimal_config(tmp_path)
    exp_cfg = load_exp_config(str(cfg_path))

    assert exp_cfg.exp_id == "unit_exp"
    assert exp_cfg.base_profile == "demo_profile"
    assert exp_cfg.base_make_vars["CORPUS_ID"] == "web1"
    assert exp_cfg.axes[0].name == "dataset"
    assert exp_cfg.scheduler.parallel == 1


def test_generate_run_plan(tmp_path: Path):
    cfg_path = make_minimal_config(tmp_path)
    exp_cfg = load_exp_config(str(cfg_path))
    plan = generate_run_plan(exp_cfg)

    assert len(plan) == 2  # two axis values x 1 repeat
    assert plan[0].run_id.endswith("000000")
    assert plan[1].axis_values["dataset"] == "b"
    assert plan[0].make_vars["SEED"] == "7"


def test_load_exp_config_real_file():
    cfg_path = Path("configs/superior/exp_ideo_balancing_sweep.yml")
    exp_cfg = load_exp_config(str(cfg_path))

    assert exp_cfg.exp_id == "ideo_balancing_sweep"
    assert exp_cfg.scheduler.parallel == 1
    plan = generate_run_plan(exp_cfg)
    assert len(plan) == 8


def test_dry_run_writes_plan(tmp_path: Path):
    cfg_path = make_minimal_config(tmp_path)
    exp_cfg = load_exp_config(str(cfg_path))
    exp_dir = Path("superior") / exp_cfg.exp_id
    if exp_dir.exists():
        for child in exp_dir.glob("**/*"):
            if child.is_file():
                child.unlink()
        for child in sorted(exp_dir.glob("**/*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        exp_dir.rmdir()

    proc = subprocess.run(
        [
            "python",
            "-m",
            "scripts.superior.superior_orchestrator",
            "--exp-config",
            str(cfg_path),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan_path = exp_dir / "plan.tsv"
    assert plan_path.exists()
    content = plan_path.read_text(encoding="utf-8")
    assert "run_id" in content.splitlines()[0]
    assert f"Generated {len(generate_run_plan(exp_cfg))} runs" in proc.stdout


def test_write_and_read_runs_tsv(tmp_path: Path):
    path = tmp_path / "runs.tsv"
    rows = {
        "run_a": {"run_id": "run_a", "exp_id": "exp", "profile": "p", "stage": "s"},
        "run_b": {"run_id": "run_b", "exp_id": "exp", "profile": "p", "stage": "s"},
    }
    write_runs_tsv(rows, path)
    loaded = read_runs_tsv(path)

    assert set(loaded.keys()) == {"run_a", "run_b"}
    assert loaded["run_a"]["run_id"] == "run_a"


def test_run_analysis_hooks(tmp_path: Path):
    runs_dir = tmp_path / "superior" / "unit_exp"
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs_tsv_path = runs_dir / "runs.tsv"

    metrics_path = tmp_path / "reports/web1/view1/family1/modelA/metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"accuracy": 0.75, "macro_f1": 0.6}), encoding="utf-8")

    run_row = {
        "run_id": "unit_run_0",
        "exp_id": "unit_exp",
        "profile": "demo_profile",
        "stage": "pipeline",
        "status": "success",
        "return_code": 0,
        "family": "family1",
        "model_id": "modelA",
        "corpus_id": "web1",
        "dataset_id": "web1",
        "view": "view1",
        "train_prop": "",
        "axis_values_json": json.dumps({}),
        "make_vars_json": json.dumps({"TRAIN_PROP": 0.5}),
        "overrides_json": json.dumps({}),
        "metrics_path": str(metrics_path),
        "log_path": str(runs_dir / "logs" / "unit_run_0.log"),
        "started_at": 0,
        "finished_at": 1,
        "duration_s": 1,
    }
    write_runs_tsv({run_row["run_id"]: run_row}, runs_tsv_path)

    hooks = [
        {"type": "report_markdown", "path": runs_dir / "report.md"},
        {"type": "curves", "metrics": ["accuracy"], "x_axis": "TRAIN_PROP", "group_by": ["family"]},
    ]
    exp_cfg = _dummy_exp_config(hooks)

    run_analysis_hooks(exp_cfg, runs_tsv_path)

    metrics_global = runs_dir / "metrics_global.tsv"
    report_md = runs_dir / "report.md"
    plot_path = runs_dir / "plots" / "accuracy__vs__TRAIN_PROP.png"

    assert metrics_global.exists()
    assert report_md.exists()
    if plt is None:
        assert not plot_path.exists()
    else:
        assert plot_path.exists()
