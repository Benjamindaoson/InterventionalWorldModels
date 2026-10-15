import json
import subprocess

import iwm.g0_preflight as preflight
from iwm.g0_preflight import inspect


def fixture_upstream(tmp_path, *, wandb=True, dataset_name="pusht_expert_train.lance", remote="https://github.com/lucas-maes/le-wm.git"):
    upstream = tmp_path / "le-wm"
    (upstream / "config/train/data").mkdir(parents=True)
    (upstream / "config/train").mkdir(parents=True, exist_ok=True)
    (upstream / "config/train/launcher").mkdir(parents=True)
    (upstream / "config/train/data/pusht.yaml").write_text(
        f"dataset:\n  name: {dataset_name}\n", encoding="utf-8"
    )
    (upstream / "config/train/lewm.yaml").write_text(
        "defaults:\n  - _self_\n  - launcher: local\ntrainer:\n  max_epochs: 1\n",
        encoding="utf-8",
    )
    (upstream / "config/train/launcher/local.yaml").write_text(
        "# @package _global_\nwandb:\n  enabled: false\n" if wandb else "# @package _global_\n",
        encoding="utf-8",
    )
    (upstream / "train.py").write_text("print(cfg.wandb.enabled)\n", encoding="utf-8")
    (upstream / "jepa.py").write_text("class JEPA: pass\n", encoding="utf-8")
    (upstream / "module.py").write_text(
        "class SIGReg: pass\nclass ARPredictor: pass\nclass Embedder: pass\n",
        encoding="utf-8",
    )
    (upstream / "LICENSE").write_text(
        "MIT License\nCopyright (c) 2026 Lucas Maes\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", str(upstream)], check=True)
    subprocess.run(
        ["git", "-C", str(upstream), "remote", "add", "origin", remote],
        check=True,
    )
    subprocess.run(["git", "-C", str(upstream), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(upstream), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
        check=True,
    )
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    return upstream, commit


def test_wrong_upstream_revision_blocks(tmp_path):
    upstream, commit = fixture_upstream(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "pusht_expert_train.lance").write_bytes(b"test data")
    report = inspect(upstream, data, expected_commit="0" * 40)
    assert report["status"] == "BLOCKED"
    assert any(item["code"] == "upstream_commit" for item in report["blockers"])
    assert report["upstream"]["commit"] == commit


def test_config_data_mismatch_and_missing_wandb_block(tmp_path):
    upstream, commit = fixture_upstream(tmp_path, wandb=False)
    data = tmp_path / "data"
    data.mkdir()
    (data / "pusht_expert_train.h5.zst").write_bytes(b"archive")
    report = inspect(upstream, data, expected_commit=commit)
    codes = {item["code"] for item in report["blockers"]}
    assert "dataset_missing" in codes
    assert "dataset_format_mismatch" in codes
    assert "wandb_config_missing" in codes
    assert report["status"] == "BLOCKED"


def test_local_static_pass_is_not_training_claim(tmp_path):
    upstream, commit = fixture_upstream(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "pusht_expert_train.lance").write_bytes(b"test data")
    report = inspect(upstream, data, expected_commit=commit, hash_data=True)
    assert report["status"] == "PASS_STATIC_ONLY"
    assert report["blockers"] == []
    assert report["training_executed"] is False
    assert report["checkpoint"] is None
    assert len(report["data"]["sha256"]) == 64


def test_autodl_requires_gpu_and_budget(tmp_path):
    upstream, commit = fixture_upstream(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "pusht_expert_train.lance").write_bytes(b"test data")
    report = inspect(upstream, data, mode="autodl", expected_commit=commit, gpu_probe=lambda: None)
    codes = {item["code"] for item in report["blockers"]}
    assert {"gpu_unavailable", "budget_missing"} <= codes
    assert report["status"] == "BLOCKED"
    json.dumps(report)


def test_composed_launcher_supplies_wandb(tmp_path):
    upstream, commit = fixture_upstream(tmp_path, wandb=True)
    data = tmp_path / "data"
    data.mkdir()
    (data / "pusht_expert_train.lance").write_bytes(b"test data")
    report = inspect(upstream, data, expected_commit=commit)
    assert "wandb_config_missing" not in {item["code"] for item in report["blockers"]}


def test_remote_credentials_never_leave_report(tmp_path):
    upstream, commit = fixture_upstream(tmp_path, remote="https://user:secret-token@github.com/lucas-maes/le-wm.git")
    data = tmp_path / "data"
    data.mkdir()
    (data / "pusht_expert_train.lance").write_bytes(b"test data")
    report = inspect(upstream, data, expected_commit=commit)
    payload = json.dumps(report)
    assert "secret-token" not in payload
    assert "user:" not in payload
    assert report["upstream"]["remote"] == "https://github.com/lucas-maes/le-wm.git"


def test_nonfinite_budget_and_negative_disk_threshold_block(tmp_path):
    upstream, commit = fixture_upstream(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "pusht_expert_train.lance").write_bytes(b"test data")
    report = inspect(
        upstream, data, mode="autodl", expected_commit=commit,
        max_gpu_hours=float("nan"), hourly_rate_cny=float("inf"),
        min_free_gb=-1, gpu_probe=lambda: {"name": "fixture GPU"},
    )
    codes = {item["code"] for item in report["blockers"]}
    assert "budget_missing" in codes
    assert "disk_threshold_invalid" in codes
    assert report["cost_ceiling_cny"] is None
    json.dumps(report, allow_nan=False)


def test_autodl_data_root_must_match_loader(tmp_path):
    upstream, commit = fixture_upstream(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "pusht_expert_train.lance").write_bytes(b"test data")
    kwargs = dict(
        mode="autodl", expected_commit=commit, hash_data=True,
        max_gpu_hours=2, hourly_rate_cny=3,
        min_free_gb=0.01,
        gpu_probe=lambda: {"name": "fixture GPU"},
    )
    mismatched = inspect(upstream, data, loader_data_root=tmp_path / "other", **kwargs)
    assert "loader_data_root_mismatch" in {item["code"] for item in mismatched["blockers"]}
    matched = inspect(upstream, data, loader_data_root=data, **kwargs)
    assert matched["status"] == "PASS_STATIC_ONLY"
    assert matched["cost_ceiling_cny"] == 6


def test_hash_read_error_produces_blocked_receipt(tmp_path, monkeypatch):
    upstream, commit = fixture_upstream(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "pusht_expert_train.lance").write_bytes(b"test data")
    monkeypatch.setattr(preflight, "_sha256", lambda _: (_ for _ in ()).throw(PermissionError("denied")))
    report = inspect(upstream, data, expected_commit=commit, hash_data=True)
    assert report["status"] == "BLOCKED"
    assert "file_read_error" in {item["code"] for item in report["blockers"]}
    json.dumps(report)
