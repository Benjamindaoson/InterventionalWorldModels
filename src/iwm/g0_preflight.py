"""Read-only G0 checks for a pinned upstream LeWorldModel checkout.

This module never launches training, downloads data, or creates checkpoints.
Passing the static checks is not evidence of model reproduction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit


OFFICIAL_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
OFFICIAL_REMOTE = "https://github.com/lucas-maes/le-wm"
OFFICIAL_DATA_REVISION = "655cd446b9929369d7d406001da85c15d1457850"


def _git(upstream: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(upstream), *args],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_sha256(path: Path) -> tuple[str, int]:
    """Hash a file or a sorted directory tree without following symlinks."""
    if path.is_symlink():
        raise OSError("dataset symlink requires separate source audit")
    if path.is_file():
        return _sha256(path), path.stat().st_size
    digest = hashlib.sha256()
    total_size = 0
    files = sorted(item for item in path.rglob("*") if item.is_file() or item.is_symlink())
    if not files:
        raise OSError("dataset directory has no files")
    for item in files:
        if item.is_symlink():
            raise OSError("dataset symlink requires separate source audit")
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        digest.update(relative.encode("utf-8") + b"\0" + str(size).encode() + b"\0")
        digest.update(bytes.fromhex(_sha256(item)))
        total_size += size
    return digest.hexdigest(), total_size


def _safe_remote(remote: str | None) -> str | None:
    if not remote:
        return None
    try:
        parsed = urlsplit(remote)
    except ValueError:
        return "<invalid remote>"
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        try:
            port = parsed.port
        except ValueError:
            return "<invalid remote>"
        host = parsed.hostname + (f":{port}" if port else "")
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    if "@" in remote:
        return "<redacted>@" + remote.rsplit("@", 1)[1].split("?", 1)[0]
    return "<unrecognized remote>"


def probe_gpu() -> dict | None:
    """Return GPU and PyTorch CUDA evidence, or None if unusable."""
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        torch_info = subprocess.check_output(
            [sys.executable, "-c", "import json,torch; print(json.dumps({'version':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available()}))"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        ).strip()
        parsed = json.loads(torch_info)
        if not smi or not parsed["available"]:
            return None
        return {"nvidia_smi": smi.splitlines(), "pytorch": parsed}
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, KeyError):
        return None


def inspect(
    upstream: Path,
    data_root: Path,
    *,
    mode: str = "local",
    expected_commit: str = OFFICIAL_COMMIT,
    max_gpu_hours: float | None = None,
    hourly_rate_cny: float | None = None,
    min_free_gb: float = 30.0,
    gpu_probe=probe_gpu,
    hash_data: bool = False,
    loader_data_root: Path | None = None,
) -> dict:
    """Inspect prerequisites without changing upstream, data, or run state."""
    if mode not in {"local", "autodl"}:
        raise ValueError("mode must be local or autodl")
    upstream = Path(upstream).resolve()
    data_root = Path(data_root).resolve()
    blockers: list[dict[str, str]] = []

    def block(code: str, detail: str) -> None:
        blockers.append({"code": code, "detail": detail})

    commit = _git(upstream, "rev-parse", "HEAD")
    remote = _safe_remote(_git(upstream, "remote", "get-url", "origin"))
    dirty = _git(upstream, "status", "--porcelain")
    if commit != expected_commit:
        block("upstream_commit", f"expected {expected_commit}; observed {commit or 'N/A'}")
    normalized_remote = (remote or "").removesuffix(".git").rstrip("/").lower()
    if normalized_remote != OFFICIAL_REMOTE.lower():
        block("upstream_remote", f"expected {OFFICIAL_REMOTE}; observed {remote or 'N/A'}")
    if dirty is None:
        block("upstream_git_unavailable", "cannot inspect upstream working tree")
    elif dirty:
        block("upstream_dirty", "upstream checkout contains uncommitted files")

    required_files = {
        "train": upstream / "train.py",
        "jepa": upstream / "jepa.py",
        "module": upstream / "module.py",
        "license": upstream / "LICENSE",
        "train_config": upstream / "config/train/lewm.yaml",
        "launcher_config": upstream / "config/train/launcher/local.yaml",
        "data_config": upstream / "config/train/data/pusht.yaml",
    }
    contents = {name: _read(path) for name, path in required_files.items()}
    file_hashes = {}
    for name, path in required_files.items():
        if contents[name] is None:
            block("upstream_file_missing", f"{name}: {path}")
        else:
            try:
                file_hashes[name] = _sha256(path)
            except OSError:
                block("file_read_error", f"could not hash upstream {name}")

    license_text = contents["license"] or ""
    if license_text and ("MIT License" not in license_text or "Lucas Maes" not in license_text):
        block("license_mismatch", "upstream MIT attribution not verified")
    symbols = {"jepa": ("class JEPA",), "module": ("class SIGReg", "class ARPredictor", "class Embedder")}
    for name, expected_symbols in symbols.items():
        if contents[name] is not None:
            for symbol in expected_symbols:
                if symbol not in contents[name]:
                    block("upstream_symbol_missing", f"{name}: {symbol}")

    train_text = contents["train"] or ""
    train_cfg = contents["train_config"] or ""
    launcher_cfg = contents["launcher_config"] or ""
    if "cfg.wandb" in train_text and contents["train_config"] is not None:
        direct = bool(re.search(r"(?m)^wandb\s*:\s*(?:#.*)?$", train_cfg))
        composed = (
            bool(re.search(r"(?m)^\s*-\s*launcher:\s*local\s*$", train_cfg))
            and "# @package _global_" in launcher_cfg
            and bool(re.search(r"(?m)^wandb\s*:\s*(?:#.*)?$", launcher_cfg))
        )
        if not (direct or composed):
            block("wandb_config_missing", "cfg.wandb is not present in direct or known launcher-composed config")

    data_cfg = contents["data_config"] or ""
    name_match = re.search(r"(?m)^\s*name\s*:\s*([^\s#]+)", data_cfg)
    dataset_name = name_match.group(1).strip("'\"") if name_match else None
    dataset_path = None
    dataset_size_bytes = None
    dataset_sha256 = None
    if not dataset_name:
        block("dataset_name_missing", "Push-T data config does not name a dataset")
    elif not data_root.is_dir():
        block("data_root_missing", str(data_root))
    else:
        candidates = [data_root / dataset_name, data_root / "datasets" / dataset_name]
        dataset_path = next((p for p in candidates if p.exists()), candidates[0])
        try:
            exists = dataset_path.exists()
            empty_file = exists and dataset_path.is_file() and dataset_path.stat().st_size == 0
        except OSError:
            exists = False
            empty_file = False
            block("file_read_error", "could not inspect configured dataset")
        if not exists or empty_file:
            block("dataset_missing", f"configured dataset not found or empty: {dataset_path}")
            stem = dataset_name.removesuffix(".lance").removesuffix(".h5")
            alternatives = [data_root / f"{stem}.h5.zst", data_root / f"{stem}.h5"]
            found = [str(p) for p in alternatives if p.exists()]
            if found:
                block("dataset_format_mismatch", f"found {found}; config requires {dataset_name}")
        else:
            try:
                if dataset_path.is_file():
                    dataset_size_bytes = dataset_path.stat().st_size
                elif not any(dataset_path.iterdir()):
                    block("dataset_empty", f"configured dataset directory is empty: {dataset_path}")
                if hash_data:
                    dataset_sha256, dataset_size_bytes = _dataset_sha256(dataset_path)
                else:
                    block("data_hash_unverified", "run with --hash-data to bind local dataset bytes")
            except OSError:
                block("file_read_error", "could not inspect or hash configured dataset")

    gpu = None
    free_disk_gb = None
    cost_ceiling_cny = None
    loader_root = Path(loader_data_root).resolve() if loader_data_root is not None else (Path(os.environ["LOCAL_DATASET_DIR"]).resolve() if os.environ.get("LOCAL_DATASET_DIR") else None)
    if mode == "autodl":
        if loader_root != data_root:
            block("loader_data_root_mismatch", "LOCAL_DATASET_DIR must resolve to the checked --data-root")
        gpu = gpu_probe()
        if gpu is None:
            block("gpu_unavailable", "nvidia-smi and CUDA-enabled PyTorch must both work")
        if (max_gpu_hours is None or hourly_rate_cny is None
            or not math.isfinite(max_gpu_hours) or not math.isfinite(hourly_rate_cny)
            or max_gpu_hours <= 0 or hourly_rate_cny <= 0
            or not math.isfinite(max_gpu_hours * hourly_rate_cny)):
            block("budget_missing", "supply positive max GPU hours and hourly CNY rate")
        else:
            cost_ceiling_cny = round(max_gpu_hours * hourly_rate_cny, 2)
        if not math.isfinite(min_free_gb) or min_free_gb <= 0:
            block("disk_threshold_invalid", "minimum free disk must be a finite positive GiB value")
        if data_root.exists():
            try:
                free_disk_gb = round(shutil.disk_usage(data_root).free / 1024**3, 2)
                if math.isfinite(min_free_gb) and min_free_gb > 0 and free_disk_gb < min_free_gb:
                    block("disk_low", f"{free_disk_gb} GiB free; require at least {min_free_gb} GiB")
            except OSError:
                block("disk_probe_error", "could not inspect free disk")

    blocker_codes = {item["code"] for item in blockers}
    def check(*codes: str, applicable: bool = True) -> str:
        return "NOT_APPLICABLE" if not applicable else ("BLOCKED" if blocker_codes.intersection(codes) else "PASS")
    checks = {
        "upstream_identity": check("upstream_commit", "upstream_remote", "upstream_git_unavailable", "upstream_dirty"),
        "upstream_files": check("upstream_file_missing", "file_read_error", "upstream_symbol_missing", "license_mismatch"),
        "composed_config": check("wandb_config_missing", "upstream_file_missing"),
        "data_presence": check("data_root_missing", "dataset_name_missing", "dataset_missing", "dataset_format_mismatch", "dataset_empty", "file_read_error"),
        "data_integrity": check("data_hash_unverified", "data_root_missing", "dataset_name_missing", "dataset_missing", "dataset_format_mismatch", "dataset_empty", "file_read_error"),
        "loader_binding": check("loader_data_root_mismatch", applicable=mode == "autodl"),
        "gpu": check("gpu_unavailable", applicable=mode == "autodl"),
        "disk": check("disk_low", "disk_threshold_invalid", "disk_probe_error", applicable=mode == "autodl"),
        "budget": check("budget_missing", applicable=mode == "autodl"),
    }

    return {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED" if blockers else "PASS_STATIC_ONLY",
        "mode": mode,
        "upstream": {"path": str(upstream), "remote": remote, "commit": commit, "expected_commit": expected_commit, "file_sha256": file_hashes},
        "data": {"root": str(data_root), "reference_hf_revision": OFFICIAL_DATA_REVISION, "configured_name": dataset_name, "configured_path": str(dataset_path) if dataset_path else None, "size_bytes": dataset_size_bytes, "sha256": dataset_sha256, "sha256_kind": ("file" if dataset_path.is_file() else "tree-v1") if dataset_sha256 and dataset_path else None, "decode_verified": False, "loader_data_root": str(loader_root) if loader_root else None},
        "gpu": gpu,
        "free_disk_gb": free_disk_gb,
        "cost_ceiling_cny": cost_ceiling_cny,
        "training_executed": False,
        "checkpoint": None,
        "checks": checks,
        "blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only G0 LeWM Push-T preflight; never trains or downloads")
    parser.add_argument("--upstream", type=Path, required=True, help="existing official le-wm Git checkout")
    parser.add_argument("--data-root", type=Path, required=True, help="existing Push-T data root")
    parser.add_argument("--mode", choices=("local", "autodl"), default="local")
    parser.add_argument("--max-gpu-hours", type=float)
    parser.add_argument("--hourly-rate-cny", type=float)
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    parser.add_argument("--hash-data", action="store_true", help="hash configured dataset file or sorted directory tree")
    parser.add_argument("--output", type=Path, help="optional JSON receipt file")
    args = parser.parse_args(argv)
    report = inspect(args.upstream, args.data_root, mode=args.mode, max_gpu_hours=args.max_gpu_hours, hourly_rate_cny=args.hourly_rate_cny, min_free_gb=args.min_free_gb, hash_data=args.hash_data)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"] == "PASS_STATIC_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
