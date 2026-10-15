"""Bounded G0 Lance probes. Never resolves a Hugging Face repository name.

Run in the isolated LeWM environment. The fixture checks the stable-worldmodel
loader contract; the remote mode reads two scalar rows and one image with Lance.
Neither mode trains a model or proves HDF5/Lance equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import urllib.request
from pathlib import Path


REPO = "galilai-group/lewm-pusht"
REVISION = "ea321e392348e3c65a18ab0d685f00e57be2c3e0"
REMOTE_URI = f"hf://datasets/{REPO}/pusht_expert_train.lance"
COLS = ("episode_idx", "step_idx", "action", "pixels", "proprio", "state")


def _head() -> str:
    with urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/{REPO}", timeout=20
    ) as response:
        return json.load(response)["sha"]


def _jpeg() -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (224, 224), (31, 47, 59)).save(out, format="JPEG")
    return out.getvalue()


def fixture_probe() -> dict:
    import lancedb
    import pyarrow as pa
    import stable_worldmodel as swm

    n = 50
    img = _jpeg()
    table = pa.table(
        {
            "episode_idx": pa.array([i // 25 for i in range(n)], type=pa.int32()),
            "step_idx": pa.array([i % 25 for i in range(n)], type=pa.int32()),
            "action": pa.array([[0.1, -0.2]] * n, type=pa.list_(pa.float32(), 2)),
            "pixels": pa.array([img] * n, type=pa.binary()),
            "proprio": pa.array([[1.0, 2.0, 3.0, 4.0]] * n, type=pa.list_(pa.float32(), 4)),
            "state": pa.array([[0.0] * 7] * n, type=pa.list_(pa.float32(), 7)),
        }
    )
    with tempfile.TemporaryDirectory(prefix="iwm-g0-lance-") as temp:
        lancedb.connect(temp).create_table("pusht_expert_train", data=table)
        ds = swm.data.load_dataset(
            str(Path(temp) / "pusht_expert_train.lance"),
            num_steps=4,
            frameskip=5,
            keys_to_load=["pixels", "action", "proprio", "state"],
            keys_to_cache=["action", "proprio", "state"],
        )
        sample = ds[0]
        shapes = {
            key: list(value.shape)
            for key, value in sample.items()
            if hasattr(value, "shape")
        }
        expected = {
            "pixels": [4, 3, 224, 224],
            "action": [4, 10],
            "proprio": [4, 4],
            "state": [4, 7],
        }
        if len(ds) == 0 or any(shapes.get(key) != shape for key, shape in expected.items()):
            raise RuntimeError(f"Fixture Loader contract mismatch: {shapes}")
        return {
            "status": "PASS_FIXTURE_ONLY",
            "rows": n,
            "windows": len(ds),
            "shapes": shapes,
        }


def remote_probe() -> dict:
    import lance
    from PIL import Image

    before = _head()
    if before != REVISION:
        raise RuntimeError(f"HF HEAD changed before read: {before}")
    ds = lance.dataset(REMOTE_URI)
    if tuple(ds.schema.names) != COLS:
        raise RuntimeError(f"Unexpected columns: {ds.schema.names}")
    scalar = ds.take(
        [0, 1], columns=["episode_idx", "step_idx", "action", "proprio", "state"]
    ).to_pylist()
    payload = ds.take([0], columns=["pixels"]).column("pixels")[0].as_py()
    image = Image.open(io.BytesIO(payload))
    image.load()
    after = _head()
    if after != REVISION:
        raise RuntimeError(f"HF HEAD changed after read: {after}")
    if image.size != (224, 224) or image.mode != "RGB":
        raise RuntimeError(f"Unexpected image metadata: {image.size}, {image.mode}")
    return {
        "status": "PASS_BOUNDED_LANCE_READ",
        "revision_before": before,
        "revision_after": after,
        "uri": REMOTE_URI,
        "row_count": ds.count_rows(),
        "schema": str(ds.schema),
        "scalar_rows": scalar,
        "image_0": {
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "format": image.format,
            "size": list(image.size),
            "mode": image.mode,
        },
        "swm_loader_verified": False,
        "hdf5_equivalence_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fixture", "remote"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fixture_probe() if args.mode == "fixture" else remote_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": result["status"], "output": str(args.output)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
