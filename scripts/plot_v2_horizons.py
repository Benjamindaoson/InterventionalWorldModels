"""Render audited horizon curves directly from frozen per-run evaluation JSON."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ARMS = ("blind_pos", "conditioned_pos", "paired_pos", "blind_vel",
        "conditioned_vel", "rssm_vel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists; preserve prior plots")
    args.output.mkdir(parents=True)
    families = ("iid_test", "physics_ood", "geometry_ood")
    fig, axes = plt.subplots(1, len(families), figsize=(15, 4.4), sharey=True)
    raw_rows = []
    for axis, family in zip(axes, families, strict=True):
        for arm in ARMS:
            curves = []
            for seed in (17, 29, 43):
                path = args.evaluations / f"{arm}-s{seed}-{family}.json"
                evaluation = json.loads(path.read_text(encoding="utf-8"))
                curve = np.asarray(evaluation["drift_mse_by_horizon"], dtype=float)
                curves.append(curve)
                raw_rows.extend({"family": family, "arm": arm, "seed": seed,
                                 "horizon": h, "position_mse_m2": float(value),
                                 "source_file": str(path)}
                                for h, value in enumerate(curve, 1))
            samples = np.stack(curves)
            horizon = np.arange(1, samples.shape[1] + 1)
            axis.plot(horizon, samples.mean(0), label=arm)
        axis.set_title(family.replace("_", " "))
        axis.set_xlabel("Prediction horizon (simulation steps)")
        axis.set_yscale("log")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Free-running xyz MSE (m²), mean of 3 seeds")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "horizon_mse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    with (args.output / "horizon_mse_raw.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_rows[0])
        writer.writeheader()
        writer.writerows(raw_rows)
    print(json.dumps({"curves": len(families) * len(ARMS),
                      "raw_rows": len(raw_rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
