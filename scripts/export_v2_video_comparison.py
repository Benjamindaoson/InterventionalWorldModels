"""Export aligned ground-truth/reconstruction/teacher/free RGB as a GIF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-npz", type=Path, required=True)
    parser.add_argument("--branch", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve existing video")
    arrays = np.load(args.scene_npz)
    names = ("ground_truth", "reconstructed", "teacher_forced", "free_running")
    shapes = [arrays[name].shape for name in names]
    if len(set(shapes)) != 1 or len(shapes[0]) != 5:
        raise ValueError("four RGB videos must have the same [branch,time,H,W,3] shape")
    branches, frames, height, width, channels = shapes[0]
    if not 0 <= args.branch < branches or channels != 3:
        raise ValueError("invalid branch or RGB shape")
    scale = 4
    strip_height = 24
    images = []
    for t in range(frames):
        canvas = Image.new("RGB", (len(names) * width * scale, height * scale + strip_height), "white")
        draw = ImageDraw.Draw(canvas)
        for index, name in enumerate(names):
            frame = Image.fromarray(arrays[name][args.branch, t].astype(np.uint8))
            canvas.paste(frame.resize((width * scale, height * scale), Image.Resampling.NEAREST),
                         (index * width * scale, strip_height))
            draw.text((index * width * scale + 3, 5), name.replace("_", " "), fill="black")
        draw.text((canvas.width - 44, 5), f"t={t}", fill="black")
        images.append(canvas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(args.output, save_all=True, append_images=images[1:],
                   duration=80, loop=0, optimize=False)
    print(json.dumps({"output": str(args.output), "branch": args.branch,
                      "frames": frames, "aligned_source": str(args.scene_npz)}))


if __name__ == "__main__":
    main()
