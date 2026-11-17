"""
prepare_classification_dataset.py

Convert the prepared YOLO detection dataset into the folder layout that YOLO
classification models (yolov8*-cls) expect, keeping the exact same
train/val/test split so detector and classifier results are comparable:

    <output>/train/<class name>/*.jpg
    <output>/val/<class name>/*.jpg
    <output>/test/<class name>/*.jpg

The class of each image is read from its YOLO label file (first number).
Every class gets a folder in every split (empty if needed) so class indices
are identical across splits. Files are hard-linked when possible (no extra
disk space), otherwise copied.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a classification dataset from the YOLO detection split.")
    parser.add_argument("--data", default="data.yaml", help="YOLO data.yaml with path + class names.")
    parser.add_argument("--output-dir", required=True, help="Where to create the classification dataset.")
    return parser.parse_args()


def short_name(img: Path, max_stem: int = 60) -> str:
    """Keep file names short enough for Windows' 260-character path limit (stays unique via a hash)."""
    if len(img.stem) <= max_stem:
        return img.name
    digest = hashlib.md5(img.stem.encode("utf-8")).hexdigest()[:8]
    return f"{img.stem[:max_stem]}_{digest}{img.suffix}"


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    data = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    det_root = Path(data["path"])
    names: dict[int, str] = {int(k): v for k, v in data["names"].items()}
    out_root = Path(args.output_dir).resolve()

    for split in SPLITS:
        for name in names.values():
            (out_root / split / name).mkdir(parents=True, exist_ok=True)

        counts: Counter[str] = Counter()
        img_dir = det_root / "images" / split
        lbl_dir = det_root / "labels" / split
        for img in sorted(img_dir.iterdir()):
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.is_file():
                continue
            first = lbl.read_text(encoding="utf-8").split()
            if not first:
                continue
            name = names[int(first[0])]
            link_or_copy(img, out_root / split / name / short_name(img))
            counts[name] += 1

        empty = [n for n in names.values() if counts[n] == 0]
        print(f"{split}: {sum(counts.values())} images, {len(names)} classes"
              + (f" (empty: {', '.join(empty)})" if empty else ""))

    print(f"Classification dataset ready: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
