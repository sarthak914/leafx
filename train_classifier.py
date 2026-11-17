"""
train_classifier.py

Train a YOLOv8 classification model (Ultralytics) on the PlantDoc classification
dataset built by prepare_classification_dataset.py.

Why a classifier: PlantDoc's labels are one class per image (our detection boxes
cover the whole image), so this is really an image-classification task. A
classifier fine-tuned from ImageNet weights learns it more directly than the
detector does.

Outputs (Ultralytics run directory):
- runs/classify/train/weights/best.pt   (best validation top-1 accuracy)
- runs/classify/train/results.csv       (per-epoch loss, top-1, top-5)
- runs/classify/train/confusion_matrix*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 classifier on PlantDoc.")
    parser.add_argument(
        "--data",
        required=True,
        help="Classification dataset root containing train/, val/, test/ class folders.",
    )
    parser.add_argument(
        "--model",
        default="yolov8s-cls.pt",
        help="Pretrained classification model (default: yolov8s-cls.pt, ImageNet weights).",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60).")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size (default: 224).")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--device", default="", help='Device, e.g. "cpu" or "0". Empty = auto.')
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers (default: 2).")
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Stop early if validation accuracy hasn't improved for this many epochs (default: 15).",
    )
    parser.add_argument("--name", default="train", help="Run name under runs/classify/ (default: train).")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training set to use (e.g. 0.1 for a quick speed test).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run from runs/classify/<name>/weights/last.pt (same settings).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        print("Ultralytics is not installed. Install with: pip install -r requirements.txt")
        print(f"Import error: {e}")
        return 1

    repo_root = Path.cwd()
    data_dir = Path(args.data).resolve()
    if not (data_dir / "train").is_dir():
        print(f"ERROR: {data_dir / 'train'} not found. Run prepare_classification_dataset.py first.")
        return 2

    if args.resume:
        last_pt = repo_root / "runs" / "classify" / args.name / "weights" / "last.pt"
        if not last_pt.is_file():
            print(f"ERROR: nothing to resume, {last_pt} not found.")
            return 3
        print(f"Resuming from {last_pt}")
        results = YOLO(str(last_pt)).train(resume=True)
        return report(results, repo_root, args, data_dir)

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_dir),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "patience": int(args.patience),
        "fraction": float(args.fraction),
        # Absolute so newer Ultralytics versions don't nest it under their own runs dir.
        "project": str(repo_root / "runs" / "classify"),
        "name": args.name,
        "exist_ok": True,
        "plots": True,
        "verbose": True,
    }
    if args.device.strip():
        train_kwargs["device"] = args.device.strip()

    print("Starting training with settings:")
    for k in sorted(train_kwargs):
        print(f"- {k}: {train_kwargs[k]}")
    results = model.train(**train_kwargs)
    return report(results, repo_root, args, data_dir)


def report(results, repo_root: Path, args: argparse.Namespace, data_dir: Path) -> int:
    """Print where the weights are and score the best model once on the held-out test split."""
    from ultralytics import YOLO  # type: ignore

    run_dir = Path(str(getattr(results, "save_dir", repo_root / "runs" / "classify" / args.name)))
    best_pt = run_dir / "weights" / "best.pt"
    print("\nTraining complete.")
    print(f"- Best weights: {best_pt.resolve()}{' (missing!)' if not best_pt.exists() else ''}")

    # Final, unbiased score on the held-out test split.
    if best_pt.exists() and (data_dir / "test").is_dir():
        metrics = YOLO(str(best_pt)).val(data=str(data_dir), split="test", imgsz=int(args.imgsz), plots=False)
        print(f"\nTEST top-1 accuracy: {metrics.top1:.4f}   top-5 accuracy: {metrics.top5:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
