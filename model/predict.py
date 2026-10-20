"""Prediction module for YOLOv8 plant disease detection.

Shared by the command-line tool (this file) and the Flask API (app.py):
- resolves which trained weights to use
- maps raw model class labels to (plant, condition)
- derives infected area, severity, plant health and a treatment suggestion
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_REL = Path("runs") / "detect" / "train" / "weights" / "best.pt"

# PlantDoc class label (folder name) -> (plant, condition).
CLASS_INFO: Dict[str, tuple[str, str]] = {
    "Apple Scab Leaf": ("Apple", "Apple Scab"),
    "Apple leaf": ("Apple", "Healthy"),
    "Apple rust leaf": ("Apple", "Cedar Apple Rust"),
    "Bell_pepper leaf": ("Bell Pepper", "Healthy"),
    "Bell_pepper leaf spot": ("Bell Pepper", "Bacterial Leaf Spot"),
    "Blueberry leaf": ("Blueberry", "Healthy"),
    "Cherry leaf": ("Cherry", "Healthy"),
    "Corn Gray leaf spot": ("Corn", "Gray Leaf Spot"),
    "Corn leaf blight": ("Corn", "Northern Leaf Blight"),
    "Corn rust leaf": ("Corn", "Common Rust"),
    "Peach leaf": ("Peach", "Healthy"),
    "Potato leaf early blight": ("Potato", "Early Blight"),
    "Potato leaf late blight": ("Potato", "Late Blight"),
    "Raspberry leaf": ("Raspberry", "Healthy"),
    "Soyabean leaf": ("Soybean", "Healthy"),
    "Squash Powdery mildew leaf": ("Squash", "Powdery Mildew"),
    "Strawberry leaf": ("Strawberry", "Healthy"),
    "Tomato Early blight leaf": ("Tomato", "Early Blight"),
    "Tomato Septoria leaf spot": ("Tomato", "Septoria Leaf Spot"),
    "Tomato leaf": ("Tomato", "Healthy"),
    "Tomato leaf bacterial spot": ("Tomato", "Bacterial Spot"),
    "Tomato leaf late blight": ("Tomato", "Late Blight"),
    "Tomato leaf mosaic virus": ("Tomato", "Mosaic Virus"),
    "Tomato leaf yellow virus": ("Tomato", "Yellow Leaf Curl Virus"),
    "Tomato mold leaf": ("Tomato", "Leaf Mold"),
    "Tomato two spotted spider mites leaf": ("Tomato", "Spider Mites"),
    "grape leaf": ("Grape", "Healthy"),
    "grape leaf black rot": ("Grape", "Black Rot"),
}

# General first-line advice per condition (not a substitute for a local agronomist).
PESTICIDE_RECOMMENDATIONS: Dict[str, str] = {
    "Healthy": "No treatment needed. Keep monitoring weekly and maintain balanced watering and nutrition.",
    "Apple Scab": "Remove fallen infected leaves; spray a fungicide such as Captan or Myclobutanil from green-tip to petal fall.",
    "Cedar Apple Rust": "Remove nearby juniper galls if possible; spray Myclobutanil in spring during wet weather.",
    "Bacterial Leaf Spot": "Use a copper-based bactericide; use disease-free seed and avoid overhead watering.",
    "Bacterial Spot": "Use a copper-based bactericide; use disease-free seed and avoid overhead watering.",
    "Gray Leaf Spot": "Rotate crops and plough in residue; if severe, apply Azoxystrobin or Propiconazole.",
    "Northern Leaf Blight": "Grow resistant hybrids and rotate crops; if severe before tasselling, apply Propiconazole.",
    "Common Rust": "Grow resistant hybrids; if infection is early and heavy, apply a triazole or strobilurin fungicide.",
    "Early Blight": "Use Mancozeb or Chlorothalonil; remove infected lower leaves and rotate crops.",
    "Late Blight": "Use Copper fungicide or Mancozeb; destroy badly infected plants and keep foliage dry.",
    "Powdery Mildew": "Spray sulphur, potassium bicarbonate or Neem oil; improve air circulation between plants.",
    "Septoria Leaf Spot": "Remove infected lower leaves and mulch the soil; spray Chlorothalonil or Copper fungicide.",
    "Mosaic Virus": "No chemical cure. Remove infected plants, disinfect tools and grow resistant varieties.",
    "Yellow Leaf Curl Virus": "Control whiteflies (yellow sticky traps, Neem oil); remove infected plants; grow resistant varieties.",
    "Leaf Mold": "Reduce humidity and improve ventilation; spray Chlorothalonil or Copper fungicide.",
    "Spider Mites": "Spray water on leaf undersides; use insecticidal soap or Neem oil, or a miticide such as Abamectin if heavy.",
    "Black Rot": "Remove mummified fruit and infected leaves; spray Mancozeb or Myclobutanil from bud break.",
}


@dataclass
class DetectionSummary:
    plant: str
    disease: str
    is_healthy: bool
    confidence_pct: float
    severity_pct: float
    severity: str
    health_pct: float
    recommendation: str
    bbox_xyxy: tuple[float, float, float, float]
    infected_area_pct: float


def _pick_best_weights_by_map(root: Path) -> Optional[Path]:
    """
    Choose the best checkpoint by highest validation mAP50-95 if results.csv exists.
    Falls back to mAP50 if mAP50-95 is unavailable.
    """
    best_weight: Optional[Path] = None
    best_score = -1.0
    for csv_path in root.rglob("results.csv"):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df.columns = [c.strip() for c in df.columns]
            metric_col = None
            for c in df.columns:
                if "metrics/mAP50-95(B)" in c:
                    metric_col = c
                    break
            if metric_col is None:
                for c in df.columns:
                    if "metrics/mAP50(B)" in c:
                        metric_col = c
                        break
            if metric_col is None:
                continue
            score = float(df[metric_col].max())
            candidate = csv_path.parent / "weights" / "best.pt"
            if candidate.exists() and score > best_score:
                best_score = score
                best_weight = candidate.resolve()
        except Exception:
            continue
    return best_weight


def find_weights(root: Path = REPO_ROOT) -> Optional[Path]:
    """Resolve trained weights: PLANTDOC_WEIGHTS env var, then train_yolo.py output, then best run by mAP."""
    env_raw = os.environ.get("PLANTDOC_WEIGHTS", "").strip()
    if env_raw:
        env_path = Path(env_raw).expanduser()
        if not env_path.is_absolute():
            env_path = (root / env_path).resolve()
        if env_path.is_file():
            return env_path

    default = root / DEFAULT_WEIGHTS_REL
    if default.is_file():
        return default.resolve()

    runs_dir = root / "runs"
    if runs_dir.exists():
        best = _pick_best_weights_by_map(runs_dir)
        if best is not None:
            return best
        for p in runs_dir.rglob("best.pt"):
            return p.resolve()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 plant disease prediction module.")
    parser.add_argument(
        "--weights",
        type=str,
        default="auto",
        help="Path to trained model weights (default: auto-detect under runs/).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Path to input image. If empty, script asks with input().",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inference_outputs",
        help="Directory to save annotated output image.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image size (default: the size the model was trained at).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable test-time augmentation for potentially better predictions.",
    )
    parser.add_argument("--device", type=str, default="", help='Device: "cpu", "0", etc.')
    return parser.parse_args()


def resolve_image_path(image_arg: str) -> Path:
    """Resolve image from CLI arg or terminal input()."""
    image_input = image_arg.strip()
    if not image_input:
        image_input = input("Enter input image path: ").strip()
    if not image_input:
        raise ValueError("No image path provided.")
    image_path = Path(image_input)
    if not image_path.is_absolute():
        image_path = (Path.cwd() / image_path).resolve()
    return image_path


def load_model(weights_path: Path | str):
    """Load YOLOv8 model and handle import/path errors."""
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Ultralytics import failed. Install with: pip install -r requirements.txt"
        ) from e

    if isinstance(weights_path, str) and weights_path.lower() == "auto":
        found = find_weights()
        if found is None:
            raise FileNotFoundError(
                f"Model weights not found. Train first (python train_yolo.py) so that "
                f"{DEFAULT_WEIGHTS_REL} exists, or pass --weights."
            )
        weights_path = found
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    return YOLO(str(weights_path)), weights_path


def class_info(raw_class_name: str) -> tuple[str, str]:
    """Map a raw class label to (plant, condition)."""
    if raw_class_name in CLASS_INFO:
        return CLASS_INFO[raw_class_name]
    # PlantVillage-style labels, e.g. "Tomato___Early_blight".
    if "___" in raw_class_name:
        plant, condition = raw_class_name.split("___", 1)
        condition = condition.replace("_", " ").strip()
        plant = plant.replace("_", " ").replace(",", "").strip().title()
        return plant, "Healthy" if condition.lower() == "healthy" else condition.title()
    return "Unknown", raw_class_name


def recommend_pesticide(disease_name: str) -> str:
    """Return recommendation for known diseases, else fallback."""
    return PESTICIDE_RECOMMENDATIONS.get(
        disease_name,
        "No specific recommendation available. Consult local agronomist.",
    )


def compute_infected_area_pct(bbox_xyxy: tuple[float, float, float, float], image_w: int, image_h: int) -> float:
    """Compute infected area % using bounding box area / total image area."""
    x1, y1, x2, y2 = bbox_xyxy
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    bbox_area = box_w * box_h
    image_area = float(max(1, image_w * image_h))
    return min(100.0, max(0.0, (bbox_area / image_area) * 100.0))


def estimate_lesion_pct(image_bgr: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> Optional[float]:
    """
    Estimate the diseased share of the leaf inside the box with HSV colour thresholding:
    plant pixels = saturated and not dark (hue < 90 in OpenCV's 0-180 scale);
    lesion pixels = plant pixels that are yellow/brown (hue < 35) rather than green.
    Returns None if the box has too few plant pixels to judge.
    """
    import cv2  # installed with ultralytics

    h_img, w_img = image_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox_xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    plant = (sat >= 40) & (val >= 40) & (hue < 90)
    n_plant = int(plant.sum())
    if n_plant < 100:
        return None
    lesion = plant & (hue < 35)
    return 100.0 * float(lesion.sum()) / n_plant


def severity_level(severity_pct: float) -> str:
    """Bucket the infected share of the leaf into a label."""
    if severity_pct < 10.0:
        return "Low"
    if severity_pct < 30.0:
        return "Medium"
    return "Severe"


def summarize_detections(result: Any, model: Any) -> List[DetectionSummary]:
    """Extract class, confidence, bbox, and derive severity + health + recommendation."""
    summaries: List[DetectionSummary] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return summaries

    img_h, img_w = result.orig_shape
    model_names = getattr(model, "names", {})

    for box in boxes:
        cls_id = int(box.cls[0]) if box.cls is not None else -1
        raw_name = str(model_names.get(cls_id, cls_id))
        plant, disease_name = class_info(raw_name)
        is_healthy = disease_name == "Healthy"

        conf = float(box.conf[0]) if box.conf is not None else 0.0
        confidence_pct = max(0.0, min(100.0, conf * 100.0))

        coords = box.xyxy[0].tolist()
        bbox_xyxy = (float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))
        # Training boxes cover the whole image, so box area alone says little; prefer the
        # colour-based lesion share inside the box and fall back to box area if it can't be judged.
        lesion_pct = estimate_lesion_pct(result.orig_img, bbox_xyxy)
        if lesion_pct is not None:
            infected_area_pct = lesion_pct
        else:
            infected_area_pct = compute_infected_area_pct(bbox_xyxy, img_w, img_h)
        if is_healthy:
            # A healthy-leaf box is not an infected region.
            infected_area_pct = 0.0
            severity_pct = 0.0
            severity = "None"
        else:
            severity_pct = max(0.0, min(100.0, infected_area_pct))
            severity = severity_level(severity_pct)
        health_pct = max(0.0, min(100.0, 100.0 - infected_area_pct))

        summaries.append(
            DetectionSummary(
                plant=plant,
                disease=disease_name,
                is_healthy=is_healthy,
                confidence_pct=confidence_pct,
                severity_pct=severity_pct,
                severity=severity,
                health_pct=health_pct,
                recommendation=recommend_pesticide(disease_name),
                bbox_xyxy=bbox_xyxy,
                infected_area_pct=infected_area_pct,
            )
        )

    return summaries


def save_annotated_image(result: Any, image_path: Path, output_dir: Path) -> Path:
    """Save image with YOLO bounding boxes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_bgr = result.plot()
    if not isinstance(annotated_bgr, np.ndarray):
        raise RuntimeError("Unexpected output type while plotting detections.")
    annotated_rgb = annotated_bgr[:, :, ::-1]
    annotated_image = Image.fromarray(annotated_rgb)
    out_path = output_dir / f"{image_path.stem}_predicted{image_path.suffix}"
    annotated_image.save(out_path)
    return out_path


def print_report(summary: DetectionSummary) -> None:
    """Print final report in required clean format."""
    print(f"Plant: {summary.plant}")
    print(f"Disease: {summary.disease}")
    print(f"Confidence: {summary.confidence_pct:.2f}%")
    print(f"Severity: {summary.severity} ({summary.severity_pct:.2f}%)")
    print(f"Health: {summary.health_pct:.2f}%")
    print(f"Recommendation: {summary.recommendation}")


def run_prediction(
    model: Any,
    image_path: Path,
    output_dir: Path,
    imgsz: Optional[int],
    conf: float,
    augment: bool,
    device: Optional[str],
) -> tuple[List[DetectionSummary], Path]:
    """Run YOLO predict, build summaries, and save annotated image."""
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Invalid image path: {image_path}")

    kwargs: Dict[str, Any] = {
        "source": str(image_path),
        "conf": float(conf),
        "augment": bool(augment),
        "verbose": False,
    }
    if imgsz:
        kwargs["imgsz"] = int(imgsz)
    if device and device.strip():
        kwargs["device"] = device.strip()

    results = model.predict(**kwargs)
    if not results:
        raise RuntimeError("No prediction result returned by model.")

    result = results[0]
    summaries = summarize_detections(result, model)
    out_path = save_annotated_image(result, image_path, output_dir)
    return summaries, out_path


def main() -> int:
    args = parse_args()
    try:
        image_path = resolve_image_path(args.image)
        output_dir = (Path.cwd() / args.output_dir).resolve()
        weights_arg: Path | str = "auto" if args.weights.strip().lower() == "auto" else (Path.cwd() / args.weights).resolve()
        model, used_weights = load_model(weights_arg)

        summaries, output_image = run_prediction(
            model=model,
            image_path=image_path,
            output_dir=output_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            augment=args.augment,
            device=args.device,
        )

        print(f"Using model weights: {used_weights}")
        if not summaries:
            print("No detection found.")
            print(f"Annotated output saved at: {output_image}")
            return 0

        # Report highest-confidence detection first for clean single-result output.
        best = max(summaries, key=lambda s: s.confidence_pct)
        print_report(best)
        print(f"Annotated output saved at: {output_image}")
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
