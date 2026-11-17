"""Leaf disease classifier (YOLOv8-cls), Grad-CAM explanations, and the combined diagnosis.

Milestone 3 pipeline (used by both app.py and the CLI):
1. Classifier  -> which plant/condition (whole image, top-1 + top-3 probabilities)
2. Detector    -> where the leaf is (box), if detector weights are available
3. Colour check inside the box -> infected area %, severity, plant health
4. Grad-CAM    -> heatmap of the image regions that drove the classifier's decision
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from model.predict import (
    REPO_ROOT,
    class_info,
    estimate_lesion_pct,
    recommend_pesticide,
    severity_level,
)

DEFAULT_CLS_WEIGHTS_REL = Path("runs") / "classify" / "train" / "weights" / "best.pt"


def find_classifier_weights(root: Path = REPO_ROOT) -> Optional[Path]:
    """Resolve classifier weights: PLANTDOC_CLS_WEIGHTS env var, then train_classifier.py output."""
    env_raw = os.environ.get("PLANTDOC_CLS_WEIGHTS", "").strip()
    if env_raw:
        env_path = Path(env_raw).expanduser()
        if not env_path.is_absolute():
            env_path = (root / env_path).resolve()
        if env_path.is_file():
            return env_path
    default = root / DEFAULT_CLS_WEIGHTS_REL
    return default.resolve() if default.is_file() else None


def grad_cam(classifier: Any, image: Image.Image, class_id: int) -> np.ndarray:
    """
    Grad-CAM for a YOLOv8 classifier.

    Takes the feature maps of the last backbone block (C2f, layer 8), weights each
    channel by the average gradient of the class logit, sums them and keeps the
    positive part. Returns a 0-1 heatmap for the centre square the model sees
    (Ultralytics resizes the short side and centre-crops).
    """
    import torch
    from ultralytics.data.augment import classify_transforms

    net = classifier.model
    net.eval()
    imgsz = int(classifier.overrides.get("imgsz", 224) or 224)
    x = classify_transforms(imgsz)(image.convert("RGB")).unsqueeze(0)

    store: dict[str, torch.Tensor] = {}

    def keep_activation(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        output.retain_grad()
        store["act"] = output

    handle = net.model[-2].register_forward_hook(keep_activation)
    try:
        with torch.enable_grad():
            x.requires_grad_(True)
            out = net(x)
            logits = out[1] if isinstance(out, (tuple, list)) else out
            logits[0, class_id].backward()
    finally:
        handle.remove()

    act = store["act"][0].detach()
    grads = store["act"].grad[0].detach()
    weights = grads.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * act).sum(0)).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def overlay_heatmap(image: Image.Image, cam: np.ndarray, size: int = 448) -> Image.Image:
    """Blend a Grad-CAM heatmap over the centre square of the image (the part the classifier sees)."""
    import cv2  # installed with ultralytics

    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    square = cv2.resize(rgb[top : top + side, left : left + side], (size, size))
    cam_big = cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.uint8(255 * np.clip(cam_big, 0, 1)), cv2.COLORMAP_JET)[:, :, ::-1]
    blended = (0.55 * square + 0.45 * colored).astype(np.uint8)
    return Image.fromarray(blended)


def draw_box(image: Image.Image, bbox: tuple[float, float, float, float], label: str) -> Image.Image:
    """Draw the leaf box with the classifier's label (so box and diagnosis always agree)."""
    import cv2

    canvas = np.asarray(image.convert("RGB")).copy()
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    thick = max(2, round(min(canvas.shape[:2]) / 250))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (16, 185, 129), thick)
    scale = max(0.5, min(canvas.shape[:2]) / 900)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    # Keep the label a little inside the image so rounded corners in the UI never clip it.
    margin = max(4, round(min(canvas.shape[:2]) * 0.03))
    x_text = min(max(x1, margin), max(margin, canvas.shape[1] - tw - margin))
    y_text = max(th + base + margin, y1)
    cv2.rectangle(canvas, (x_text - 3, y_text - th - base - 3), (x_text + tw + 3, y_text + 1), (16, 185, 129), -1)
    cv2.putText(canvas, label, (x_text, y_text - base), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)
    return Image.fromarray(canvas)


def diagnose(
    image: Image.Image,
    classifier: Any,
    detector: Any = None,
    conf: float = 0.25,
    with_heatmap: bool = True,
) -> dict[str, Any]:
    """
    Full diagnosis of one leaf image. Returns a dict with:
    status ("detected" | "no_detection"), plant, disease, isHealthy, confidence, top3,
    severity, severityLevel, plantHealth, suggestion, bbox, annotated (PIL), heatmap (PIL or None).
    """
    image = image.convert("RGB")
    result = classifier.predict(source=image, verbose=False)[0]
    probs = result.probs
    names = classifier.names
    top_ids = [int(i) for i in probs.top5[:3]]
    top_conf = [float(c) * 100.0 for c in probs.top5conf[:3]]
    top3 = []
    for cid, pct in zip(top_ids, top_conf):
        plant, disease = class_info(str(names[cid]))
        top3.append({"plant": plant, "disease": disease, "confidence": round(pct, 2)})

    best_id, best_pct = top_ids[0], top_conf[0]
    plant, disease = class_info(str(names[best_id]))
    if best_pct < conf * 100.0:
        return {
            "status": "no_detection",
            "message": (
                f"Not confident enough (best guess {best_pct:.0f}% < {conf:.0%}). "
                "Try a closer, well-lit photo of a single leaf."
            ),
            "top3": top3,
            "annotated": image,
            "heatmap": None,
        }

    # Where is the leaf? Use the detector's most confident box; otherwise the whole image.
    w, h = image.size
    bbox: tuple[float, float, float, float] = (0.0, 0.0, float(w), float(h))
    if detector is not None:
        det = detector.predict(source=image, conf=0.10, verbose=False)[0]
        if det.boxes is not None and len(det.boxes) > 0:
            i = int(det.boxes.conf.argmax())
            bbox = tuple(float(v) for v in det.boxes.xyxy[i].tolist())  # type: ignore[assignment]

    is_healthy = disease == "Healthy"
    if is_healthy:
        infected = 0.0
    else:
        bgr = np.asarray(image)[:, :, ::-1]
        lesion = estimate_lesion_pct(bgr, bbox)
        infected = lesion if lesion is not None else 0.0
    severity = "None" if is_healthy else severity_level(infected)

    heatmap = None
    if with_heatmap:
        try:
            heatmap = overlay_heatmap(image, grad_cam(classifier, image, best_id))
        except Exception:
            heatmap = None

    return {
        "status": "detected",
        "plant": plant,
        "disease": disease,
        "isHealthy": is_healthy,
        "confidence": round(best_pct, 2),
        "top3": top3,
        "severity": round(infected, 2),
        "severityLevel": severity,
        "plantHealth": round(100.0 - infected, 2),
        "suggestion": recommend_pesticide(disease),
        "bbox": [round(v, 1) for v in bbox],
        "annotated": draw_box(image, bbox, f"{plant} - {disease} {best_pct:.0f}%"),
        "heatmap": heatmap,
    }
