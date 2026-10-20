# Milestone 2 — Viva Notes & Demo Guide

Project: **AgroNiddan (LeafX) — AI Plant Disease Detection from Leaf Images**

---

## 1. Demo script (5 minutes)

**Before the viva:** double-click `run_demo.cmd` (or `python app.py`), wait for the browser to open at
http://127.0.0.1:5000. Test leaves are in `demo_images/` (taken from the held-out test split, never seen in
training; these are ones the model gets right). `demo_images/hard_cases/` has two it gets wrong (tomato late
blight → squash powdery mildew; corn gray leaf spot → corn rust) — use them to discuss limitations.
A phone photo of a real leaf also works, but expect mistakes: test accuracy is ~39 %.

1. **Problem (30 s)** — Home page: farmers lose a large share of their yield to diseases they spot late; experts are scarce.
2. **Data (1 min)** — open `reports/plantdoc_class_distribution.png` and `reports/plantdoc_class_imbalance.png`:
   28 classes, 2,572 field images, imbalanced, 12 bad files found by our integrity check.
3. **Model (1 min)** — open `reports/training/results.png` (loss going down, mAP going up) and
   `reports/training/confusion_matrix_normalized.png`. Say: test mAP@50 = 0.435, top-1 accuracy 39 %, CPU baseline.
4. **Live scan (2 min)** — Scan → Upload Image → Scan Now → Results: annotated box, plant, condition, confidence,
   severity, health bar, treatment. Then show History.
5. **Honesty check** — upload a non-leaf image: the app says *"Nothing detected"* instead of inventing a disease.
6. **CLI (optional)** — `python model/predict.py --image <leaf.jpg>` prints the same report and saves an annotated image.

If the browser fails, the CLI in step 6 is the backup demo.

---

## 2. M1 — Problem domain (3 marks)

- **Domain:** precision agriculture / computer vision for crop health.
- **Problem:** visual diagnosis of leaf diseases is slow, needs experts, and errors lead to wrong or excessive pesticide use.
- **Users:** small farmers, home gardeners, agri-extension workers — people with a phone camera but no plant pathologist.
- **Input → output:** leaf photo → plant + condition (28 classes) + confidence + severity + treatment.
- **Why hard:** real field photos have clutter, multiple leaves, varying light; many diseases look alike (e.g. early vs late blight); classes are small and imbalanced.

## 3. M2 — Existing systems & feasibility (3 marks)

| Existing approach | Limitation | What we do |
|---|---|---|
| Manual inspection by experts / Krishi Vigyan Kendras | slow, not scalable, not available everywhere | instant, on any device with a browser |
| CNN classifiers trained on **PlantVillage** (≈99 % lab accuracy, e.g. Mohanty et al. 2016) | lab images on plain backgrounds; accuracy drops sharply on field photos | we train on **PlantDoc** — real field images |
| Apps like Plantix | closed source, need internet & account | open, local, explainable pipeline |
| Plain classifiers | one label per image, no location | **YOLO detection** gives a box → location + area for severity |

**Feasibility**
- *Technical:* open-source stack (Ultralytics YOLOv8, PyTorch, Flask, React); transfer learning makes small datasets workable; baseline trains on a laptop CPU, full model on free Colab T4.
- *Data:* PlantDoc is public (CC-BY-4.0), 2.5k images, 28 classes.
- *Economic:* zero licence cost; runs on existing hardware.
- *Operational:* web app usable from a phone; ~100–300 ms inference per image on CPU.

## 4. Objectives & methodology (3 marks)

**Objectives**
1. Build a clean, reproducible dataset pipeline (EDA, integrity check, split, resize).
2. Train a YOLOv8 detector for 28 leaf classes and evaluate with mAP, precision, recall.
3. Convert detections into farmer-friendly output: plant, condition, severity, health, treatment.
4. Deliver it through a REST API and a web UI with scan history.

**Methodology (what each script does)**
1. `plantdoc_preprocessing.py` — counts per class, opens every image to catch corrupt files, size stats, plots.
2. `generate_yolo_labels_from_folders.py` — PlantDoc's classification release has no lesion boxes, so each image gets one full-image box (`class 0.5 0.5 1.0 1.0`) labelled with its folder name.
3. `dataset_preparation.py` — pairs images with labels, shuffles (seed 42), splits 70/20/10, converts to RGB JPEG ≤640 px (faster loading; normalized labels stay valid).
4. `train_yolo.py` — transfer learning from COCO-pretrained `yolov8n.pt`; CPU baseline: 15 epochs, 256 px, batch 16, backbone frozen (`--freeze 10`). Ultralytics adds augmentation (mosaic, HSV jitter, flips, scaling).
5. `model/predict.py` — inference + post-processing, shared by CLI and API.
6. `app.py` + `DTI/` — Flask API and React UI.

## 5. Algorithms / techniques (3 marks)

- **YOLOv8 (Ultralytics, 2023)** — one-stage detector: CSPDarknet-style backbone (C2f blocks) → PAN-FPN neck (multi-scale features) → **anchor-free decoupled head** (separate box & class branches).
  - Loss: box = **CIoU + DFL** (distribution focal loss), class = **BCE**. These are the `box_loss`, `dfl_loss`, `cls_loss` columns in `results.csv`.
  - Post-processing: confidence threshold (0.25) + **NMS** (IoU 0.7).
  - *Why YOLO:* real-time on CPU, gives location (needed for severity), easy transfer learning, exports to ONNX/TFLite for mobile later.
  - *Why the `n` (nano) variant now:* 3.2 M params, trains on a CPU laptop. Final: `s` variant on GPU.
- **Transfer learning + freezing** — start from COCO weights; freezing the first 10 layers (backbone) means only neck + head learn → ~2× faster, less overfitting on 1.8k images.
- **Metrics**
  - Precision = TP / (TP + FP); Recall = TP / (TP + FN).
  - **mAP@50** = mean over classes of the area under the precision-recall curve, a box counts as correct if IoU ≥ 0.5.
  - **mAP@50-95** = same, averaged over IoU 0.5…0.95 (stricter).
  - IoU = overlap area / union area of predicted and true box.
- **Severity (classical CV on top of DL)** — inside the detected box, convert to HSV; plant pixels = saturated, not dark, hue < 90; lesion pixels = plant pixels with yellow/brown hue (< 35) instead of green. Infected % = lesion / plant pixels; severity = infected % → Low (<10), Medium (<30), Severe; health = 100 − infected %; healthy class → severity 0, health 100.
- **Rule-based advice** — lookup table from condition to standard first-line treatment (e.g. Early Blight → Mancozeb / Chlorothalonil).

## 6. Design ↔ implementation sync (3 marks)

| Design block | Implemented in |
|---|---|
| Data analysis | `plantdoc_preprocessing.py` → `reports/` |
| Label generation | `generate_yolo_labels_from_folders.py` → `data.yaml`, `classes_mapping.txt` |
| Split + preprocessing | `dataset_preparation.py` |
| Model training | `train_yolo.py` → `runs/detect/train/weights/best.pt` |
| Inference + severity + advice | `model/predict.py` (single source of truth for CLI and API) |
| REST API | `app.py`: `GET /api/health`, `POST /api/predict`, `GET /api/history` |
| Storage | SQLite `predictions.db` (table `predictions`) |
| UI | `DTI/src/layers/ui/pages/` — Landing, Scan, Results, History |

Request flow: React reads the file → downscales to ≤1024 px → base64 → `POST /api/predict` → PIL image → YOLO →
`summarize_detections()` → JSON + annotated JPEG → Results page → row saved in SQLite → visible on History.

---

## 7. Likely viva questions

- **Why PlantDoc and not PlantVillage?** PlantVillage images are lab shots on plain backgrounds; models score ~99 % there but fail on real photos. PlantDoc is real field data — harder, but honest.
- **What are your results?** Test set (256 held-out images): mAP@50 = 0.435, precision 0.39, recall 0.39, top-1 accuracy 39.1 %. Validation mAP@50 = 0.424. mAP@50 rose from 0.066 (epoch 1) to 0.424 (epoch 15) and was still rising.
- **Your mAP is low — why?** Small dataset (~1.8k train images for 28 classes), similar-looking diseases, CPU budget (256 px, 15 epochs, frozen backbone). PlantDoc is known to be hard: in the PlantDoc paper even the best fine-tuned classifier reached only about 70 % accuracy, and models trained on lab images do far worse. Fixes planned: GPU training at 640 px with YOLOv8s, more epochs, class balancing, merging PlantVillage data.
- **Why detection if every box is the whole image?** The pipeline and API are detection-ready; for M2 we used full-image weak labels because the classification release has no lesion boxes. Next step: the PlantDoc *object-detection* release (~9k hand-drawn leaf boxes) so boxes mark individual leaves.
- **Is the severity real?** It's an estimate: the share of yellow/brown pixels in the leaf area (HSV thresholds). It fails for white symptoms (powdery mildew), viral mottling and brown backgrounds. A true infected-area % needs lesion segmentation (YOLOv8-seg) — planned.
- **What happens on a random photo?** If no box passes 25 % confidence, the API returns "no detection". (The original code returned a random disease here — we removed that.)
- **Overfitting?** Watch val loss vs train loss in `results.png`; augmentation, frozen backbone and early stopping (best.pt = best val fitness) limit it.
- **What is NMS?** Removes duplicate boxes: keep the highest-confidence box, drop others with IoU above a threshold.
- **Why Flask + React?** Clear separation: model in Python, UI in the browser; the same API can serve a mobile app later.
- **How is data split?** Random 70/20/10 with fixed seed 42; test split is never used for training.

---

## 8. Team contribution (fill in before the viva)

| Member | Part owned | Files |
|---|---|---|
| _Name 1_ | Dataset analysis & pipeline | `plantdoc_preprocessing.py`, `generate_yolo_labels_from_folders.py`, `dataset_preparation.py` |
| _Name 2_ | Model training & evaluation | `train_yolo.py`, `PlantDoc_YOLOv8_Colab.ipynb`, `reports/training/` |
| _Name 3_ | Inference & backend API | `model/predict.py`, `app.py` |
| _Name 4_ | Frontend | `DTI/src/` |

Each member should be able to run and explain their own part.

---

## 9. Next milestone (final)

GPU training (YOLOv8s, 640 px, 100 epochs) · real lesion boxes / segmentation · add PlantVillage data · ONNX
export for mobile · Gemini-based treatment chatbot · weather-based risk tips · Hindi and other regional languages ·
user accounts.
