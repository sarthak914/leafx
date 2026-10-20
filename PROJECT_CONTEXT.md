# PROJECT CONTEXT — AgroNiddan (LeafX): AI Plant Disease Detection

> Purpose of this file: a self-contained, complete description of the project so that an AI assistant (or any
> reader) can understand, explain, or help extend it **without access to the source code**. Everything below
> reflects the actual state of the code at Milestone 2.

---

## 1. One-paragraph summary

AgroNiddan (LeafX) is a web application that diagnoses plant leaf diseases from a photo. A user uploads (or
captures with a camera) a leaf image in a React web UI; the image is sent to a Python Flask REST API, which runs a
**YOLOv8n object-detection model** (Ultralytics, PyTorch) fine-tuned on the **PlantDoc** dataset (28 classes, 13
plant species, real-field images). The detection is post-processed into farmer-friendly output — plant name,
condition (disease or "Healthy"), confidence %, a heuristic severity % and level (None/Low/Medium/Severe), a
plant-health %, and a rule-based treatment suggestion — returned as JSON together with an annotated image (bounding
boxes drawn), and logged to a SQLite history table. The project is a university AI/ML course project; this is the
**Milestone 2 (M2)** build, i.e. a deliberately partial but working end-to-end implementation. Additional features
(chatbot, weather, auth, community, multi-language) existed in an earlier full version and are planned for the
final milestone.

---

## 2. Academic context

- Course: AI/ML project, evaluated in milestones. M2 evaluation (12 marks) criteria:
  1. Review of existing systems and project feasibility (3)
  2. Objectives and methodology of the proposed work (3)
  3. Relevance of algorithms / techniques (3)
  4. Synchronization of project design & implementation (3)
- The M2 report must also cover the M1 criterion: Problem domain identification and analysis (3).
- A viva and at least a partial live demo are required; marks are per individual contribution.
- Original codebase author credited in README: Harsh Goyal (github.com/harshgoyalhgm). Team-member names are
  placeholders in `VIVA_NOTES.md` section 8.

---

## 3. Problem domain (M1)

- **Domain:** precision agriculture; computer vision for crop health.
- **Problem:** leaf diseases cause major yield loss; diagnosis by eye needs experts who are scarce in rural areas;
  late or wrong diagnosis leads to crop loss or excessive/incorrect pesticide use.
- **Users:** small farmers, home gardeners, agricultural extension workers with a phone camera.
- **Input → Output:** one leaf photo → plant + condition (one of 28 classes) + confidence + severity + treatment.
- **Challenges:** cluttered real-field backgrounds, multiple leaves, lighting variation, visually similar diseases
  (e.g. early vs late blight), small and imbalanced classes.

---

## 4. Existing systems & feasibility (M2)

| Existing approach | Limitation | This project |
|---|---|---|
| Manual expert inspection / agricultural extension centres | slow, not scalable, not always available | instant, any browser |
| CNN classifiers on PlantVillage (≈99 % accuracy in lab, e.g. Mohanty et al. 2016) | lab images on plain backgrounds; accuracy collapses on field photos | trained on PlantDoc real-field images |
| Commercial apps (e.g. Plantix) | closed source, need account/internet | open, local, explainable pipeline |
| Pure image classifiers | one label, no location | YOLO detection gives box → location + area for severity |

PlantDoc paper (Singh et al., CoDS-COMAD 2020): even its best fine-tuned classifier reached only ~70 % accuracy —
the dataset is genuinely hard.

**Feasibility:** technical (open-source stack, transfer learning, trains on a laptop CPU, full training on free
Colab T4 GPU); data (public, CC-BY-4.0); economic (no licence cost); operational (web UI usable on a phone;
~150–300 ms inference per image on CPU).

---

## 5. Dataset

- **Name:** PlantDoc (classification release), github.com/pratikkayal/PlantDoc-Dataset, licence CC-BY-4.0.
- **Size:** 2,572 images (2,336 official train + 236 official test), web-scraped real-field photos, varied sizes.
- **Classes (28):** 0 Apple Scab Leaf, 1 Apple leaf, 2 Apple rust leaf, 3 Bell_pepper leaf, 4 Bell_pepper leaf spot,
  5 Blueberry leaf, 6 Cherry leaf, 7 Corn Gray leaf spot, 8 Corn leaf blight, 9 Corn rust leaf, 10 Peach leaf,
  11 Potato leaf early blight, 12 Potato leaf late blight, 13 Raspberry leaf, 14 Soyabean leaf,
  15 Squash Powdery mildew leaf, 16 Strawberry leaf, 17 Tomato Early blight leaf, 18 Tomato Septoria leaf spot,
  19 Tomato leaf, 20 Tomato leaf bacterial spot, 21 Tomato leaf late blight, 22 Tomato leaf mosaic virus,
  23 Tomato leaf yellow virus, 24 Tomato mold leaf, 25 Tomato two spotted spider mites leaf, 26 grape leaf,
  27 grape leaf black rot. (Names without a disease, e.g. "Tomato leaf", are the healthy class of that plant.)
- **Imbalance:** e.g. Corn leaf blight 191 images vs spider mites 2 images (train only, absent from official test).
- **Integrity check result:** 12 problematic files (corrupt or unusual image modes) out of 2,572.
- **Final split used for training** (after pairing, random shuffle with seed 42, 70/20/10):
  train 1,796 · val 513 · test 256 (total 2,565 labelled pairs).
- **Labels:** the classification release has no lesion bounding boxes, so every image is given ONE box covering the
  whole image (`<class_id> 0.5 0.5 1.0 1.0`, YOLO normalized format) labelled with its folder name. This is a
  "weak label" design: detection then effectively acts as classification + whole-leaf localisation.
- **Storage:** raw data and the prepared YOLO dataset live outside the project (to keep them out of OneDrive), at
  `C:/Users/Naman Sharma/PlantDoc_data/` (`PlantDoc-Dataset-master/`, `labels_src/`, `dataset/`). Images in
  `dataset/` were converted to RGB JPEG with the longest side ≤ 640 px (~254 MB).

---

## 6. Objectives (M2)

1. Reproducible dataset pipeline: EDA, integrity check, labels, split, resize.
2. Train a YOLOv8 detector on 28 PlantDoc classes; evaluate with precision, recall, mAP@50, mAP@50-95.
3. Turn raw detections into farmer-friendly output (plant, condition, severity, health, treatment).
4. Serve it through a REST API and a web UI with scan history.

---

## 7. Methodology / pipeline (script by script)

All scripts are in the project root unless noted. Run order:

### 7.1 `plantdoc_preprocessing.py` — EDA + integrity check
- Args: `--dataset <folder with train/<class>/ and test/<class>/>` (required), `--output-dir` (default `reports`).
- Class `PlantDocPreprocessor`: counts images per class per split; opens every image with PIL to detect corrupt
  files, zero-size or unusual modes; computes width/height/file-size stats; builds a pandas DataFrame; writes
  `reports/plantdoc_preprocessing_report.txt`; saves plots `reports/plantdoc_class_distribution.png` (train vs test
  bars per class) and `reports/plantdoc_class_imbalance.png` (sorted horizontal bars of total per class).
- Uses matplotlib Agg backend (no windows). Handles classes missing from a split.

### 7.2 `generate_yolo_labels_from_folders.py` — weak labels
- Args: `--source-root`, `--labels-out`, `--dataset-dir`.
- Discovers class names = sorted union of folder names in train/ and test/ (Python case-sensitive sort, so the
  lowercase "grape" classes come last), assigns ids 0..27, writes one `.txt` label per image
  (`labels_out/<split>/<class>/<stem>.txt`), writes `classes_mapping.txt` (id → name) and regenerates `data.yaml`
  (`path: <dataset-dir>`, `train: images/train`, `val: images/val`, `test: images/test`, `nc: 28`, `names`).

### 7.3 `dataset_preparation.py` — split + resize into YOLO layout
- Args: `--source-dirs`, `--labels-root`, `--output-dir`, `--train/--val/--test` ratios (0.7/0.2/0.1),
  `--seed` (42), `--mode copy|move`, `--skip-missing-labels`, `--dry-run`, `--max-size` (used: 640).
- Pairs each image with its label, writes `dataset_preparation_report.json`, shuffles, splits, and writes
  `dataset/images/{train,val,test}` + `dataset/labels/{train,val,test}`. File names are prefixed with the class
  to avoid collisions (`Tomato_leaf_late_blight__<stem>.jpg`). With `--max-size`, images are re-encoded as RGB
  JPEG (quality 95) with the longest side capped — labels stay valid because they are normalized.

### 7.4 `train_yolo.py` — training
- Args: `--data` (data.yaml), `--model` (yolov8n.pt), `--epochs`, `--imgsz`, `--batch` (-1 auto), `--device`,
  `--workers` (default 2), `--freeze` (default 0).
- Validates that data.yaml resolves and train images exist, loads the COCO-pretrained model, calls
  `model.train(...)` with `project=<repo>/runs`, `name=detect/train`, `exist_ok=True`, `plots=True`, prints
  artifact paths and final metrics.
- **Milestone 2 baseline run (actual command):**
  `python train_yolo.py --data data.yaml --model yolov8n.pt --epochs 15 --imgsz 256 --batch 16 --device cpu --workers 2 --freeze 10`
- Hardware: Intel i5-1035G1 laptop CPU (4C/8T), 7.8 GB RAM, no usable CUDA GPU (PyTorch CPU build). ~6.5 min per
  epoch. These settings (small image size, frozen backbone, 15 epochs) were chosen to fit CPU/RAM limits.
- Outputs: `runs/detect/train/` with `weights/best.pt` (best validation fitness), `weights/last.pt`,
  `results.csv`, `results.png`, confusion matrices, PR/F1/P/R curves, `labels.jpg`, train/val batch previews.
  Copies of the key plots go to `reports/training/`.

### 7.5 `PlantDoc_YOLOv8_Colab.ipynb` — GPU training (planned for final)
- Google Colab T4 workflow: mount Drive, check/split dataset, write data.yaml with the same 28 names, train
  **YOLOv8s**, 640 px, 100 epochs, batch 16, validate, sample prediction, export best.pt/results to Drive.

### 7.6 `model/predict.py` — shared inference module (used by CLI and API)
- `REPO_ROOT`, `DEFAULT_WEIGHTS_REL = runs/detect/train/weights/best.pt`.
- `find_weights(root)`: order = env var `PLANTDOC_WEIGHTS` → default path → the run under `runs/` with the highest
  validation mAP50-95 in its `results.csv` → any `best.pt` under `runs/` → None.
- `CLASS_INFO`: maps each of the 28 raw labels → (plant, condition), e.g. "Tomato leaf late blight" →
  ("Tomato", "Late Blight"), "Apple leaf" → ("Apple", "Healthy"), "Corn leaf blight" → ("Corn", "Northern Leaf Blight").
  Also parses PlantVillage-style labels ("Tomato___Early_blight") as a fallback.
- `PESTICIDE_RECOMMENDATIONS`: condition → short first-line advice, e.g. Early Blight → "Use Mancozeb or
  Chlorothalonil; remove infected lower leaves and rotate crops"; Late Blight → Copper fungicide or Mancozeb;
  Mosaic Virus → no chemical cure, remove plants; Yellow Leaf Curl Virus → control whiteflies; Healthy → no treatment.
  Unknown → "Consult local agronomist".
- `estimate_lesion_pct(image_bgr, bbox)`: classical colour analysis inside the detected box with OpenCV HSV
  (hue scale 0–180). Plant pixels = saturation ≥ 40, value ≥ 40, hue < 90. Lesion pixels = plant pixels with
  hue < 35 (yellow/brown instead of green). Returns lesion/plant × 100, or None if < 100 plant pixels.
- `compute_infected_area_pct` = box area / image area × 100 — used only as a fallback when the colour estimate
  returns None (box area alone is meaningless here because training boxes cover the whole image).
- Severity % = infected area %. `severity_level`: <10 Low, <30 Medium, else Severe.
- `summarize_detections(result, model)` → list of `DetectionSummary(plant, disease, is_healthy, confidence_pct,
  severity_pct, severity, health_pct, recommendation, bbox_xyxy, infected_area_pct)`. For a healthy class: infected
  area 0, severity 0 / "None", health 100. Otherwise health = 100 − infected area.
- CLI: `python model/predict.py --image <path> [--weights auto] [--conf 0.25] [--imgsz] [--augment] [--device]`
  prints Plant, Disease, Confidence, Severity, Health, Recommendation, and saves an annotated image to
  `inference_outputs/`.

### 7.7 `app.py` — Flask REST API
- Loads `.env` (python-dotenv), resolves weights via `find_weights`, loads the YOLO model once at startup. If no
  weights are found, the server still starts and `/api/predict` returns HTTP 503 with instructions.
- Strict JSON provider (no NaN/Infinity, numpy → Python types). CORS enabled. Runs on `127.0.0.1:5000`.
- SQLite `predictions.db`, table `predictions(id, created_at, plant, disease, confidence_pct, severity_pct,
  severity_level, health_pct, recommendation)`.
- **Endpoints**
  - `GET /api/health` → `{status:"ok", modelLoaded, weights, weightsPath, classes}`.
  - `POST /api/predict` body `{imageDataUrl:"data:image/...;base64,...", conf?:0.25}`.
    Decodes to a PIL RGB image (passed directly to YOLO — raw numpy would be treated as BGR), runs
    `MODEL.predict`, draws boxes (`result.plot()`), summarizes detections sorted by confidence.
    - Detected → `{status:"detected", plant, disease, isHealthy, confidence, severity, severityLevel, plantHealth,
      suggestion, bbox, annotatedImage (JPEG data URL), detections:[...], inferenceMs}`; row saved to history.
    - Nothing above threshold → `{status:"no_detection", message, annotatedImage, detections:[], inferenceMs}`
      (not saved). The original code returned a random fake disease here; this was removed on purpose.
    - Bad image → 400 `{error}`; no model → 503; other failures → 500.
  - `GET /api/history` → last 25 predictions (newest first).
  - `GET /` and any non-API path → serves the built React app from `DTI/dist` (SPA fallback to index.html).

### 7.8 `DTI/` — React frontend
- Stack: React 19, Vite 7, Tailwind CSS 3, Framer Motion, React Router 7, i18next (English only at M2).
- Dark green "glass" theme (colors brand #061A13, leafPrimary #10B981, leafSecondary #34D399).
- Routes: `/` Landing (hero + 3-step "Capture → Detect → Advise"), `/scan` (webcam via getUserMedia with frame
  capture, or file upload; images downscaled client-side to ≤1024 px JPEG; "Scan Now" posts to `/api/predict`;
  result + image stored in localStorage), `/results` (annotated image, plant, condition, confidence, severity
  coloured by level, animated health bar, inference time, recommendation, disclaimer, list of all detections; or
  a "Nothing detected" card), `/history` (fetches `/api/history`; cards for total / healthy / diseased; table).
- Structure: `src/layers/ui/pages` (Landing, Scan, Results, History), `src/layers/ui/components` (Navbar,
  BottomNav for mobile, Button, Card, PageTransition), `src/layers/ai/components/ScanOverlay.jsx` (scan animation),
  `src/layers/services/api` (`apiClient.js` fetch wrapper with clear errors, `predictionApi.js`).
- Built output is in `DTI/dist` and is served by Flask, so the demo needs no Node.js. Vite dev proxy forwards
  `/api` to Flask for `npm run dev`.

### 7.9 Other files
- `run_demo.cmd` — Windows launcher: uses `venv` Python if present else system Python, checks `DTI/dist`, opens the
  browser after ~8 s, runs `app.py`.
- `requirements.txt` — pandas, numpy, Pillow, matplotlib, ultralytics, Flask, flask-cors, python-dotenv.
- `.env.example` — only optional `PLANTDOC_WEIGHTS`. (Previously contained real Gemini/OpenWeather API keys;
  removed.)
- `data.yaml`, `classes_mapping.txt` — generated config (28 classes; `path` is an absolute path on this laptop).
- `reports/` — EDA report and plots; `reports/training/` — training curves.
- `demo_images/` — 8 images from the held-out test split for the live demo.
- `README.md` — overview, scope table, run instructions. `VIVA_NOTES.md` — demo script and viva Q&A.

---

## 8. Algorithms and techniques (M2 criterion 3)

- **YOLOv8 (Ultralytics, 2023):** single-stage detector. Backbone: CSPDarknet-style with C2f blocks + SPPF.
  Neck: PAN-FPN multi-scale feature fusion. Head: **anchor-free, decoupled** (separate box-regression and
  classification branches) at 3 scales. Loss: box = **CIoU + Distribution Focal Loss (DFL)**, class = **BCE**
  (shown as `box_loss`, `dfl_loss`, `cls_loss`). Post-processing: confidence threshold + **NMS** (IoU 0.7).
- **YOLOv8n** ("nano"): ~3.2 M parameters, ~8.7 GFLOPs at 640 px — chosen for CPU training/inference. YOLOv8s
  (~11 M params) is planned for GPU training.
- **Transfer learning:** start from COCO-pretrained weights; `freeze=10` freezes the backbone (first 10 modules)
  so only neck + head are trained → faster, less overfitting on ~1.8k images.
- **Augmentation (Ultralytics defaults):** mosaic, HSV colour jitter, horizontal flip, random scale/translate;
  mosaic disabled for the last 10 epochs.
- **Optimizer:** Ultralytics `optimizer=auto` (selects AdamW for short runs), warmup epochs, linear LR decay.
- **Metrics:** Precision = TP/(TP+FP); Recall = TP/(TP+FN); IoU = intersection/union of boxes; mAP@50 = mean
  over classes of PR-curve area with IoU ≥ 0.5; mAP@50-95 = averaged over IoU thresholds 0.50–0.95.
  `best.pt` = epoch with best fitness (0.1·mAP50 + 0.9·mAP50-95). Because boxes are whole-image, mAP here mostly
  reflects classification correctness.
- **Post-processing (classical CV + rules):** infected area estimated by HSV colour thresholding inside the
  detected box (yellow/brown vs green plant pixels) → severity level and plant health; rule-based treatment lookup.
  This is an estimate: it misses white symptoms (powdery mildew), viral mottling, and can be fooled by brown
  backgrounds.

---

## 9. Results (Milestone 2 baseline, CPU)

YOLOv8n (3.0 M params after fusing, 8.1 GFLOPs), 256 px, frozen backbone, 15 epochs, batch 16, CPU, 54 min total.

| Split | Images | Precision | Recall | mAP@50 | mAP@50-95 | Top-1 accuracy |
|---|---|---|---|---|---|---|
| Validation | 513 | 0.368 | 0.487 | 0.424 | 0.423 | — |
| Test (held out) | 256 | 0.390 | 0.387 | 0.435 | 0.435 | 39.1 % (100/256) |

Top-1 = highest-confidence box has the correct class; on the test set 96 were wrong and 60 had no box above 25 %
confidence. Validation mAP@50 per epoch: 0.066, 0.136, 0.183, 0.224, 0.256, 0.299, 0.345, 0.356, 0.384, 0.403,
0.408, 0.411, 0.418, 0.423, 0.422 (best.pt = epoch 14) — still rising, so more epochs/GPU would help.
mAP@50 ≈ mAP@50-95 because boxes are whole-image (any correct-class box overlaps almost perfectly).
Inference ≈ 40–200 ms per image on CPU. Best per-class validation mAP@50 includes Apple leaf (healthy) 0.686;
weak classes include Apple Scab (0.22) and Bell pepper leaf (0.30).
Plots: `reports/training/results.png`, `confusion_matrix_normalized.png`, `BoxPR_curve.png`, `BoxF1_curve.png`.
Live demo images (`demo_images/`) are test images the model classifies correctly; `demo_images/hard_cases/` holds
two confident mistakes (tomato late blight → squash powdery mildew; corn gray leaf spot → corn common rust).

Interpretation: modest accuracy is expected for a 15-epoch CPU baseline on a hard real-field dataset; the
pipeline is complete and accuracy is the main target for the final milestone.

---

## 10. Design ↔ implementation map (M2 criterion 4)

| Design block | Implementation |
|---|---|
| Data analysis | `plantdoc_preprocessing.py` → `reports/` |
| Labelling | `generate_yolo_labels_from_folders.py` → `data.yaml`, `classes_mapping.txt` |
| Split + preprocessing | `dataset_preparation.py` |
| Training | `train_yolo.py` / Colab notebook → `runs/detect/train/weights/best.pt` |
| Inference + severity + advice | `model/predict.py` (single source of truth) |
| API | `app.py` (`/api/health`, `/api/predict`, `/api/history`) |
| Storage | SQLite `predictions.db` |
| UI | `DTI/src` → built `DTI/dist` |

Request flow: browser reads image → downscale ≤1024 px → base64 → POST `/api/predict` → PIL → YOLOv8 →
`summarize_detections` → JSON + annotated JPEG → Results page → SQLite → History page.

---

## 11. Known limitations (stated honestly)

1. Weak full-image labels → model localises the whole image, not individual leaves/lesions; "infected area" is a
   colour-based estimate. Fix: PlantDoc object-detection release (~9k hand-drawn leaf boxes) or YOLOv8-seg for
   lesion masks.
2. Small, imbalanced dataset (spider mites class has 2 images).
3. CPU-limited baseline (256 px, 15 epochs, frozen backbone) → modest mAP.
4. Treatment advice is a static lookup, not personalised; shown with a "consult an agronomist" disclaimer.
5. Random split may place near-duplicate web images in different splits.
6. `data.yaml` holds an absolute local dataset path.

---

## 12. Scope: what M2 includes vs what is deferred

**Included:** everything in sections 7.1–7.9.

**Removed from the earlier full version (kept in the original zip, planned for final):** Google Gemini chatbot
(`/chat`) and Gemini treatment write-ups (`/api/recommendation`); OpenWeather weather page with farming tips;
user signup/login (SQLite, SHA-256 hashes); community forum (posts, likes, comments, follows, topics); shop page;
dashboard with hard-coded stats; Hindi/Marathi/Gujarati/Tamil/Telugu translations; a 38-class PlantVillage
training notebook; legacy static HTML pages and duplicate asset folders.

**Final milestone plan:** YOLOv8s on GPU (640 px, ~100 epochs) · real leaf/lesion boxes or segmentation ·
add PlantVillage data · ONNX/TFLite export for mobile · Gemini chatbot · weather-based risk tips · regional
languages · user accounts.

---

## 13. How to run

```
pip install -r requirements.txt
python app.py            # or double-click run_demo.cmd → http://127.0.0.1:5000
python model/predict.py --image demo_images/<file>.jpg
```
Frontend rebuild (only after editing `DTI/src`): `cd DTI && npm install && npm run build`.
