# 🌿 AgroNiddan (LeafX) — AI Plant Disease Detection

**Milestone 2 build.** A YOLOv8 model trained on the PlantDoc dataset detects the condition of a leaf from a photo,
estimates severity and plant health, and suggests a first-line treatment. It is served through a Flask REST API
and a React web app.

---

## 📌 Milestone 2 scope

| Status | Part |
|---|---|
| ✅ Done | Dataset analysis: class counts, image integrity check, image size stats, plots (`reports/`) |
| ✅ Done | Data pipeline: folder labels → YOLO labels → 70/20/10 split → resize to 640 px |
| ✅ Done | Baseline YOLOv8n training (transfer learning from COCO weights) + metrics/plots |
| ✅ Done | Inference module: disease mapping (28 classes), severity, plant health, treatment advice |
| ✅ Done | Flask API: `/api/health`, `/api/predict`, `/api/history` (SQLite) |
| ✅ Done | React UI: Home → Scan (upload / camera) → Results (annotated image) → History |
| ⏳ Final | Full GPU training (YOLOv8s, 640 px, 100 epochs) on Colab — `PlantDoc_YOLOv8_Colab.ipynb` |
| ⏳ Final | Lesion-level boxes / segmentation for a true infected-area %, more data (PlantVillage) |
| ⏳ Final | AI chatbot, weather-based tips, regional languages, user accounts |

---

## 🧠 How it works

```
Leaf photo (upload / camera)
        │  React: resized to ≤1024 px, sent as base64
        ▼
POST /api/predict  (Flask)
        │
        ▼
YOLOv8n detector  (best.pt, 28 PlantDoc classes)
        │  boxes + class + confidence
        ▼
model/predict.py
  ├─ class → (plant, condition)          e.g. "Tomato leaf late blight" → Tomato, Late Blight
  ├─ infected area % = yellow/brown share of plant pixels inside the box (HSV colour thresholding)
  ├─ severity % = infected area  → Low (<10) / Medium (<30) / Severe
  ├─ plant health % = 100 − infected area  (healthy class → 100 %)
  └─ treatment suggestion (rule-based lookup)
        │
        ▼
JSON + annotated image → Results page, saved to SQLite history
```

If nothing is detected above the confidence threshold (25 %), the API says so honestly; it never makes up a result.

---

## 📊 Dataset

**PlantDoc** (Singh et al., 2020, CC-BY-4.0): 2,572 real-field leaf images (scraped from the web, cluttered
backgrounds), 13 plant species, 28 classes (17 diseases + healthy leaves).

| Step | Script | Output |
|---|---|---|
| EDA + integrity check | `plantdoc_preprocessing.py` | `reports/` (report + 2 plots) — 12 problematic files found |
| Labels from folders (1 full-image box per image) | `generate_yolo_labels_from_folders.py` | `labels_src/`, `data.yaml`, `classes_mapping.txt` |
| Split 70/20/10 (seed 42) + resize ≤640 px | `dataset_preparation.py` | `dataset/images|labels/{train,val,test}` — 1,796 / 513 / 256 |

---

## 📈 Baseline results (Milestone 2)

See `reports/training/` for `results.png`, `confusion_matrix_normalized.png`, `PR_curve.png` and `results.csv`.

YOLOv8n, COCO-pretrained, backbone frozen, 15 epochs, 256 px, batch 16, CPU (Intel i5-1035G1), 54 min.

| Split | Images | Precision | Recall | mAP@50 | mAP@50-95 | Top-1 accuracy* |
|---|---|---|---|---|---|---|
| Validation | 513 | 0.368 | 0.487 | 0.424 | 0.423 | — |
| Test (held out) | 256 | 0.390 | 0.387 | 0.435 | 0.435 | 39.1 % (100/256) |

\*Top-1: the highest-confidence box has the correct class (60 test images had no box above 25 % confidence).
mAP@50 rose steadily from 0.066 (epoch 1) to 0.424 (epoch 15) and was still improving — longer GPU training is
the main lever for the final milestone. Inference: ~40–200 ms per image on CPU.

---

## 🚀 Running the demo

Prerequisites: Python 3.10+ with `pip install -r requirements.txt`. The trained weights must be at
`runs/detect/train/weights/best.pt` (or set `PLANTDOC_WEIGHTS` in `.env`).

```bash
# Windows: double-click run_demo.cmd, or
python app.py
# open http://127.0.0.1:5000
```

Command-line prediction:

```bash
python model/predict.py --image path/to/leaf.jpg
```

Rebuilding the frontend (only if you change `DTI/src`):

```bash
cd DTI
npm install
npm run build
```

## 🔁 Reproducing training

```bash
python plantdoc_preprocessing.py --dataset <PlantDoc-Dataset> --output-dir reports
python generate_yolo_labels_from_folders.py --source-root <PlantDoc-Dataset> --labels-out <data>/labels_src --dataset-dir <data>/dataset
python dataset_preparation.py --source-dirs <PlantDoc-Dataset>/train <PlantDoc-Dataset>/test --labels-root <data>/labels_src --output-dir <data>/dataset --skip-missing-labels --mode copy --max-size 640
python train_yolo.py --epochs 15 --imgsz 256 --batch 16 --device cpu --freeze 10   # CPU baseline
```

On a GPU (Colab T4) use `PlantDoc_YOLOv8_Colab.ipynb` (YOLOv8s, 640 px, 100 epochs).

---

## 🏗️ Project structure

```
app.py                     Flask REST API + serves the built frontend
model/predict.py           Shared inference logic (CLI + API)
plantdoc_preprocessing.py  Dataset EDA + integrity check
generate_yolo_labels_from_folders.py
dataset_preparation.py     Split + resize into YOLO layout
train_yolo.py              YOLOv8 training
PlantDoc_YOLOv8_Colab.ipynb  GPU training notebook
data.yaml                  YOLO dataset config (28 classes)
reports/                   EDA plots + training curves
runs/detect/train/         Training run (weights/best.pt)
DTI/                       React + Vite + Tailwind frontend (src/, dist/)
```

## 🛠️ Tech stack

Python, Ultralytics YOLOv8, PyTorch, Flask, SQLite, Pillow, pandas, matplotlib · React 19, Vite, Tailwind CSS,
Framer Motion, React Router.

## 👨‍💻 Credits

Original codebase: **Harsh Goyal** ([github.com/harshgoyalhgm](https://github.com/harshgoyalhgm)).
Dataset: D. Singh et al., *"PlantDoc: A Dataset for Visual Plant Disease Detection"*, CoDS-COMAD 2020.
