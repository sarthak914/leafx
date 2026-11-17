# 🌿 AgroNiddan (LeafX) — AI Plant Disease Detection

**Milestone 3 build.** Upload or photograph a leaf: a YOLOv8 **classifier** trained on the PlantDoc dataset names the
plant and its disease (28 classes), a **Grad-CAM heatmap** shows which part of the leaf drove the decision, a YOLOv8
**detector** + colour analysis estimate severity and plant health, and a **Gemini-powered assistant** answers
follow-up questions. Users can sign up and keep their own scan history. Flask REST API + React web app.

---

## 📌 Scope by milestone

| Status | Part |
|---|---|
| ✅ M2 | Dataset analysis, data pipeline (labels → 70/20/10 split → resize), YOLOv8n detector baseline |
| ✅ M2 | Flask API + React UI (Home → Scan → Results → History), honest "nothing detected" answers |
| ✅ **M3** | **YOLOv8s classifier** on the same split: test accuracy **39.1 % → 67.6 %** (top-5: 95.7 %) |
| ✅ **M3** | **Grad-CAM** explainable-AI heatmap on every scan, top-3 predictions |
| ✅ **M3** | **User accounts** (sign up / log in / log out, scrypt-hashed passwords) + per-user scan history |
| ✅ **M3** | **AI farming assistant** (Google Gemini) with scan context, multilingual, automatic model fallback |
| ✅ **M3** | Robustness: 54-case API test suite, input validation, mobile layout fixes |
| ⏳ Next | Stronger backbone / larger images on GPU to pass 70 %, lesion segmentation, regional-language UI |

---

## 🧠 How it works (M3)

```
Leaf photo (upload / camera)
        │  React: resized to ≤1024 px, sent as base64
        ▼
POST /api/predict  (Flask)
        │
        ├─► YOLOv8s-cls classifier (runs/classify/train/weights/best.pt)
        │      → plant + condition + confidence, top-3
        ├─► YOLOv8n detector (runs/detect/train/weights/best.pt)
        │      → box around the leaf (whole image if none)
        ├─► HSV colour check inside the box
        │      → infected % = yellow/brown share of plant pixels → severity Low/Medium/Severe, health %
        ├─► Grad-CAM on the classifier's last feature layer → heatmap
        └─► rule-based treatment suggestion
        ▼
JSON + annotated image + heatmap → Results page → saved to the user's history (if logged in)
        │
        └─► "Ask the AI assistant" → POST /api/chat (Gemini, with the scan as context)
```

If the classifier's best guess is below 25 % confidence, the app says it isn't sure instead of guessing.

---

## 📊 Results

Same PlantDoc split for both models: 1,791 train / 513 val / 256 test (held out, scored once).

| Model | Milestone | Test top-1 | Test top-5 | Notes |
|---|---|---|---|---|
| YOLOv8n detector (full-image boxes) | M2 | 39.1 % | — | 15 epochs, 256 px, frozen backbone (mAP@50 0.435) |
| **YOLOv8s-cls classifier** | **M3** | **67.6 %** (173/256) | **95.7 %** | 60 epochs, 224 px, ImageNet-pretrained, best = epoch 57 |

Validation accuracy: 21.6 % (epoch 1) → 63.4 % (epoch 5) → 68.0 % (epoch 57, best). CPU training (Intel i5-1035G1),
2.6 h for the resumed part, about 5 min per epoch. Inference ≈ 0.4–0.7 s per scan including Grad-CAM.

Per-class test accuracy and the most common confusions are in `reports/training_cls/test_report.txt`; curves and
the confusion matrix are in `reports/training_cls/`. Strongest: corn rust, potato late blight, raspberry,
strawberry (100 %). Weakest: soybean (0/4), tomato early blight and leaf mold (1/6). Most common mistake: potato
early blight → late blight (5×).

---

## 🚀 Running the app

Prerequisites: Python 3.10+ and `pip install -r requirements.txt`.

```bash
# Windows: double-click run_demo.cmd (picks the Python that has the packages), or
python app.py
# then open http://127.0.0.1:5000
```

**AI assistant:** copy `.env.example` to `.env` and set `GEMINI_API_KEY=` (free key from
https://aistudio.google.com/app/apikey). Without a key everything else works; the assistant says it isn't configured.

Command-line prediction (saves the annotated image and Grad-CAM heatmap):

```bash
python model/predict.py --image demo_images/Corn_rust_leaf__Corn-Southern-Rust-7-17-F1.jpg
python model/predict.py --image <leaf.jpg> --detector-only   # Milestone 2 behaviour
```

Rebuilding the frontend (only after changing `DTI/src`): `cd DTI && npm install && npm run build`.

## 🔌 API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Status, loaded models, whether the assistant is configured |
| POST | `/api/predict` | `{imageDataUrl, conf?}` → diagnosis, top-3, severity, annotated image, heatmap |
| GET | `/api/history` | Logged-in user's last 25 scans (401 for guests) |
| POST | `/api/auth/signup` · `/api/auth/login` · `/api/auth/logout` | Accounts (session cookie) |
| GET | `/api/auth/me` | Current user or `null` |
| POST | `/api/chat` | `{message, history?, context?}` → Gemini reply |

## 🔁 Reproducing training

```bash
python plantdoc_preprocessing.py --dataset <PlantDoc-Dataset> --output-dir reports
python generate_yolo_labels_from_folders.py --source-root <PlantDoc-Dataset> --labels-out <data>/labels_src --dataset-dir <data>/dataset
python dataset_preparation.py --source-dirs <PlantDoc-Dataset>/train <PlantDoc-Dataset>/test --labels-root <data>/labels_src --output-dir <data>/dataset --skip-missing-labels --mode copy --max-size 640
python train_yolo.py --epochs 15 --imgsz 256 --batch 16 --device cpu --freeze 10           # M2 detector
python prepare_classification_dataset.py --data data.yaml --output-dir <data>/dataset_cls   # M3: same split, class folders
python train_classifier.py --data <data>/dataset_cls --model yolov8s-cls.pt --epochs 60 --device cpu
python train_classifier.py --data <data>/dataset_cls --resume                               # continue after an interruption
```

---

## 🏗️ Project structure

```
app.py                         Flask API (predict, history, auth, chat) + serves the built frontend
model/predict.py               Detector inference, class → (plant, disease), colour severity, treatment table, CLI
model/classifier.py            Classifier, Grad-CAM, combined diagnose() used by API and CLI
plantdoc_preprocessing.py      Dataset EDA + integrity check
generate_yolo_labels_from_folders.py · dataset_preparation.py   Labels, split, resize
prepare_classification_dataset.py    Same split in class-folder layout for the classifier
train_yolo.py · train_classifier.py  Training (detector / classifier)
reports/                       EDA plots, detector (training/) and classifier (training_cls/) results
runs/detect/train/weights/best.pt    M2 detector      runs/classify/train/weights/best.pt   M3 classifier
DTI/                           React + Vite + Tailwind frontend (src/, built dist/)
demo_images/                   Held-out test images for the live demo (+ hard_cases/)
PROJECT_CONTEXT.md             Complete technical description and change log
VIVA_NOTES.md                  Demo script and viva Q&A
```

## 🛠️ Tech stack

Python, Ultralytics YOLOv8, PyTorch, OpenCV, Flask, SQLite, Werkzeug security, Google Gen AI SDK (Gemini) ·
React 19, Vite, Tailwind CSS, Framer Motion, React Router, i18next.

## 👨‍💻 Credits

Original codebase: **Harsh Goyal** ([github.com/harshgoyalhgm](https://github.com/harshgoyalhgm)).
Dataset: D. Singh et al., *"PlantDoc: A Dataset for Visual Plant Disease Detection"*, CoDS-COMAD 2020.
