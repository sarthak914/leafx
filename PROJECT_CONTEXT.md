# PROJECT CONTEXT — AgroNiddan (LeafX): AI Plant Disease Detection

> Purpose of this file: a self-contained, complete description of the project so that an AI assistant (or any
> reader) can understand, explain, or help extend it **without access to the source code**. It reflects the actual
> state of the code at **Milestone 3 (M3)**, and section 14 records exactly what changed since Milestone 2 and why.

---

## 1. One-paragraph summary

AgroNiddan (LeafX) is a web application that diagnoses plant leaf diseases from a photo. A user uploads (or
captures with a webcam) a leaf image in a React web UI; it is sent to a Python Flask REST API. A **YOLOv8s
classification model** (Ultralytics, PyTorch), fine-tuned on the **PlantDoc** dataset (28 classes, 13 plant
species, real field photos), decides the plant and condition; a **YOLOv8n detector** (from Milestone 2) finds the
leaf box; an **HSV colour analysis** inside the box estimates the infected share of the leaf (severity and plant
health); **Grad-CAM** produces a heatmap of the image regions behind the decision; and a rule-based table suggests
a treatment. Results are returned as JSON with an annotated image and heatmap, and saved to the user's private scan
history if they are logged in (SQLite, scrypt-hashed passwords, session cookies). A **Google Gemini assistant**
answers follow-up farming questions, optionally using the latest scan as context. Final classifier accuracy on the
held-out test set: **67.6 % top-1, 95.7 % top-5** (Milestone 2 detector: 39.1 %).

---

## 2. Academic context

- University AI/ML course project, evaluated in milestones with a viva and a live demo; marks per individual
  contribution.
- M1 criterion: problem domain identification and analysis. M2 criteria: review of existing systems and
  feasibility; objectives and methodology; relevance of algorithms/techniques; synchronization of design and
  implementation. M3 continues the same project: better accuracy and more features.
- Original codebase author: Harsh Goyal (github.com/harshgoyalhgm). The M2/M3 versions were trimmed, fixed and
  extended from it. Team-member names are placeholders in `VIVA_NOTES.md` section 8.
- Repository: github.com/sarthak914/leafx (branch `main`). Commit author configured locally as `sarthak914`.

---

## 3. Problem domain

- **Domain:** precision agriculture; computer vision for crop health.
- **Problem:** leaf diseases cause major yield loss; diagnosis by eye needs experts who are scarce in rural areas;
  late or wrong diagnosis leads to crop loss or excessive/incorrect pesticide use.
- **Users:** small farmers, home gardeners, agricultural extension workers with a phone camera.
- **Input → output:** one leaf photo → plant + condition (28 classes) + confidence + severity + health + treatment
  + visual explanation, and a chat assistant for follow-up questions.
- **Challenges:** cluttered field backgrounds, multiple leaves, lighting variation, visually similar diseases
  (e.g. potato early vs late blight), small and imbalanced classes.

## 4. Existing systems & feasibility

| Existing approach | Limitation | This project |
|---|---|---|
| Manual expert inspection / extension centres | slow, not scalable | instant, any browser |
| CNNs on PlantVillage (≈99 % in the lab, Mohanty et al. 2016) | plain backgrounds; accuracy collapses on field photos | trained on PlantDoc real-field images |
| Commercial apps (e.g. Plantix) | closed source, black box | open pipeline with Grad-CAM explanations |
| Plain image classifiers | no location / severity | classifier + detector box + colour severity |

PlantDoc paper (D. Singh et al., "PlantDoc: A Dataset for Visual Plant Disease Detection", CoDS-COMAD 2020,
arXiv 1911.10317): the best fine-tuned classifier reached about 70 % accuracy, so the dataset is genuinely hard.

**Feasibility:** open-source stack; transfer learning works with ~1.8k training images; everything trains on a
laptop CPU (Intel i5-1035G1, 4 cores/8 threads, 7.8 GB RAM, no usable CUDA GPU — the MX230 is not supported by the
installed CPU-only PyTorch 2.11); Gemini has a free API tier; the UI works on phones.

---

## 5. Dataset

- **PlantDoc classification release**, github.com/pratikkayal/PlantDoc-Dataset, licence CC-BY-4.0.
- 2,572 images (2,336 official train + 236 official test), web-scraped field photos of varied sizes.
- **28 classes** (folder names; names without a disease are the healthy class of that plant):
  0 Apple Scab Leaf, 1 Apple leaf, 2 Apple rust leaf, 3 Bell_pepper leaf, 4 Bell_pepper leaf spot,
  5 Blueberry leaf, 6 Cherry leaf, 7 Corn Gray leaf spot, 8 Corn leaf blight, 9 Corn rust leaf, 10 Peach leaf,
  11 Potato leaf early blight, 12 Potato leaf late blight, 13 Raspberry leaf, 14 Soyabean leaf,
  15 Squash Powdery mildew leaf, 16 Strawberry leaf, 17 Tomato Early blight leaf, 18 Tomato Septoria leaf spot,
  19 Tomato leaf, 20 Tomato leaf bacterial spot, 21 Tomato leaf late blight, 22 Tomato leaf mosaic virus,
  23 Tomato leaf yellow virus, 24 Tomato mold leaf, 25 Tomato two spotted spider mites leaf, 26 grape leaf,
  27 grape leaf black rot. (Python's case-sensitive sort puts the lowercase "grape" classes last.)
- **Imbalance:** e.g. Corn leaf blight 191 images vs spider mites 2 (train only; absent from val/test).
- **Integrity check:** 12 problematic files (corrupt or unusual image modes); unreadable ones are skipped.
- **Split (seed 42, 70/20/10), used by BOTH models:** train **1,791**, val **513**, test **256**. (The split step
  reported 1,796 train pairs; 5 train files were lost to file-name collisions that differ only in letter case, which
  Windows treats as the same file. Val and test are complete.)
- **Detection layout** (`dataset/`): RGB JPEGs, longest side ≤ 640 px, one label per image
  `<class> 0.5 0.5 1.0 1.0` (a full-image box, because the classification release has no lesion boxes).
- **Classification layout** (`dataset_cls/`, new in M3): `train|val|test/<class name>/*.jpg`, built from the same
  split, so classifier and detector are evaluated on identical test images.
- **Location:** outside OneDrive to avoid syncing ~1 GB: `C:/Users/Naman Sharma/PlantDoc_data/`
  (`PlantDoc-Dataset-master/`, `labels_src/`, `dataset/`, `dataset_cls/`, `dti_build/` frontend build folder,
  model backups, training logs). Not in git.

---

## 6. Objectives

1. Reproducible dataset pipeline: EDA, integrity check, labels, split, resize. *(M2)*
2. Baseline detector and evaluation with mAP/precision/recall. *(M2)*
3. Higher accuracy with a classifier on the same split; evaluate top-1/top-5. *(M3)*
4. Explainability (Grad-CAM), severity, health and treatment output. *(M3)*
5. REST API + web UI with accounts, private history and an AI assistant. *(M3)*

---

## 7. Pipeline and scripts

Run order (paths in angle brackets are on the local machine):

1. `plantdoc_preprocessing.py --dataset <PlantDoc-Dataset> --output-dir reports` — class counts per split,
   opens every image (PIL) to find corrupt/invalid files, size stats, pandas DataFrame, writes
   `reports/plantdoc_preprocessing_report.txt`, `reports/plantdoc_class_distribution.png` (train vs test bars),
   `reports/plantdoc_class_imbalance.png` (sorted totals). Matplotlib Agg backend; handles classes missing from a split.
2. `generate_yolo_labels_from_folders.py --source-root ... --labels-out ... --dataset-dir ...` — class ids from the
   sorted union of folder names, one full-image label per image, writes `classes_mapping.txt` and `data.yaml`
   (`path: <dataset>`, `train/val/test: images/<split>`, `nc: 28`, `names`).
3. `dataset_preparation.py --source-dirs ... --labels-root ... --output-dir ... --skip-missing-labels --mode copy --max-size 640`
   — pairs images with labels, writes `dataset_preparation_report.json`, shuffles (seed 42), splits 70/20/10,
   re-encodes as RGB JPEG (quality 95, longest side ≤ 640), names prefixed with the class to avoid collisions.
4. `train_yolo.py` — M2 detector: `--data data.yaml --model yolov8n.pt --epochs 15 --imgsz 256 --batch 16 --device cpu --workers 2 --freeze 10`;
   output `runs/detect/train/` (absolute `project` path so Ultralytics doesn't nest folders).
5. **`prepare_classification_dataset.py --data data.yaml --output-dir <data>/dataset_cls`** *(M3)* — reads each
   image's class from its YOLO label, creates a folder for **every** class in **every** split (empty folders are
   allowed by torchvision ≥ 0.18, which keeps class indices identical across splits), **hard-links** files (no
   extra disk space; falls back to copying), shortens file names longer than 60 characters to
   `<first 60 chars>_<8-char md5>.jpg` (some web-scraped names broke Windows' 260-character path limit).
   Result: 1,791 / 513 / 256 images; spider mites empty in test.
6. **`train_classifier.py`** *(M3)* — `--data <dataset_cls> --model yolov8s-cls.pt --epochs 60 --imgsz 224 --batch 32 --device cpu --workers 2 --patience 15 --name train`,
   plus `--fraction` (quick speed tests) and **`--resume`** (continues `runs/classify/<name>/weights/last.pt` with
   the same settings, optimizer state and learning-rate schedule). After training it scores `best.pt` once on the
   test split. Output `runs/classify/train/`.
7. `PlantDoc_YOLOv8_Colab.ipynb` — optional GPU (Colab T4) detector training notebook from M2 (YOLOv8s, 640 px).

---

## 8. Models and training

### 8.1 Milestone 2 detector (kept, now only finds the leaf box)
YOLOv8n (3.0 M params fused, 8.1 GFLOPs), COCO-pretrained, backbone frozen (`freeze=10`), 15 epochs, 256 px,
batch 16, CPU, 54 min. Validation mAP@50 0.424; test mAP@50 0.435, precision 0.39, recall 0.39, **top-1 39.1 %**
(100/256; 60 images had no box above 25 %). Because boxes are whole-image, mAP@50 ≈ mAP@50-95.

### 8.2 Milestone 3 classifier (decides the disease)
- **Architecture:** YOLOv8s-cls — Conv/C2f backbone (layers 0–8), `Classify` head (1×1 conv 512→1280, SiLU,
  global average pool, dropout 0, linear 1280→28). 5,111,068 parameters fused, 12.5 GFLOPs at 224 px.
- **Pretraining:** ImageNet (`yolov8s-cls.pt`, 73.8 % ImageNet top-1); 156/158 tensors transferred (the 1000-class
  linear layer was replaced by a 28-class one).
- **Settings:** 60 epochs, 224 px (resize short side + centre crop), batch 32, `optimizer=auto` → **AdamW,
  lr 0.000313, momentum 0.9**, weight decay 0.0005, 3 warm-up epochs, linear LR decay (`lrf` 0.01), RandAugment,
  random erasing 0.4, horizontal flip 0.5, HSV jitter, cross-entropy loss, seed 0, deterministic.
- **Speed (CPU):** ~1.8 s/iteration for yolov8n-cls vs ~4.8 s for yolov8s-cls in a 10 % benchmark; ~5–5.5 min per
  full epoch for yolov8s-cls. The stronger s model was chosen to maximise accuracy.
- **Training history (validation top-1 % per epoch):**
  1:21.6 2:42.5 3:54.8 4:58.5 5:63.4 6:63.0 7:64.5 8:63.9 9:65.1 10:65.3 11:63.0 12:63.0 13:64.5 14:61.6 15:64.5
  16:63.9 17:64.1 18:64.9 19:65.7 20:65.3 21:63.9 22:65.9 23:66.5 24:66.3 25:66.9 26:65.7 27:64.7 28:66.9 29:67.1
  30:66.1 31:67.8 32:65.5 33:64.9 34:66.5 35:66.9 36:65.9 37:66.9 38:66.7 39:66.5 40:67.4 41:67.3 42:66.9 43:64.9
  44:66.9 45:66.9 46:66.5 47:67.1 48:67.8 49:65.5 50:67.1 51:65.9 52:67.4 53:66.7 54:65.5 55:66.5 56:67.4
  **57:68.0** 58:68.0 59:68.0 60:67.4.
  Train loss 3.25 → 0.10; validation loss 2.94 → 1.69 (gap = some overfitting, limited by augmentation and by
  keeping the best validation epoch).
- **Interruptions:** the laptop shut down after epoch 16. Training resumed from `last.pt` at epoch 17
  (`--resume`). The run had early stopping `patience=15`, which would have stopped it at epoch 25 because accuracy
  had plateaued; on request it was raised to 60 by editing `train_args.patience` inside `last.pt` so all 60 epochs
  ran. Backups of the epoch-10 best and the epoch-16 checkpoint were kept in `PlantDoc_data/`. A second apparent
  shutdown was only sleep — the process survived and finished.
- **Final model:** `runs/classify/train/weights/best.pt` = epoch 57 (val top-1 68.0 %, top-5 94.2 %; 10.3 MB).
- **Test set (256 held-out images, scored once): top-1 67.6 % (173/256), top-5 95.7 %**, ~41 ms/image.
- **Per-class test accuracy:** Soyabean 0/4, Tomato Early blight 1/6, Tomato mold 1/6, Bell pepper leaf spot 2/6,
  Tomato bacterial spot 4/11, Potato early blight 5/13, Cherry 2/5, Tomato late blight 4/9, Tomato healthy 3/6,
  Corn gray leaf spot 3/5, Tomato mosaic 5/8, Bell pepper healthy 7/11, Grape healthy 8/12, Tomato Septoria 15/21,
  Apple healthy 6/8, Apple rust 6/8, Corn leaf blight 12/16, Blueberry 13/17, Grape black rot 7/9, Apple scab 8/10,
  Tomato yellow virus 5/6, Peach 12/14, Squash powdery mildew 14/15, Corn rust 8/8, Potato late blight 6/6,
  Raspberry 9/9, Strawberry 7/7. (Full table: `reports/training_cls/test_report.txt`.)
- **Most common confusions:** Potato early blight → late blight (5×), Tomato bacterial spot → Septoria (4×),
  Bell pepper healthy → leaf spot (3×), Corn leaf blight → gray leaf spot (3×), Grape → Raspberry (3×),
  Tomato early blight → Septoria (3×).
- **Test-time augmentation experiment** (decided on validation only): original 68.2 %, +flip 67.3 %, +flip+zoom
  68.0 %, +flip+zoom+flipped-zoom 68.8 % — only +0.6 % (3 images) for 4× the compute, so **not used**.
- **Goal vs result:** the target was 70 %+; the final test accuracy is 67.6 %. Reaching 70 % would need a new run
  (stronger backbone such as EfficientNetV2-S / ConvNeXt-Tiny, larger images, or GPU training).

---

## 9. Inference pipeline (`model/classifier.py` and `model/predict.py`)

`diagnose(image, classifier, detector, conf=0.25, with_heatmap=True)`:
1. **Classify** the RGB PIL image → `probs.top5`; keep the top 3 as `{plant, disease, confidence}`.
2. If top-1 confidence < `conf` × 100 → `status: "no_detection"` with a "Not confident enough (best guess X %)"
   message (no history entry).
3. **Leaf box:** detector predicts with `conf=0.10`; the most confident box is used, otherwise the whole image.
4. **Severity:** healthy class → infected 0 %, severity "None". Otherwise `estimate_lesion_pct` in OpenCV HSV
   (hue 0–180) inside the box: plant pixels = saturation ≥ 40, value ≥ 40, hue < 90; lesion pixels = plant pixels
   with hue < 35 (yellow/brown); infected % = lesion / plant × 100 (needs ≥ 100 plant pixels, else 0).
   Severity % = infected %; level: < 10 Low, < 30 Medium, else Severe. Health % = 100 − infected %.
5. **Grad-CAM:** forward hook on `net.model[-2]` (layer 8, C2f, 512 × 7 × 7 at 224 px) keeps the activation and its
   gradient; preprocessing is Ultralytics' own `classify_transforms(imgsz)`; back-propagate the **logit** (the head
   returns `(softmax, logits)` in eval mode) of the predicted class; channel weights = mean gradient per channel;
   CAM = ReLU(Σ weight × activation), normalised to 0–1, cubic-upscaled to 448 × 448, JET colour map, blended
   55 % image / 45 % heatmap over the **centre square** of the photo (the region the classifier sees).
6. **Annotated image:** `draw_box` draws the box in green with the label "Plant - Disease N%" (classifier's label,
   so box and diagnosis always agree); the label is inset ≥ 3 % from the image edges so rounded UI corners never
   clip it.
7. **Treatment:** lookup in `PESTICIDE_RECOMMENDATIONS` (e.g. Early Blight → "Use Mancozeb or Chlorothalonil; remove
   infected lower leaves and rotate crops"; Late Blight → Copper fungicide or Mancozeb; Mosaic Virus → no chemical
   cure, remove plants; Yellow Leaf Curl Virus → control whiteflies; Spider Mites → water spray, insecticidal soap,
   Neem oil or Abamectin; Healthy → no treatment). Unknown → "Consult local agronomist".

`model/predict.py` also contains: `CLASS_INFO` (28 raw labels → (plant, condition), plus a PlantVillage-style
`Plant___Disease` parser), `find_weights` (env `PLANTDOC_WEIGHTS` → `runs/detect/train/weights/best.pt` → best run
by mAP → any `best.pt`), `summarize_detections` (detector-only path) and the **CLI**:
`python model/predict.py --image <file> [--conf 0.25] [--output-dir inference_outputs] [--detector-only]` — uses
the classifier pipeline when `runs/classify/train/weights/best.pt` exists (prints plant, disease, confidence,
severity, health, recommendation, top 3; saves `<name>_predicted.jpg` and `<name>_gradcam.jpg`), otherwise or with
`--detector-only` the M2 detector path. Missing image → "ERROR: Invalid image path".

`find_classifier_weights`: env `PLANTDOC_CLS_WEIGHTS` → `runs/classify/train/weights/best.pt` → None.

---

## 10. Backend API (`app.py`, Flask, 127.0.0.1:5000, `debug=False`)

Startup: sets `YOLO_AUTOINSTALL=false` before importing Ultralytics (stops it pip-installing packages mid-request),
loads `.env` (python-dotenv), loads the detector and classifier once; the server still starts without weights
(predict then returns 503). Strict JSON provider (no NaN/Infinity; numpy → Python types). CORS enabled.
Session key: `FLASK_SECRET_KEY` from `.env`, else a random 64-hex key generated once into `.flask_secret`
(git-ignored). Cookies: HTTP-only, SameSite=Lax.

**Database** (`predictions.db`, SQLite, git-ignored):
- `users(id, created_at, name, email UNIQUE, password_hash)`
- `predictions(id, created_at, plant, disease, confidence_pct, severity_pct, severity_level, health_pct,
  recommendation, user_id → users.id)` — `init_db` adds `user_id` automatically to M2-era databases (old rows kept,
  with no owner, so they appear in nobody's history).

**Endpoints**
- `GET /api/health` → `{status, modelLoaded, classifier, detector, classes, chatbot}` (chatbot = key present).
- `POST /api/predict` body `{imageDataUrl: "data:image/...;base64,...", conf?: number}`.
  `conf` is parsed safely (non-numbers/null → 0.25; clamped to 0–1). Bad or non-image payload → 400
  "Could not read the uploaded file as an image." Response (detected): `{status, plant, disease, isHealthy,
  confidence, top3[], severity, severityLevel, plantHealth, suggestion, bbox[x1,y1,x2,y2], annotatedImage (JPEG data
  URL), heatmapImage (JPEG data URL or null), inferenceMs, saved}`. `saved` is true only for logged-in users and
  detected results. Without a classifier, the M2 detector-only path is used (`top3` empty, no heatmap).
- `GET /api/history` → logged-in user's last 25 scans (newest first); guests get 401 "Log in to see your scan history."
- `POST /api/auth/signup` `{name, email, password}` → 201 `{user}` and logs in. Validation: all required; email
  regex; password ≥ 8 characters; name truncated to 80 chars; email trimmed + lower-cased; duplicate → 409.
  Passwords hashed with Werkzeug `generate_password_hash` (**scrypt**, salted).
- `POST /api/auth/login` `{email, password}` → `{user}`; wrong email or password → 401 "Wrong email or password."
  (same message for both, so login doesn't reveal which emails exist; signup does say "already exists", a normal
  trade-off for a student app). No rate limiting on login yet. `POST /api/auth/logout` clears the session.
  `GET /api/auth/me` → `{user}` or `{user: null}`. All SQL is parameterised (no injection).
- `POST /api/chat` `{message, history?: [{role: "user"|"assistant", text}], context?: {plant, disease, confidence,
  severityLevel}}` → `{reply}`. No key → 503 "not configured"; empty message → 400; > 1000 chars → 400; history
  must be a list (anything else ignored), last 10 turns, each ≤ 2000 chars; context must be an object with
  `disease`. Details in section 11.
- `GET /` and any non-API path → built React app (`DTI/dist`, SPA fallback to `index.html`); unknown `/api/*` → 404 JSON.

---

## 11. AI farming assistant (Gemini)

- SDK: `google-genai` (the current Google Gen AI SDK; the older `google-generativeai` is deprecated).
  Key: `GEMINI_API_KEY` in `.env`. Optional `GEMINI_MODEL` to prefer a specific model.
- **System prompt rules:** only plant/farming topics (otherwise reply exactly "I can only help with plant and
  farming questions."); short, practical, beginner-friendly; cover cause/symptoms/prevention/treatment; prefer
  integrated pest management and organic options first; remind users to follow pesticide labels and ask a local
  agronomist; reply in the user's language (e.g. Hindi); mention that scan results come from an AI that can be wrong.
- **Scan context:** sent as an extra user turn ("Context from my latest leaf scan (AI estimate): plant …,
  condition …, confidence …%, severity …") followed by a model turn "Got it. How can I help?".
- **Model selection and resilience:** order = last model that worked → `GEMINI_MODEL` → defaults
  `gemini-3.5-flash-lite`, `gemini-3.8-flash`, `gemini-3.1-flash-lite`; if all fail, it asks Google for the
  available `gemini-X.Y-flash` / `-flash-lite` models (newest first). Up to 4 attempts, 25 s timeout each. It moves
  to the next model on retired (404 / NOT_FOUND), overloaded (503 / UNAVAILABLE), deadline (504), rate-limit
  (429 / RESOURCE_EXHAUSTED) or timeout errors. Invalid key → 502 "Gemini rejected the API key…"; all attempts
  failed → 503 "The AI assistant is busy right now (Google's servers are overloaded). Please try again in a minute."
  Settings: temperature 0.4, max 700 output tokens.
- **Why this design:** in testing, `gemini-2.5-flash` returned 404 "no longer available to new users", and the
  newest `gemini-3.x-flash` models returned 503 "high demand" / 504 deadline errors for minutes at a time; the lite
  models answered in 1–5 s. Turning "thinking" off was tried but isn't supported by every model, so it isn't used.
- **Verified answers (real key):** tomato late-blight prevention (bulleted, practical), off-topic cricket question
  refused, Potato Early Blight context used with an "AI can be wrong" caveat, a Hindi wheat-rust question answered in
  Hindi, a follow-up ("organic option?", "can I eat the tubers?") answered using the chat history; the fallback
  switched models automatically when one was overloaded.

---

## 12. Frontend (`DTI/`)

React 19, Vite 7, Tailwind CSS 3, Framer Motion, React Router 7, i18next (English strings in `src/i18n/en.json`).
Dark green glass theme (brand #061A13, leafPrimary #10B981, leafSecondary #34D399). The built app (`DTI/dist`) is
served by Flask, so the demo needs no Node.js. It was built from a copy in `PlantDoc_data/dti_build` (keeps
`node_modules` out of OneDrive) with `npx vite build --outDir <project>/DTI/dist`.

- `main.jsx` wraps the app in `I18nextProvider` → `BrowserRouter` → **`AuthProvider`**.
- `layers/services/auth/AuthContext.jsx` *(M3)*: `user`, `loading`, `login`, `signup`, `logout`; loads `/api/auth/me` on start.
- `layers/services/api/`: `apiClient.js` (fetch wrapper; clear errors; errors carry `.status`), `predictionApi.js`,
  `authApi.js` *(M3)*, `chatApi.js` *(M3)*.
- **Pages** (`layers/ui/pages/`):
  - `Landing` — hero, Start Scan / View History, 3 steps (Capture → Diagnose → Advise).
  - `Scan` — webcam (getUserMedia) + frame capture or file upload; images downscaled to ≤ 1024 px JPEG; posts to
    `/api/predict`; stores image + result in localStorage; goes to Results.
  - `Results` — tabs **Detection** (annotated image) / **Heatmap (Grad-CAM)** with an explanation line; plant,
    condition, confidence, severity coloured by level, animated health bar, inference time, "Saved to your history"
    or "Login to save scans"; recommendation; **"Ask the AI assistant"** button; disclaimer; **Top 3 predictions**
    with bars; "Not sure" card for low-confidence results.
  - `Assistant` *(M3)* — chat with Gemini; checkbox "Include my latest scan (Plant - Disease)"; when opened from
    Results the question is pre-filled ("My Potato leaf was diagnosed with Late Blight. What should I do?");
    replies rendered by a small safe formatter (`**bold**`, `*`/`-` bullets, nested bullets; no HTML injection);
    input max 1000 chars; on phones the Send button sits below the input.
  - `History` — guests see "Log in to see your own scan history"; users see cards (total / healthy / diseased) and
    a table (when, plant, condition, confidence, severity).
  - `Login` *(M3)* — log in / create account toggle, client-side checks, then redirect to History; when logged in it
    shows "Signed in as …" and a Log out button.
- `Navbar` (desktop): Home, Scan, Results, Assistant, History + a Login / "👤 FirstName" link. `BottomNav` (phones):
  Home, Scan, Results, Ask AI, History.

---

## 13. Files

```
app.py                              Flask API: predict, history, auth, chat; serves DTI/dist
model/__init__.py
model/predict.py                    Classes, treatments, colour severity, detector path, CLI
model/classifier.py                 (M3) classifier, Grad-CAM, overlay, draw_box, diagnose()
plantdoc_preprocessing.py           EDA + integrity check
generate_yolo_labels_from_folders.py
dataset_preparation.py
prepare_classification_dataset.py   (M3)
train_yolo.py                       detector training
train_classifier.py                 (M3) classifier training with --resume
PlantDoc_YOLOv8_Colab.ipynb         GPU detector notebook (M2)
data.yaml, classes_mapping.txt      dataset config (absolute local path inside data.yaml)
reports/                            EDA (plots, report), training/ (M2 detector), training_cls/ (M3 classifier:
                                    results.png, results.csv, confusion_matrix_normalized.png, val_batch0_pred.jpg,
                                    test_report.txt)
runs/detect/train/weights/best.pt   M2 detector (git-tracked)
runs/classify/train/weights/best.pt M3 classifier (git-tracked)
demo_images/ (+ hard_cases/)        held-out test images for the demo
DTI/src, DTI/dist                   frontend source and build
run_demo.cmd                        Windows launcher
requirements.txt, .env.example, .gitignore
README.md, VIVA_NOTES.md, PROJECT_CONTEXT.md
```

`requirements.txt`: pandas, numpy, Pillow, matplotlib, ultralytics, Flask, flask-cors, python-dotenv,
**google-genai**, **pi-heif**. `.env.example`: `GEMINI_API_KEY=`, optional `GEMINI_MODEL`, `PLANTDOC_CLS_WEIGHTS`,
`PLANTDOC_WEIGHTS`, `FLASK_SECRET_KEY`. Git ignores `.env`, `.flask_secret`, `*.db`, all `.pt` except the two
`best.pt` files, datasets, `node_modules`, caches.

---

## 14. What's new in Milestone 3 and why

| Change | Why |
|---|---|
| **Classifier (YOLOv8s-cls) decides the disease**; detector kept only for the leaf box | Labels are one-per-image, so the task is classification; ImageNet features learn it directly. Test accuracy **39.1 % → 67.6 %** on the same images. |
| `prepare_classification_dataset.py` (same split, hard links, all class folders, short names) | Fair comparison with M2; no extra disk; consistent class indices; Windows path-length limit. |
| `train_classifier.py` with `--resume`, `--fraction` | The laptop shut down mid-training; resuming keeps the optimizer and schedule. `--fraction` was used to benchmark n vs s models. |
| Early-stopping patience raised 15 → 60 after the first interruption | Requested to run all 60 epochs; `best.pt` still keeps the best epoch. |
| **Grad-CAM heatmap** in API, UI and CLI | Explainable AI: shows the model looks at the lesions. |
| **Top-3 predictions** | Shows uncertainty; look-alike diseases are visible to the user. |
| "Not confident" rule now uses the classifier's top-1 (< 25 %) | Keeps the M2 honesty rule after changing models. |
| **Severity = colour-based infected %** (late M2 / M3); no longer averaged with confidence | With full-image boxes, box area was always ~100 % (health ~0 %); averaging confidence made every confident prediction "Severe". |
| Box label drawn with the classifier's diagnosis and inset from edges | Box and text always agree; rounded UI corners were clipping the first letter. |
| **User accounts** (signup/login/logout, scrypt, session cookie) + **per-user history**; guests can scan but nothing is saved | Requested feature; privacy; history makes sense per person. |
| DB migration adds `user_id` automatically | Existing M2 databases keep working. |
| **Gemini assistant** with scan context, history, multilingual, model fallback chain, 25 s timeouts | Requested feature. Fallback added after real tests hit retired (404) and overloaded (503/504) models. |
| Safe formatting of assistant replies | Gemini answers in Markdown; raw `**` looked broken. |
| `parse_conf` for the confidence parameter | `conf: "abc"` or `null` crashed the server (500). |
| Chat ignores non-list `history` | A JSON object as history crashed the server (500). |
| `YOLO_AUTOINSTALL=false`, `pi-heif` added to requirements | Ultralytics silently pip-installed `pi-heif` in the middle of a request (≈20 s freeze, needs internet). |
| `run_demo.cmd` finds a Python that has the packages (venv → `py -3.13/3.12/3.11/3.10` → `python`), saved with CRLF | A newly installed Python 3.12 without packages became the default `python`. |
| Mobile fixes: full-width chat input, Send below it on phones | The fixed-width Send button left almost no space to type. |
| New pages/links: Assistant, Login, "Ask AI" in the phone nav | Access to the new features. |

Bugs found by testing and fixed: the 3 API crashes above, the Gemini default model retired, the fallback picking
the same retired model, the label clipping, the mobile chat layout, raw Markdown in replies, and the runtime
auto-install.

---

## 15. Testing performed (M3)

- **API suite (54 cases, all pass):** health; non-JSON body; missing/garbled/non-base64/non-image payloads;
  `conf` as text, 5, null; images that are plain colour, grayscale, RGBA PNG, palette PNG, 8 × 8, 4000 × 3000 and
  2000 × 40; full response fields; guest scans not saved; top-3 order; health + severity = 100; signup validation
  (missing fields, bad email, short password, non-JSON, numeric password, `<script>` name truncated to 80, email
  normalised, duplicate in another case → 409); history isolation between two users; logout; wrong password,
  unknown email and SQL-injection attempts → 401; scrypt hash stored; chat empty / too long / bad history / bad
  context; unknown API → 404 JSON; all six SPA routes; migration of an M2-schema database.
- **Real chatbot:** 5 question types plus UI flow from Results → Assistant and a follow-up (all answered).
- **Browser (desktop + 375 px phone):** signup, scans of all 11 demo images (9/9 normal demos correct at
  89–100 %, hard cases: corn gray leaf spot now correct, tomato late blight wrong at 42.6 %), Results tabs, heatmap,
  top-3, saved note, Assistant formatting, History (12 rows), logout → 401; no horizontal overflow on any page.
- **CLI:** classifier mode (top-3 + Grad-CAM saved), `--detector-only`, missing file. **Launcher:** starts with
  `py -3.13`, health OK.
- Timing: first scan ~4 s (warm-up), then ~0.4–0.7 s per scan with Grad-CAM on this CPU.

---

## 16. Environment notes

- Windows 11; project in `C:\Users\Naman Sharma\OneDrive\Desktop\AiMl project\Plant-disease-detction-main`.
- Python **3.13** has all packages (torch 2.11 CPU, ultralytics 8.4.39, Flask 3.1, google-genai). A separate
  Python 3.12 is first on PATH without packages — use `run_demo.cmd` or `py -3.13 app.py`.
- Node 22.11 for rebuilding the frontend.
- `.env` holds the real Gemini key (never committed).

## 17. Limitations

1. Test accuracy 67.6 % (target was 70 %): small, imbalanced, hard dataset; CPU-only training at 224 px.
2. Weak classes: soybean, tomato early blight, tomato leaf mold; look-alike blights get confused.
3. Some overfitting (train loss 0.10 vs val loss 1.69).
4. Severity is a colour estimate — misses white symptoms (powdery mildew), viral mottling; brown backgrounds can fool it.
5. Grad-CAM covers only the centre square the classifier sees.
6. The assistant needs internet and depends on Google's model availability; answers can be wrong.
7. Treatment advice is a static table, shown with a disclaimer.
8. Development server (Flask) — fine for demos, not for public deployment.

## 18. Next steps

Stronger backbone (EfficientNetV2-S / ConvNeXt-Tiny) or larger images (320 px), ideally on a GPU, to pass 70 %;
class balancing; PlantDoc object-detection labels or YOLOv8-seg for real lesion areas; ONNX/TFLite export for a
mobile app; weather-based risk tips; Hindi/regional-language UI; production server (waitress/gunicorn).
