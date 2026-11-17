# Milestone 3 — Viva Notes & Demo Guide

Project: **AgroNiddan (LeafX) — AI Plant Disease Detection from Leaf Images**

---

## 1. Demo script (6–7 minutes)

**Before the viva:** double-click `run_demo.cmd`; the browser opens http://127.0.0.1:5000 after ~8 s (the first
scan takes ~4 s to warm up, then ~0.5 s). Make sure `.env` has `GEMINI_API_KEY` and the laptop is online (needed
only for the assistant). Test leaves are in `demo_images/` — taken from the **held-out test split**, never used in
training; the final model gets all 9 right. `demo_images/hard_cases/` has two harder ones (one now correct, one
still wrong at low confidence) for discussing limitations.

1. **Problem (30 s)** — Home page: late-detected leaf diseases cost farmers yield; experts are scarce.
2. **Data (45 s)** — `reports/plantdoc_class_distribution.png`, `plantdoc_class_imbalance.png`: 28 classes,
   2,572 real field images, imbalanced, 12 bad files found by the integrity check.
3. **Model improvement (1 min)** — `reports/training_cls/results.png`: M2 detector **39.1 %** → M3 classifier
   **67.6 %** test accuracy (top-5 **95.7 %**). Show the confusion matrix (`confusion_matrix_normalized.png`).
4. **Log in (20 s)** — Login → create an account (shows the auth feature).
5. **Live scan (1.5 min)** — Scan → Upload → Scan Now → Results: plant, disease, confidence, severity, health bar,
   top-3, treatment, "Saved to your history".
6. **Explainable AI (45 s)** — click **Heatmap (Grad-CAM)**: red = where the model looked (the lesions).
7. **AI assistant (1 min)** — "Ask the AI assistant": question is pre-filled from the scan; ask a follow-up
   (e.g. "Can I still eat the tubers?") or a question in Hindi.
8. **History + privacy (20 s)** — History shows only this user's scans; log out → History asks you to log in.
9. **Honesty check (20 s)** — upload a non-leaf picture → "Not sure" instead of a made-up disease.

Backup if the browser fails: `python model/predict.py --image demo_images/<file>.jpg` (prints the diagnosis and
saves the annotated image + Grad-CAM heatmap).

---

## 2. M1 — Problem domain

- **Domain:** precision agriculture / computer vision for crop health.
- **Problem:** visual diagnosis of leaf diseases is slow, needs experts, and errors lead to wrong or excessive pesticide use.
- **Users:** small farmers, home gardeners, agri-extension workers with a phone camera.
- **Input → output:** leaf photo → plant + condition (28 classes) + confidence + severity + treatment + explanation.
- **Why hard:** cluttered field photos, multiple leaves, varying light, look-alike diseases, small imbalanced classes.

## 3. M2 — Existing systems & feasibility

| Existing approach | Limitation | What we do |
|---|---|---|
| Manual inspection by experts | slow, not scalable | instant, any browser |
| CNNs on **PlantVillage** (≈99 % in the lab, Mohanty et al. 2016) | plain backgrounds; collapse on field photos | trained on **PlantDoc** real field images |
| Commercial apps (Plantix) | closed, black box | open pipeline + Grad-CAM explanation |
| Plain classifiers | no location / severity | classifier + detector + colour severity |

Feasibility: open-source stack, transfer learning, trains on a laptop CPU, free Gemini tier, runs on any browser.

## 4. Objectives & methodology

1. Reproducible data pipeline — `plantdoc_preprocessing.py`, `generate_yolo_labels_from_folders.py`, `dataset_preparation.py` (seed-42 70/20/10 split, resize ≤640 px).
2. M2 baseline detector — `train_yolo.py` (YOLOv8n, 15 epochs, 256 px, frozen backbone) → 39.1 %.
3. **M3 classifier** — `prepare_classification_dataset.py` (same split, class folders, hard links) → `train_classifier.py` (YOLOv8s-cls, ImageNet-pretrained, 60 epochs, 224 px) → 67.6 %.
4. Post-processing — `model/classifier.py` `diagnose()`: classifier + detector box + HSV severity + Grad-CAM + treatment.
5. Delivery — `app.py` (REST API, accounts, chat) + React UI (`DTI/src`).

## 5. Algorithms / techniques

- **Why we switched to a classifier (M3):** PlantDoc gives one label per image, so our detector boxes were the whole image — the task is really *classification*. A classifier learns it directly from strong ImageNet features: 39.1 % → 67.6 % on the same test images.
- **YOLOv8s-cls:** CSPDarknet backbone (Conv + C2f blocks) → classification head (1×1 conv to 1280 channels → global average pool → linear 28-way). Loss: cross-entropy. 5.1 M parameters, 12.5 GFLOPs at 224 px.
- **Training details:** AdamW (lr 0.000313, auto-selected), 3 warm-up epochs, RandAugment + random erasing (0.4) + horizontal flips, batch 32, 224 px, best epoch kept by validation top-1 (`best.pt` = epoch 57).
- **Detector (kept from M2):** YOLOv8n — anchor-free decoupled head, CIoU + DFL box loss, NMS; now only used to find the leaf box.
- **Grad-CAM (explainable AI):** take the feature maps of the last backbone block (layer 8, 512 × 7 × 7), weight each channel by the average gradient of the predicted class's score, sum, ReLU, normalise, upscale, colour (red = most important).
- **Severity:** HSV colour thresholds inside the leaf box — infected % = yellow/brown pixels ÷ plant pixels → Low (<10), Medium (<30), Severe; health = 100 − infected %.
- **Metrics:** top-1 accuracy = share of images whose top prediction is right; top-5 = right answer among the 5 best guesses. (Detector: precision, recall, IoU, mAP@50.)
- **Assistant:** Google Gemini via the `google-genai` SDK, a system prompt restricting it to farming topics, scan context, last 10 messages of history, automatic fallback across models.
- **Security:** scrypt password hashing (Werkzeug), signed HTTP-only session cookies, per-user data isolation, parameterised SQL (no injection).

## 6. Design ↔ implementation sync

| Design block | Implemented in |
|---|---|
| Data analysis / pipeline | `plantdoc_preprocessing.py`, `generate_yolo_labels_from_folders.py`, `dataset_preparation.py`, `prepare_classification_dataset.py` |
| Training | `train_yolo.py` (detector), `train_classifier.py` (classifier, `--resume`) |
| Diagnosis | `model/classifier.py` (`diagnose`, `grad_cam`), `model/predict.py` (classes, severity, advice, CLI) |
| API | `app.py`: `/api/health`, `/api/predict`, `/api/history`, `/api/auth/*`, `/api/chat` |
| Storage | SQLite `predictions.db`: `users`, `predictions` (with `user_id`) |
| UI | `DTI/src/layers/ui/pages/`: Landing, Scan, Results, Assistant, History, Login |

---

## 7. Likely viva questions

- **What are your results?** Test (256 unseen images): **67.6 % top-1, 95.7 % top-5**. Validation best 68.0 % (epoch 57). M2 detector: 39.1 %.
- **Why not 70 %+?** PlantDoc is hard — in the PlantDoc paper the best fine-tuned model reached ≈70 %; we're close on a laptop CPU. Accuracy plateaued around 66–68 % from epoch ~25. Next: stronger backbone (EfficientNetV2 / ConvNeXt), larger images (320 px), GPU training.
- **Did you try anything else?** Test-time augmentation (averaging flipped/zoomed views) — only +0.6 % on validation and 4× slower, so we didn't use it.
- **Where does it fail?** Soybean (0/4), tomato early blight and leaf mold (1/6). Most common confusion: potato early blight → late blight (5×) — visually similar.
- **Overfitting?** Train loss fell to 0.10 while validation loss stayed ≈1.7 — some overfitting; augmentation and keeping the best validation epoch limit its effect.
- **How did you keep the evaluation fair?** Same seed-42 split as M2; model chosen on validation only; test set scored once at the end.
- **What is Grad-CAM showing?** Which image regions increased the predicted class's score — on diseased leaves it highlights lesions, so the model is looking at the right thing.
- **Training was interrupted — how?** The laptop shut down twice; Ultralytics checkpoints every epoch (`last.pt`), so we resumed from epoch 17 with the same optimizer state (`train_classifier.py --resume`).
- **How are passwords stored?** Never in plain text: scrypt hashes (salted, slow to brute-force).
- **Why might the chatbot fail?** It needs internet and a Gemini key; if a model is overloaded or retired, the app automatically tries other models and shows a clear message if all fail.
- **What happens on a random photo?** If the top prediction is below 25 % confidence, the app says it isn't sure.
- **Why PlantDoc and not PlantVillage?** PlantVillage is lab images (~99 % but useless on real photos); PlantDoc is real field data.

---

## 8. Team contribution (fill in before the viva)

| Member | Part owned | Files |
|---|---|---|
| _Name 1_ | Dataset analysis & pipeline | `plantdoc_preprocessing.py`, `generate_yolo_labels_from_folders.py`, `dataset_preparation.py`, `prepare_classification_dataset.py` |
| _Name 2_ | Model training & evaluation, Grad-CAM | `train_yolo.py`, `train_classifier.py`, `model/classifier.py`, `reports/` |
| _Name 3_ | Backend API, accounts, chatbot | `app.py`, `model/predict.py` |
| _Name 4_ | Frontend | `DTI/src/` |

Each member should be able to run and explain their own part.

---

## 9. Next steps

Stronger backbone / larger images on GPU (target > 70 %) · lesion segmentation for true severity · more data
(PlantVillage) · ONNX/TFLite export for a mobile app · weather-based disease-risk tips · Hindi and regional-language UI.
