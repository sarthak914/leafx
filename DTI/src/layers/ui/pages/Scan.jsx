import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import Button from "../components/Button";
import ScanOverlay from "../../ai/components/ScanOverlay";
import { runPrediction } from "../../services/api/predictionApi";

const MAX_SIDE = 1024;

// Shrink large photos so the upload and the localStorage copy stay small.
function downscale(dataUrl) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/jpeg", 0.9));
    };
    img.onerror = () => resolve(dataUrl);
    img.src = dataUrl;
  });
}

function Scan() {
  const { t } = useTranslation();
  const [image, setImage] = useState("");
  const [scanning, setScanning] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [error, setError] = useState("");
  const videoRef = useRef(null);
  const navigate = useNavigate();

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraOn(true);
      }
    } catch {
      setCameraOn(false);
    }
  };

  const captureFrame = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth || 1280;
    canvas.height = videoRef.current.videoHeight || 720;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    downscale(canvas.toDataURL("image/jpeg", 0.9)).then(setImage);
  };

  const runScan = () => {
    if (!image) return;
    setError("");
    setScanning(true);
    runPrediction(image)
      .then((prediction) => {
        localStorage.setItem("leafx-image", image);
        localStorage.setItem("leafx-result", JSON.stringify(prediction));
        navigate("/results");
      })
      .catch((err) => {
        setError(err?.message || "Failed to run prediction");
      })
      .finally(() => {
        setScanning(false);
      });
  };

  return (
    <section className="page-shell space-y-6">
      <h2 className="text-3xl font-bold">{t("scan_title")}</h2>
      <div className="glass-panel rounded-2xl p-4">
        <div className="relative overflow-hidden rounded-xl border border-leafPrimary/40 bg-black/30">
          {!image ? (
            <div className="aspect-video">
              <video ref={videoRef} autoPlay playsInline className="h-full w-full object-cover" />
            </div>
          ) : (
            <img src={image} alt={t("captured_leaf_alt")} className="aspect-video w-full object-cover" />
          )}
          {scanning && <ScanOverlay />}
        </div>
      </div>

      <div className="flex flex-wrap gap-3">
        <Button onClick={startCamera}>{t("enable_camera")}</Button>
        {cameraOn && <Button variant="secondary" onClick={captureFrame}>{t("capture_frame")}</Button>}
        <label className="rounded-full border border-white/20 bg-white/[0.04] px-5 py-3 text-sm text-slate-200">
          {t("upload_image")}
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (!file) return;
              const reader = new FileReader();
              reader.onload = () => downscale(String(reader.result)).then(setImage);
              reader.readAsDataURL(file);
            }}
          />
        </label>
        <Button onClick={runScan} className="disabled:opacity-60" disabled={!image || scanning}>
          {scanning ? t("scanning") : t("scan_now")}
        </Button>
      </div>
      <p className="text-xs text-slate-400">{t("scan_hint")}</p>
      {error && <p className="text-sm text-rose-400">{error}</p>}
    </section>
  );
}

export default Scan;
