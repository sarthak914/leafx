import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { useTranslation } from "react-i18next";
import Button from "../components/Button";

const severityTone = {
  None: "text-leafSecondary",
  Low: "text-leafSecondary",
  Medium: "text-amber-300",
  Severe: "text-rose-400",
};

function Results() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [image, setImage] = useState("");
  const [result, setResult] = useState(null);
  const [view, setView] = useState("boxes");

  useEffect(() => {
    setImage(localStorage.getItem("leafx-image") || "");
    const raw = localStorage.getItem("leafx-result");
    setResult(raw ? JSON.parse(raw) : null);
  }, []);

  const healthScale = useMemo(() => {
    if (!result || result.status !== "detected") return 0;
    return result.plantHealth / 100;
  }, [result]);

  if (!image || !result) {
    return (
      <section className="page-shell space-y-6">
        <h2 className="text-3xl font-bold">{t("results_title")}</h2>
        <div className="glass-panel rounded-2xl p-6 text-slate-300">{t("results_empty")}</div>
      </section>
    );
  }

  const detected = result.status === "detected";
  const showHeatmap = view === "heatmap" && result.heatmapImage;

  return (
    <section className="page-shell space-y-6">
      <h2 className="text-3xl font-bold">{t("results_title")}</h2>
      <div className="glass-panel space-y-3 rounded-2xl p-4">
        {result.heatmapImage && (
          <div className="flex gap-2">
            {["boxes", "heatmap"].map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setView(v)}
                className={`rounded-full px-4 py-1.5 text-xs tracking-wider ${
                  view === v ? "bg-leafPrimary/20 text-leafPrimary" : "text-slate-300 hover:bg-white/5"
                }`}
              >
                {v === "boxes" ? t("view_detection") : t("view_heatmap")}
              </button>
            ))}
          </div>
        )}
        <div className="relative overflow-hidden rounded-xl border border-leafPrimary/40">
          <img
            src={showHeatmap ? result.heatmapImage : result.annotatedImage || image}
            alt={t("leaf_scan_alt")}
            className="max-h-[480px] w-full object-contain"
          />
        </div>
        {showHeatmap && <p className="text-xs text-slate-400">{t("heatmap_explain")}</p>}
      </div>

      {!detected ? (
        <div className="premium-card space-y-2">
          <p className="text-lg font-semibold text-amber-300">{t("no_detection_title")}</p>
          <p className="text-sm text-slate-300">{result.message}</p>
        </div>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="premium-card space-y-2">
              <p className="text-sm text-slate-300">
                {t("plant_type")}: <span className="text-textSoft">{result.plant}</span>
              </p>
              <p className="text-sm text-slate-300">
                {t("disease_name")}: <span className="text-leafSecondary">{result.disease}</span>
              </p>
              <p className="text-sm text-slate-300">
                {t("confidence")}: <span className="text-textSoft">{Number(result.confidence || 0).toFixed(2)}%</span>
              </p>
              <p className="text-sm text-slate-300">
                {t("severity")}:{" "}
                <span className={severityTone[result.severityLevel] || "text-textSoft"}>
                  {result.severityLevel} ({Number(result.severity || 0).toFixed(2)}%)
                </span>
              </p>
              <p className="text-sm text-slate-300">
                {t("plant_health")}: <span className="text-textSoft">{Number(result.plantHealth || 0).toFixed(2)}%</span>
              </p>
              <div className="pt-3">
                <div className="h-3 overflow-hidden rounded-full bg-white/10">
                  <motion.div
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: healthScale }}
                    className="h-full origin-left bg-gradient-to-r from-leafPrimary to-leafSecondary"
                  />
                </div>
              </div>
              {result.inferenceMs != null && (
                <p className="pt-2 text-xs text-slate-500">
                  {t("inference_time")}: {Number(result.inferenceMs).toFixed(0)} ms
                </p>
              )}
              <p className="text-xs text-slate-500">
                {result.saved ? (
                  t("saved_to_history")
                ) : (
                  <>
                    <Link to="/login" className="text-leafSecondary hover:underline">{t("login")}</Link>{" "}
                    {t("login_to_save")}
                  </>
                )}
              </p>
            </div>
            <div className="premium-card space-y-3">
              <p className="text-xs tracking-[0.2em] text-leafPrimary">{t("recommendation")}</p>
              <p className="text-sm leading-relaxed text-slate-200">{result.suggestion}</p>
              <Button onClick={() => navigate("/assistant", { state: { fromResults: true } })}>
                {t("ask_assistant")}
              </Button>
              <p className="border-t border-white/10 pt-3 text-xs text-slate-500">{t("disclaimer")}</p>
            </div>
          </div>

          {result.top3?.length > 1 && (
            <div className="premium-card">
              <p className="text-xs tracking-[0.2em] text-leafPrimary">{t("top_predictions")}</p>
              <ul className="mt-3 space-y-2 text-sm text-slate-300">
                {result.top3.map((p, i) => (
                  <li key={i} className="space-y-1">
                    <div className="flex justify-between">
                      <span>{p.plant} - {p.disease}</span>
                      <span>{Number(p.confidence).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
                      <div className="h-full bg-leafPrimary/70" style={{ width: `${Math.min(100, p.confidence)}%` }} />
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </section>
  );
}

export default Results;
