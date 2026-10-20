import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import Card from "../components/Card";
import { getPredictionHistory } from "../../services/api/predictionApi";

function History() {
  const { t } = useTranslation();
  const [rows, setRows] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getPredictionHistory()
      .then(setRows)
      .catch((err) => setError(err?.message || "Failed to load history"));
  }, []);

  const healthy = rows ? rows.filter((r) => r.disease === "Healthy").length : 0;

  return (
    <section className="page-shell space-y-6">
      <h2 className="text-3xl font-bold">{t("history_title")}</h2>
      {error && <p className="text-sm text-rose-400">{error}</p>}
      {!rows && !error && <p className="text-sm text-slate-400">{t("history_loading")}</p>}

      {rows && (
        <>
          <div className="grid gap-4 sm:grid-cols-3">
            <Card title={t("total_scans")} value={rows.length} subtitle="last 25" />
            <Card title={t("healthy_scans")} value={healthy} subtitle="" />
            <Card title={t("diseased_scans")} value={rows.length - healthy} subtitle="" />
          </div>

          {rows.length === 0 ? (
            <div className="glass-panel rounded-2xl p-6 text-slate-300">{t("history_empty")}</div>
          ) : (
            <div className="glass-panel overflow-x-auto rounded-2xl p-4">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-wider text-slate-400">
                  <tr>
                    <th className="p-2">{t("history_when")}</th>
                    <th className="p-2">{t("plant_type")}</th>
                    <th className="p-2">{t("disease_name")}</th>
                    <th className="p-2">{t("confidence")}</th>
                    <th className="p-2">{t("severity")}</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.id} className="border-t border-white/10 text-slate-200">
                      <td className="p-2">{new Date(r.createdAt).toLocaleString()}</td>
                      <td className="p-2">{r.plant}</td>
                      <td className="p-2">{r.disease}</td>
                      <td className="p-2">{Number(r.confidence).toFixed(1)}%</td>
                      <td className="p-2">{r.severityLevel}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </section>
  );
}

export default History;
