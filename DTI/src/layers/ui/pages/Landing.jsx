import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import Button from "../components/Button";

function Landing() {
  const { t } = useTranslation();
  const steps = [
    { title: t("step1_title"), text: t("step1_text") },
    { title: t("step2_title"), text: t("step2_text") },
    { title: t("step3_title"), text: t("step3_text") },
  ];

  return (
    <section className="page-shell space-y-8">
      <div className="glass-panel relative overflow-hidden rounded-3xl p-8 sm:p-12">
        <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(120deg,rgba(0,255,156,0.08),rgba(0,207,255,0.08),rgba(0,255,156,0.08))]" />
        <div className="relative z-10 text-center">
          <p className="text-xs tracking-[0.24em] text-leafPrimary">{t("landing_badge")}</p>
          <h1 className="mt-4 text-4xl font-extrabold tracking-tight sm:text-5xl lg:text-6xl">{t("landing_title")}</h1>
          <p className="mx-auto mt-4 max-w-2xl text-sm text-slate-300 sm:text-base">{t("landing_subtitle")}</p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <Link to="/scan"><Button>{t("start_scan")}</Button></Link>
            <Link to="/history"><Button variant="secondary">{t("view_history")}</Button></Link>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {steps.map((step) => (
          <article key={step.title} className="premium-card">
            <p className="text-sm font-semibold text-leafSecondary">{step.title}</p>
            <p className="mt-2 text-sm text-slate-300">{step.text}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

export default Landing;
