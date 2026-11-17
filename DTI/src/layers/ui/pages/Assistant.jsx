import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import Button from "../components/Button";
import { sendChat } from "../../services/api/chatApi";

function readScanContext() {
  try {
    const raw = localStorage.getItem("leafx-result");
    const r = raw ? JSON.parse(raw) : null;
    if (r?.status !== "detected") return null;
    return { plant: r.plant, disease: r.disease, confidence: r.confidence, severityLevel: r.severityLevel };
  } catch {
    return null;
  }
}

// Render the small subset of Markdown Gemini uses (**bold**, "* " / "- " bullets) without injecting HTML.
function renderInline(text) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") && part.length > 4 ? (
      <strong key={i} className="text-textSoft">{part.slice(2, -2)}</strong>
    ) : (
      part.replace(/\*([^*\s][^*]*)\*/g, "$1")
    )
  );
}

function FormattedReply({ text }) {
  return (
    <div className="space-y-1">
      {text.split("\n").map((line, i) => {
        const bullet = line.match(/^(\s*)[*-]\s+(.*)$/);
        if (bullet) {
          const nested = bullet[1].length >= 2;
          return (
            <div key={i} className={`flex gap-2 ${nested ? "pl-5" : ""}`}>
              <span className="text-leafPrimary">•</span>
              <span>{renderInline(bullet[2])}</span>
            </div>
          );
        }
        return line.trim() ? <p key={i}>{renderInline(line)}</p> : <div key={i} className="h-1" />;
      })}
    </div>
  );
}

function Assistant() {
  const { t } = useTranslation();
  const location = useLocation();
  const [useScan, setUseScan] = useState(Boolean(location.state?.fromResults));
  const [scan] = useState(readScanContext);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  useEffect(() => {
    if (location.state?.fromResults && scan && messages.length === 0) {
      setInput(t("chat_prefill", { disease: scan.disease, plant: scan.plant }));
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const send = async (event) => {
    event?.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    const history = messages;
    setMessages([...history, { role: "user", text }]);
    setInput("");
    setError("");
    setBusy(true);
    try {
      const reply = await sendChat(text, history, useScan ? scan : null);
      setMessages((m) => [...m, { role: "assistant", text: reply }]);
    } catch (err) {
      setError(err?.message || "The assistant could not answer.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="page-shell max-w-3xl space-y-4">
      <h2 className="text-3xl font-bold">{t("assistant_title")}</h2>
      <p className="text-sm text-slate-400">{t("assistant_subtitle")}</p>

      {scan && (
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input type="checkbox" checked={useScan} onChange={(e) => setUseScan(e.target.checked)} />
          {t("use_scan_context", { plant: scan.plant, disease: scan.disease })}
        </label>
      )}

      <div className="glass-panel h-[55vh] space-y-3 overflow-y-auto rounded-2xl p-4">
        {messages.length === 0 && <p className="text-sm text-slate-400">{t("chat_empty")}</p>}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-2 text-sm leading-relaxed ${
                m.role === "user" ? "whitespace-pre-wrap bg-leafPrimary/20 text-textSoft" : "bg-white/5 text-slate-200"
              }`}
            >
              {m.role === "assistant" ? <FormattedReply text={m.text} /> : m.text}
            </div>
          </div>
        ))}
        {busy && <p className="text-sm text-slate-400">{t("chat_thinking")}</p>}
        <div ref={endRef} />
      </div>

      {error && <p className="text-sm text-rose-400">{error}</p>}
      <form onSubmit={send} className="flex flex-col gap-2 sm:flex-row">
        <input
          className="input-shell min-w-0 flex-1 pt-3"
          value={input}
          maxLength={1000}
          placeholder={t("chat_placeholder")}
          onChange={(e) => setInput(e.target.value)}
        />
        <Button type="submit" disabled={busy || !input.trim()} className="disabled:opacity-60">
          {t("chat_send")}
        </Button>
      </form>
      <p className="text-xs text-slate-500">{t("chat_disclaimer")}</p>
    </section>
  );
}

export default Assistant;
