import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import Button from "../components/Button";
import { useAuth } from "../../services/auth/AuthContext";

function Login() {
  const { t } = useTranslation();
  const { user, login, signup, logout } = useAuth();
  const navigate = useNavigate();
  const [mode, setMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      if (mode === "signup") await signup(name, email, password);
      else await login(email, password);
      navigate("/history");
    } catch (err) {
      setError(err?.message || "Something went wrong");
    } finally {
      setBusy(false);
    }
  };

  if (user) {
    return (
      <section className="page-shell max-w-md space-y-4">
        <h2 className="text-3xl font-bold">{t("account_title")}</h2>
        <div className="glass-panel space-y-4 rounded-2xl p-6">
          <p className="text-slate-300">
            {t("signed_in_as")} <span className="text-leafSecondary">{user.name}</span> ({user.email})
          </p>
          <Button variant="secondary" onClick={logout}>{t("logout")}</Button>
        </div>
      </section>
    );
  }

  return (
    <section className="page-shell max-w-md space-y-4">
      <h2 className="text-3xl font-bold">{mode === "signup" ? t("create_account") : t("login")}</h2>
      <form onSubmit={submit} className="glass-panel space-y-4 rounded-2xl p-6">
        {mode === "signup" && (
          <label className="block space-y-1 text-sm text-slate-300">
            <span>{t("full_name")}</span>
            <input className="input-shell pt-3" value={name} onChange={(e) => setName(e.target.value)} required />
          </label>
        )}
        <label className="block space-y-1 text-sm text-slate-300">
          <span>{t("email")}</span>
          <input
            type="email"
            className="input-shell pt-3"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
            required
          />
        </label>
        <label className="block space-y-1 text-sm text-slate-300">
          <span>{t("password")}</span>
          <input
            type="password"
            className="input-shell pt-3"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === "signup" ? "new-password" : "current-password"}
            minLength={mode === "signup" ? 8 : undefined}
            required
          />
        </label>
        {mode === "signup" && <p className="text-xs text-slate-400">{t("password_hint")}</p>}
        {error && <p className="text-sm text-rose-400">{error}</p>}
        <Button type="submit" disabled={busy} className="w-full disabled:opacity-60">
          {busy ? "…" : mode === "signup" ? t("create_account") : t("login")}
        </Button>
        <button
          type="button"
          className="w-full text-center text-sm text-leafSecondary hover:underline"
          onClick={() => {
            setMode(mode === "signup" ? "login" : "signup");
            setError("");
          }}
        >
          {mode === "signup" ? t("already_account") : t("need_account")}
        </button>
      </form>
    </section>
  );
}

export default Login;
