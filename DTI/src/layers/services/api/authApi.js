import { fetchJson } from "./apiClient";

const post = (path, body) =>
  fetchJson(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });

export const getCurrentUser = () => fetchJson("/api/auth/me").then((d) => d.user);
export const login = (email, password) => post("/api/auth/login", { email, password }).then((d) => d.user);
export const signup = (name, email, password) =>
  post("/api/auth/signup", { name, email, password }).then((d) => d.user);
export const logout = () => post("/api/auth/logout");
