import { fetchJson } from "./apiClient";

/**
 * Ask the Gemini farming assistant. `history` is [{ role: "user" | "assistant", text }];
 * `context` is the latest scan result (optional).
 */
export async function sendChat(message, history = [], context = null) {
  const data = await fetchJson("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, context }),
  });
  return data.reply || "";
}
