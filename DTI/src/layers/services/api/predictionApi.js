import { fetchJson } from "./apiClient";

export async function runPrediction(imageDataUrl, conf = 0.2) {
  return fetchJson("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imageDataUrl, conf }),
  });
}

export async function getPredictionHistory() {
  return fetchJson("/api/history");
}
