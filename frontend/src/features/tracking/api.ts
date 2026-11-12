import type { BallType, Calibration, FirstFrameResponse, JobListItem, JobRecord, TrackingResult } from "./types";
import { getClientId } from "../../lib/client-id";

const API_BASE = (import.meta.env?.VITE_API_BASE as string) || "/api";

function authHeaders(): Record<string, string> {
  return { "X-Client-Id": getClientId() };
}

async function jsonOrError<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail: { code?: string; message?: string } = {};
    try {
      const body = await res.json();
      detail = body?.detail || body;
    } catch {/* ignore */}
    const err: Error & { code?: string } = new Error(detail.message || res.statusText);
    err.code = detail.code || `HTTP_${res.status}`;
    throw err;
  }
  return res.json();
}

export interface FirstFrameProgress {
  loaded: number;
  total: number;
}

export function uploadFirstFrame(
  file: File,
  onProgress?: (p: FirstFrameProgress) => void,
  signal?: AbortSignal,
): Promise<FirstFrameResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/track/first-frame`);
    xhr.setRequestHeader("X-Client-Id", getClientId());
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress({ loaded: e.loaded, total: e.total });
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (e) { reject(e); }
      } else {
        let code = `HTTP_${xhr.status}`;
        let message = xhr.statusText;
        try {
          const body = JSON.parse(xhr.responseText);
          code = body?.detail?.code || code;
          message = body?.detail?.message || message;
        } catch {/* ignore */}
        const err: Error & { code?: string } = new Error(message);
        err.code = code;
        reject(err);
      }
    };
    xhr.onerror = () => reject(new Error("network error"));
    if (signal) {
      signal.addEventListener("abort", () => xhr.abort());
    }
    const form = new FormData();
    form.append("video", file);
    xhr.send(form);
  });
}

export async function analyze(
  uploadId: string,
  calibration: Calibration,
  ballType: BallType,
  filename: string,
): Promise<{ job_id: string; status: string }> {
  const form = new FormData();
  form.append("upload_id", uploadId);
  form.append("calibration", JSON.stringify(calibration));
  form.append("ball_type", ballType);
  form.append("filename", filename);
  const res = await fetch(`${API_BASE}/track/analyze`, {
    method: "POST",
    body: form,
    headers: authHeaders(),
  });
  return jsonOrError(res);
}

export async function suggestCalibration(uploadId: string): Promise<{ calibration: Calibration }> {
  const res = await fetch(`${API_BASE}/track/uploads/${uploadId}/suggest`, { headers: authHeaders() });
  return jsonOrError(res);
}

export async function getJob(jobId: string, signal?: AbortSignal): Promise<JobRecord> {
  const res = await fetch(`${API_BASE}/track/jobs/${jobId}`, { signal, headers: authHeaders() });
  return jsonOrError(res);
}

export async function getResult(jobId: string, signal?: AbortSignal): Promise<TrackingResult> {
  const res = await fetch(`${API_BASE}/track/jobs/${jobId}/result`, { signal, headers: authHeaders() });
  return jsonOrError(res);
}

export async function listJobs(limit = 20): Promise<{ jobs: JobListItem[] }> {
  const res = await fetch(`${API_BASE}/track/jobs?limit=${limit}`, { headers: authHeaders() });
  return jsonOrError(res);
}

export async function getLiveFeed(
  limit = 30,
  signal?: AbortSignal,
): Promise<{ wickets: import("./types").LiveWicket[] }> {
  // Open endpoint (no client scoping) — this is an operator-wide feed, and the
  // media <img>/<video> tags can't send custom headers anyway.
  const res = await fetch(`${API_BASE}/track/live/feed?limit=${limit}`, { signal });
  return jsonOrError(res);
}

export async function cancelJob(jobId: string): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/track/jobs/${jobId}/cancel`, {
    method: "POST",
    headers: authHeaders(),
  });
  return jsonOrError(res);
}

export async function getFeedback(
  jobId: string,
): Promise<import("./types").DeliveryFeedback | null> {
  const res = await fetch(`${API_BASE}/track/jobs/${jobId}/feedback`, { headers: authHeaders() });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error("feedback fetch failed");
  return res.json();
}

export async function submitFeedback(
  jobId: string,
  feedback: import("./types").DeliveryFeedback,
): Promise<void> {
  const res = await fetch(`${API_BASE}/track/jobs/${jobId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(feedback),
  });
  if (!res.ok) throw new Error("feedback submit failed");
}
