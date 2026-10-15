import type { Metrics, ObjectView, Snapshot } from "./types";

async function parse<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(body?.error?.message ?? `${res.status} ${res.statusText}`);
  return body as T;
}

const post = (url: string, body?: unknown) =>
  fetch(url, {
    method: "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const api = {
  samples: () => fetch("/api/samples").then((r) => parse<{ id: string; label: string }[]>(r)),
  createFromSample: (sample: string) =>
    post("/api/scenes", { sample }).then((r) => parse<{ scene_id: string; version: string }>(r)),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch("/api/scenes/upload", { method: "POST", body: form }).then((r) =>
      parse<{ scene_id: string; version: string }>(r),
    );
  },
  snapshot: (sceneId: string) => fetch(`/api/scenes/${sceneId}`).then((r) => parse<Snapshot>(r)),
  startRun: (sceneId: string) => post(`/api/scenes/${sceneId}/runs`).then((r) => parse<{ run_id: string }>(r)),
  fix: (sceneId: string, name: string, fixId: string) =>
    post(`/api/scenes/${sceneId}/objects/${encodeURIComponent(name)}/fix`, { fix_id: fixId }).then((r) =>
      parse<{ run_id: string; cached: boolean }>(r),
    ),
  review: (sceneId: string, name: string, decision: "approve" | "reject") =>
    post(`/api/scenes/${sceneId}/objects/${encodeURIComponent(name)}/review`, { decision }).then((r) =>
      parse<{ object: ObjectView; metrics: Metrics }>(r),
    ),
};
