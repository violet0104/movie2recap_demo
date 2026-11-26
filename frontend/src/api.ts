export type Upload = { name: string; offset: number; total: number; complete: boolean };
export type TaskKind = "generate" | "edit" | "export";

export type Job = {
  id: string;
  status: "uploading" | "queued" | "running" | "ready" | "error";
  stage: string;
  movie: Upload | null;
  subtitles: string | null;
  source_time: number;
  event_count: number;
  duration: number | null;
  timeline_version: number | null;
  asset_version: number | null;
  error: string | null;
  task: { kind: TaskKind; keep_script?: boolean } | null;
  last_task?: TaskKind;
  stage_progress?: { done: number; total: number } | null;
};

export type SourceRange = { start: number; end: number };
export type Candidate = SourceRange & { id: string; event_id: string };
export type Segment = SourceRange & {
  id: string;
  text: string;
  source: SourceRange;
  candidate_id: string;
  candidates: Candidate[];
  clips: SourceRange[];
  event_ids: string[];
};
export type Timeline = { version: number; duration: number; segments: Segment[] };

class RequestError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) {
    const result = await response.json().catch(() => ({}));
    const detail = result.detail;
    throw new RequestError(response.status, typeof detail === "string" ? detail : detail?.[0]?.msg ?? `请求失败（${response.status}）`);
  }
  return response.json() as Promise<T>;
}

export function json(method: string, body: object): RequestInit {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

export const projectUrl = (id: string) => `/api/projects/${id}`;

export function timecode(seconds: number, precise = false) {
  const ticks = Math.max(0, Math.round(seconds * (precise ? 100 : 1)));
  const whole = precise ? Math.floor(ticks / 100) : ticks;
  const hours = Math.floor(whole / 3600);
  const parts = [Math.floor(whole / 60) % 60, whole % 60];
  if (hours) parts.unshift(hours);
  const text = parts.map((part) => String(part).padStart(2, "0")).join(":");
  return precise ? `${text}.${String(ticks % 100).padStart(2, "0")}` : text;
}

const CHUNK = 8 * 1024 * 1024;

export async function uploadMovie(id: string, movie: File, onProgress: (bytes: number) => void) {
  const base = projectUrl(id);
  let failures = 0;
  while (true) {
    try {
      let state = await request<Partial<Upload> & Pick<Upload, "offset" | "complete">>(`${base}/upload`);
      if (state.name && (state.name !== movie.name || state.total !== movie.size)) {
        throw new RequestError(422, "请选择此项目原先上传的电影文件");
      }
      onProgress(state.offset);
      while (!state.complete) {
        const query = new URLSearchParams({ name: movie.name, total: String(movie.size), offset: String(state.offset) });
        state = await request<Upload>(`${base}/movie?${query}`, {
          method: "PUT",
          body: movie.slice(state.offset, state.offset + CHUNK),
        });
        failures = 0;
        onProgress(state.offset);
      }
      return;
    } catch (error) {
      if (error instanceof RequestError && error.status < 500 && error.status !== 409) throw error;
      if (++failures >= 3) throw error;
      await new Promise((resolve) => setTimeout(resolve, 550));
    }
  }
}
