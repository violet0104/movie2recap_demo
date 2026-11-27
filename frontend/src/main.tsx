import { useEffect, useRef, useState, type FormEvent } from "react";
import { createRoot } from "react-dom/client";
import { json, projectUrl, request, timecode, uploadMovie, type Candidate, type Job, type Segment, type TaskKind, type Timeline } from "./api";
import "./style.css";

const stages: Record<string, string> = {
  queued: "等待任务进程",
  probe: "读取原片与字幕",
  events: "生成分段描述",
  scoring: "DeepSeek 评分",
  selection: "按 8% 预算选段",
  script: "撰写英文解说",
  audio: "合成旁白与对齐画面",
  render: "渲染镜头",
  encode: "烧录字幕并导出",
  editing: "更新段落与成片时间码",
};

const taskStages: Record<TaskKind, { title: string; steps: string[] }> = {
  generate: { title: "生成进度", steps: ["probe", "events", "scoring", "selection", "script", "audio", "render", "encode"] },
  edit: { title: "修改进度", steps: ["editing"] },
  export: { title: "导出进度", steps: ["render", "encode"] },
};
const progressUnits: Record<string, string> = { script: "段", audio: "段", render: "个镜头" };

function ProgressRow({ title, value, max = 1, detail, failed }: {
  title: string;
  value?: number;
  max?: number;
  detail?: string;
  failed?: boolean;
}) {
  const percent = value === undefined ? "处理中" : `${Math.floor(value / max * 100)}%`;
  return <div className={`progress-row${failed ? " failed" : ""}`}>
    <div><span>{title}</span><span className="mono">{detail ?? percent}</span></div>
    <progress aria-label={title} aria-valuetext={detail || undefined} value={value} max={max} />
  </div>;
}

function TaskProgress({ job }: { job: Job }) {
  const { title, steps } = taskStages[job.task?.kind ?? job.last_task ?? "generate"];
  const current = steps.indexOf(job.stage);
  const ready = job.status === "ready";
  return <section className="panel task-progress" aria-label={title}>
    <h2>{title}</h2>
    <div className="progress-grid">
      {steps.map((step, index) => {
        if (ready || index < current) {
          const detail = step === "events" ? `100% · ${job.event_count} 条事件` : undefined;
          return <ProgressRow key={step} title={stages[step]} value={1} detail={detail} />;
        }
        if (index !== current) return <ProgressRow key={step} title={stages[step]} value={0} detail="待处理" />;
        const failed = job.status === "error";
        const count = job.stage_progress;
        const time = step === "events" && job.duration ? { done: job.source_time, total: job.duration } : null;
        const value = time?.done ?? count?.done;
        const max = time?.total ?? count?.total;
        let detail: string | undefined;
        if (time) detail = `${timecode(time.done)} / ${timecode(time.total)} · ${job.event_count} 条事件`;
        else if (count && step === "encode") detail = `${timecode(count.done)} / ${timecode(count.total)}`;
        else if (count) detail = `${count.done} / ${count.total} ${progressUnits[step]}`;
        if (failed) detail = detail ? `失败 · ${detail}` : "失败";
        return <ProgressRow key={step} title={stages[step]} value={value ?? (failed ? 0 : undefined)}
          max={max} detail={detail} failed={failed} />;
      })}
    </div>
  </section>;
}

function Notice({ children }: { children: string }) {
  return <p className="notice" role="alert">{children}</p>;
}

function FileField({ title, accept, formats, saved, disabled, onChange }: {
  title: string;
  accept: string;
  formats: string;
  saved?: string | null;
  disabled?: boolean;
  onChange: (file: File | null) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const input = useRef<HTMLInputElement>(null);
  return <div className="file-field">
    <span className="field-title">{title}</span>
    <button type="button" className="file-select" disabled={disabled} onClick={() => input.current?.click()}>
      <span className="file-mark" aria-hidden="true">{accept === ".srt" ? "CC" : "▶"}</span>
      <span className="file-copy">
        <strong>{file?.name ?? saved ?? "选择文件"}</strong>
        <small>{formats}</small>
      </span>
      <span className="file-action">{disabled ? "已上传" : "浏览"}</span>
    </button>
    <input ref={input} type="file" accept={accept} aria-label={title} hidden onChange={(event) => {
      const next = event.currentTarget.files?.[0] ?? null;
      setFile(next);
      onChange(next);
    }} />
  </div>;
}

function UploadView({ id, job, open, update }: {
  id: string | null;
  job: Job | null;
  open: (id: string) => void;
  update: (job: Job) => void;
}) {
  const [movie, setMovie] = useState<File | null>(null);
  const [subtitles, setSubtitles] = useState<File | null>(null);
  const [sent, setSent] = useState<number | null>(null);
  const [uploading, setUploading] = useState<"movie" | "subtitles" | "generate" | null>(null);
  const [subtitleSaved, setSubtitleSaved] = useState(false);
  const [error, setError] = useState("");
  const busy = uploading !== null;
  const total = movie?.size ?? job?.movie?.total ?? 0;
  const received = sent ?? job?.movie?.offset ?? 0;
  const percent = total ? Math.floor(100 * received / total) : 0;

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setUploading("movie");
    setError("");
    const values = new FormData(event.currentTarget);
    try {
      if ((!job?.movie?.complete && !movie) || (!job?.subtitles && !subtitles)) {
        throw new Error("请选择原片和对应的 SRT 字幕");
      }
      let project = id;
      if (!project) {
        project = (await request<{ id: string }>("/api/projects", { method: "POST" })).id;
        open(project);
      }
      if (movie && !job?.movie?.complete) await uploadMovie(project, movie, setSent);
      const base = projectUrl(project);
      if (subtitles) {
        setUploading("subtitles");
        await request(`${base}/subtitles?name=${encodeURIComponent(subtitles.name)}`, { method: "PUT", body: subtitles });
        setSubtitleSaved(true);
      }
      setUploading("generate");
      update(await request<Job>(`${base}/generate`, json("POST", {
        subtitle_offset: Number(values.get("offset")),
      })));
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setUploading(null);
    }
  }

  return <>
    <header className="page-heading"><h1>制作电影解说</h1><p>上传原片和对白字幕，生成带英文旁白的解说视频。</p></header>
    <div className="layout">
      <form className="panel form-panel" onSubmit={submit}>
        <h2>项目素材</h2>
        <FileField title="电影原片" accept=".mp4,.mkv" formats="MP4 / MKV" saved={job?.movie?.name}
          disabled={busy || job?.movie?.complete} onChange={(file) => { setMovie(file); setSent(null); }} />
        <FileField title="对白字幕" accept=".srt" formats="SRT · UTF-8" saved={job?.subtitles}
          disabled={busy} onChange={(file) => { setSubtitles(file); setSubtitleSaved(false); }} />
        <label>字幕偏移（秒）<input name="offset" type="number" step="0.1" defaultValue="0" required /></label>
        <p className="meta">选段总时长上限为原片的 8%。</p>
        <div className="progress-grid">
          <ProgressRow title="上传原片" value={received} max={total || 1} />
          <ProgressRow title="上传字幕"
            value={uploading === "subtitles" ? undefined : Number(subtitleSaved || (!subtitles && !!job?.subtitles))} />
        </div>
        {error && <Notice>{error}</Notice>}
        <button className="primary" disabled={busy}>{busy ? "正在提交素材…" : received && percent < 100 ? "续传并生成" : "开始生成"}</button>
      </form>
      <section className="panel preview-panel">
        <div className="panel-heading"><h2>成片预览</h2><span className="mono meta">720p · MP4</span></div>
        <Placeholder title="等待生成成片" detail="视频内嵌英文字幕，可另存独立 SRT。" />
      </section>
    </div>
  </>;
}

function Placeholder({ title, detail }: { title: string; detail: string }) {
  return <div className="screen empty">
    <div className="screen-mark" aria-hidden="true"><span>▶</span></div>
    <strong>{title}</strong>
    <span>{detail}</span>
  </div>;
}

function SegmentEditor({ segment, remainingBudget, busy, preview, save }: {
  segment: Segment;
  remainingBudget: number | null;
  busy: boolean;
  preview: (candidate: Candidate, start?: number, end?: number) => void;
  save: (text: string, start: number, end: number, candidateId: string) => void;
}) {
  const [text, setText] = useState(segment.text);
  const [start, setStart] = useState(String(segment.source.start));
  const [end, setEnd] = useState(String(segment.source.end));
  const [candidate, setCandidate] = useState(segment.candidate_id);
  const dirty = text !== segment.text || Number(start) !== segment.source.start || Number(end) !== segment.source.end
    || candidate !== segment.candidate_id;
  const valid = start !== "" && end !== "" && Number(start) >= 0 && Number(end) > Number(start);
  const currentLength = segment.source.end - segment.source.start;
  const nextLength = valid ? Number(end) - Number(start) : currentLength;
  const delta = nextLength - currentLength;
  const overBudget = remainingBudget !== null && delta > remainingBudget + 0.05;
  return <form className="editor" onSubmit={(event) => {
    event.preventDefault();
    save(text, Number(start), Number(end), candidate);
  }}>
    <label>英文旁白<textarea lang="en" required rows={4} value={text} onChange={(event) => setText(event.target.value)} /></label>
    {segment.candidates.length > 1 && <label>候选镜头
      <select value={candidate} onChange={(event) => {
        const shot = segment.candidates.find((item) => item.id === event.target.value)!;
        setCandidate(shot.id);
        setStart(String(shot.start));
        setEnd(String(shot.end));
      }}>
        {segment.candidates.map((shot) => <option key={shot.id} value={shot.id}>
          {timecode(shot.start, true)} – {timecode(shot.end, true)} · {shot.event_id}
        </option>)}
      </select>
    </label>}
    <div className="fields">
      <label>原片入点（秒）<input type="number" min="0" step="any" required value={start} onChange={(event) => setStart(event.target.value)} /></label>
      <label>原片出点（秒）<input type="number" min="0" step="any" required value={end} onChange={(event) => setEnd(event.target.value)} /></label>
    </div>
    {overBudget && <Notice>{`当前区间比原段多 ${(delta).toFixed(1)} 秒，可能超出 8% 预算（剩余 ${Math.max(0, remainingBudget).toFixed(1)} 秒）。`}</Notice>}
    <div className="actions">
      <button type="button" className="secondary" disabled={!valid} onClick={() => {
        preview(segment.candidates.find((shot) => shot.id === candidate)!, Number(start), Number(end));
      }}>审看片段</button>
      <button className="primary compact" disabled={!dirty || !valid || busy}>保存修改</button>
    </div>
  </form>;
}

function ProjectView({ job, timeline, update }: { job: Job; timeline: Timeline | null; update: (job: Job) => void }) {
  const [selected, setSelected] = useState<string | null>(null);
  const [clip, setClip] = useState<{ url: string; start: number; end: number } | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const [videoError, setVideoError] = useState(false);
  const [clipLoading, setClipLoading] = useState(false);
  const player = useRef<HTMLVideoElement>(null);
  const base = projectUrl(job.id);
  const segment = timeline?.segments.find((item) => item.id === selected) ?? timeline?.segments[0];
  const media = job.asset_version == null ? null : `${base}/media/${job.asset_version}`;
  const source = clip?.url ?? (media && `${media}.mp4`);
  const busy = sending || job.status === "queued" || job.status === "running";
  const currentPreview = timeline && job.asset_version === timeline.version;
  const status = job.status === "error" ? "处理失败" : stages[job.stage] ?? "已就绪";

  async function submit(path: string, method = "POST", body?: object) {
    setError("");
    setSending(true);
    try {
      update(await request<Job>(`${base}${path}`, body ? json(method, body) : { method }));
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setSending(false);
    }
  }

  function preview(shot: Candidate, start = shot.start, end = shot.end) {
    const url = `${base}/clips/${shot.id}/preview?start=${start}&end=${end}`;
    if (clip?.url === url && !videoError) return;
    setVideoError(false);
    setClipLoading(true);
    if (clip?.url === url) player.current?.load();
    setClip({ url, start, end });
  }

  return <>
    <header className="page-heading project-heading">
      <div><h1>{job.movie?.name ?? "电影解说"}</h1><p>项目 <span className="mono">{job.id.slice(0, 8)}</span></p></div>
      <span className={`status ${job.status}`} role="status">{status}</span>
    </header>
    {(error || job.error) && <Notice>{error || job.error!}</Notice>}
    {job.status === "error" && job.task?.kind !== "edit" && <div className="retry"><button className="secondary" onClick={() => void submit("/retry")}>重试任务</button></div>}
    <TaskProgress job={job} />
    <div className="layout project-layout">
      <section className="panel preview-panel">
        <div className="panel-heading">
          <div><h2>{clip ? "原片审看" : "成片预览"}</h2>
            {clip ? <p className="mono meta">{timecode(clip.start, true)} – {timecode(clip.end, true)}</p>
              : media && <p className="mono meta">预览 v{job.asset_version}{timeline && !currentPreview ? ` · 当前时间线 v${timeline.version}` : ""}</p>}
          </div>
          {clip && <button className="secondary" onClick={() => { setClip(null); setVideoError(false); }}>返回成片</button>}
        </div>
        {source ? <video key={source} className="screen" ref={player} src={source} controls playsInline
            preload="metadata" autoPlay={!!clip} onLoadedMetadata={() => setClipLoading(false)} onError={() => {
              setClipLoading(false);
              setVideoError(true);
            }} />
          : <Placeholder title={status} detail="成片生成后可在此播放。" />}
        {videoError && <Notice>视频无法加载，请重新打开片段或成片。</Notice>}
        {clipLoading && clip && <div className="clip-progress"><ProgressRow title="准备审看片段" /></div>}
        {timeline && <div className="actions export">
          <button className="primary compact" disabled={busy || !!currentPreview}
            onClick={() => { setClip(null); setVideoError(false); void submit("/exports", "POST", { version: timeline.version }); }}>生成当前预览和导出</button>
          {media && <>
            <a className="secondary" href={`${media}.mp4?download=true`}>下载 MP4</a>
            <a className="secondary" href={`${media}.srt?download=true`}>下载 SRT</a>
          </>}
        </div>}
      </section>
      {segment && timeline && <section className="panel timeline-panel">
        <div className="panel-heading">
          <h2>英文旁白</h2>
          <span className="mono meta">
            {`v${timeline.version} · ${timecode(timeline.duration)}`}
            {timeline.duration_budget != null
              ? ` / ${timecode(timeline.duration_budget)}（剩余 ${Math.max(0, timeline.duration_budget - timeline.duration).toFixed(1)}s）`
              : ""}
          </span>
        </div>
        <div className="segments" aria-label="选择旁白段落">
          {timeline.segments.map((item) => <button key={item.id} type="button" aria-pressed={item.id === segment.id} className="segment" onClick={() => {
            setSelected(item.id);
            if (player.current && currentPreview && !clip) player.current.currentTime = item.start;
          }}>
            <span className="segment-time mono">{timecode(item.start)} – {timecode(item.end)}</span>
            <span className="segment-text" lang="en">{item.text}</span>
            <span className="meta mono">原片 {item.clips.map((shot) => `${timecode(shot.start, true)}–${timecode(shot.end, true)}`).join(" / ")}</span>
          </button>)}
        </div>
        <SegmentEditor key={`${timeline.version}-${segment.id}`} segment={segment}
          remainingBudget={timeline.duration_budget != null ? timeline.duration_budget - timeline.duration : null}
          busy={busy} preview={preview}
          save={(text, start, end, candidateId) => void submit("/timeline", "PATCH", {
            version: timeline.version, segment_id: segment.id, text, start, end, candidate_id: candidateId,
          })} />
      </section>}
    </div>
  </>;
}

function App() {
  const [id, setId] = useState(new URL(location.href).searchParams.get("project"));
  const [job, setJob] = useState<Job | null>(null);
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [loadError, setLoadError] = useState("");
  const polling = !job || job.status === "queued" || job.status === "running";

  useEffect(() => {
    if (!id || !polling) return;
    let alive = true;
    let timer: number;
    async function poll() {
      try {
        const result = await request<Job>(projectUrl(id!));
        if (alive) {
          setJob(result);
          setLoadError("");
          if (result.status === "queued" || result.status === "running") {
            timer = window.setTimeout(poll, 1500);
          }
        }
      } catch (error) {
        if (alive) {
          setLoadError((error as Error).message);
          timer = window.setTimeout(poll, 1500);
        }
      }
    }
    void poll();
    return () => { alive = false; clearTimeout(timer); };
  }, [id, polling]);

  useEffect(() => {
    if (!id || !job?.timeline_version || timeline?.version === job.timeline_version) return;
    let alive = true;
    request<Timeline>(`${projectUrl(id)}/timeline`).then((result) => {
      if (alive) setTimeline(result);
    }).catch((error: Error) => {
      if (alive) setLoadError(error.message);
    });
    return () => { alive = false; };
  }, [id, job, timeline?.version]);

  function open(project: string) {
    history.replaceState(null, "", `/?project=${project}`);
    setId(project);
  }

  return <>
    <nav className="topbar"><div className="wrapper nav-inner">
      <a className="brand" href="/" aria-label="Movie2Recap 首页"><span className="brand-mark" aria-hidden="true">▶</span>Movie2Recap</a>
      {id && <a className="secondary" href="/">新建项目</a>}
    </div></nav>
    <main className="wrapper">
      {loadError && <Notice>{loadError}</Notice>}
      {job && job.status !== "uploading"
        ? <ProjectView job={job} timeline={timeline} update={setJob} />
        : <UploadView id={id} job={job} open={open} update={setJob} />}
    </main>
  </>;
}

createRoot(document.getElementById("root")!).render(<App />);
