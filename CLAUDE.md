# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**DAI DUBBER** — a web app that dubs Chinese-language video into Khmer. Upload a video, it transcribes the Chinese speech (Whisper), translates to Khmer, generates timed Khmer voice-over (TTS), and muxes a new audio track over the original — keeping the original audio as a ducked background. The browser UI lets you edit the subtitle table (text, timing, per-segment voice/gender) before rendering.

## Running

```bash
# Docker (recommended) — API on :8000, Redis + Celery worker
docker compose up --build

# Local dev (no Docker)
./setup.sh                                          # clones Wav2Lip, installs deps, makes .env
redis-server                                        # 1. broker
cd backend && celery -A worker worker --loglevel=info   # 2. worker (do the heavy lifting)
cd backend && uvicorn main:api --reload             # 3. API; open http://localhost:8000
```

There is no test suite or linter. `test_man_voice.py` at the repo root is a one-off TTS experiment script, not a unit test.

### Configuration (`.env`, see `docker-compose.yml` for full list)

- `GEMINI_API_KEY` — preferred translator (falls back to Google Translate free endpoint if absent/failing).
- `WHISPER_MODEL` — `small` (default) or `tiny`; both are pre-baked into the Docker image.
- `TTS_ENGINE` — `gtts` (default, free, single Khmer voice), `voxcpm` (needs `VOXCPM_BASE_URL` of a vLLM-Omni OpenAI-compatible endpoint), `voxcpm_local` (runs VoxCPM2 **in-process**, no endpoint — needs an NVIDIA GPU and `pip install -r backend/requirements-voxcpm.txt`; real voice cloning, with built-in male/female defaults bootstrapped from gTTS; knobs `VOXCPM_LOCAL_MODEL`, `VOXCPM_LOCAL_DEVICE`; see `pipeline/voxcpm_local.py`), `azure` (Microsoft Azure AI Speech — real Khmer neural voices `km-KH-*`, needs `AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION`), or `gemini` (Google AI Studio / Gemini native TTS — 30 prebuilt voices that speak Khmer, reuses `GEMINI_API_KEY`; free tier has a low requests-per-minute limit so many segments synthesize slowly). Gemini knobs: `GEMINI_TTS_MODEL` (default `gemini-2.5-flash-preview-tts`) and `GEMINI_VOICES` (JSON list of `{id,label,gender}` to override the default 8). See the paused VoxCPM2 migration in memory.
- `BG_IDLE_DB` / `BG_SPEECH_DB` — how much the original audio is ducked in gaps vs. under Khmer speech.
- `MALE_PITCH` — tunes the default male voice pitch for the gTTS engine.

## Architecture

Three processes: **FastAPI** (`backend/main.py`, sync request handling + serves the frontend), a **Celery worker** (`backend/worker.py`, runs the two long-running jobs), and **Redis** (broker + result backend; also how the frontend polls progress). State that survives a render is in **SQLite** (`backend/db.py`, `data/projects.db`) — only saved projects, not job state.

The flow is deliberately split into two async jobs so the user can review and hand-edit between them:

1. **`extract_srt_task`** — `extract_audio` (ffmpeg → 16kHz mono WAV) → `transcribe` (Whisper, `language="zh"`) → returns segments. **Translation happens separately**, synchronously, via `POST /api/translate` (not in this task), so the user edits Khmer text in the UI before render.
2. **`render_from_segments`** — takes the (edited, translated) segments and: `synthesize_segments` (Khmer TTS per segment, each time-stretched to fit its subtitle window) → `build_audio_track` (overlay TTS clips at exact timestamps over the ducked original) → `mux_audio_only` (ffmpeg swaps the audio track, video copied) → writes a Khmer `.srt`, optionally burns subtitles in.

`pipeline/` holds the stages, each a thin wrapper over ffmpeg or a library:

- `extractor.py` / `transcriber.py` — ffmpeg audio extract + Whisper. Whisper models are cached in a module-level dict.
- `diarizer.py` — speaker detection. `assign_speakers` fingerprints each segment (librosa MFCC + pitch), classifies gender, then KMeans-clusters the male segments into up to 3 voices (`man_1/2/3`, ordered deepest-first; count auto-picked by silhouette) and labels all female segments `girl`. Adds a `speaker` key. Lightweight + no API key (uses `scikit-learn`); runs in `extract_srt_task` after transcription and degrades to gender-only on any failure. The frontend's "Speaker voices" panel maps one TTS voice per detected speaker and bulk-applies it to all that speaker's segments.
- `translator.py` — Chinese→Khmer with a **fallback chain**: Gemini → Google Translate free batch endpoint → per-segment Google → passthrough (marks `translation_failed`). No hard dependency on an API key.
- `tts.py` — the most involved stage. `VOICES` are profiles; with **gTTS (one real Khmer voice)**, distinct "voices" are faked via ffmpeg pitch-shift (`asetrate`+`atempo` to shift pitch without changing duration) + bass EQ. With **VoxCPM** the engine produces real voices so that post-processing is skipped. `_to_wav` also time-stretches each clip (chained `atempo`, max 2.0 each) to exactly fill its subtitle slot. `gender: "auto"` triggers pitch-based gender detection (`librosa.pyin`) on the original audio.
- `syncer.py` — mixes the final track: original audio ducked (`BG_IDLE_DB`/`BG_SPEECH_DB` with fades at boundaries) with Khmer TTS overlaid at each segment's start. Clips are NOT re-stretched here (already sized in `tts.py`).
- `lipsync.py` — `mux_audio_only` (the default, used by the render task) and `burn_subtitles`. `run_lipsync` (Wav2Lip GAN) exists but is **not wired into the render task** — Wav2Lip is set up by `setup.sh`/Docker but currently unused by the pipeline.
- `srt_parser.py` — SRT ↔ segment dicts.

### The segment dict is the central data structure

Everything passes lists of segment dicts. A segment accumulates keys as it flows through the pipeline: `{start, end, text}` (transcribe) → `+speaker, gender` (diarize) → `+khmer` (translate) → `+voice` (UI/edit) → `+tts_path, tts_duration` (synthesize). The same shape is what the frontend edits, what `POST /api/projects` persists (as JSON in SQLite), and what `/api/render` consumes.

### Frontend

`frontend/index.html` is a **single self-contained file** (~950 lines, vanilla JS, no build step) served by FastAPI at `/`. It drives the whole flow through `/api/*`, holds the editable segment table in a global `segs` array, auto-saves projects (debounced), and polls `/api/status/{job_id}` for worker progress.

## Conventions & gotchas

- File naming is UUID-keyed: uploads are `{job_id}{ext}` in `uploads/`; outputs are `{render_id}_*.wav|mp4|srt` in `outputs/`. Filenames from clients are always sanitized through `Path(name).name` before touching disk (`_safe_upload_path`).
- The Celery `task_id` **is** the `job_id` the frontend polls — they're deliberately the same string.
- Rate limiting in `main.py` is in-memory per-IP per-endpoint (won't hold across multiple API replicas).
- Old uploads/outputs are auto-pruned on upload (`RETENTION_DAYS`, default 30).
- The repo root holds large committed sample `.mp4`s (`output_khmer_v*.mp4`, etc.) from manual runs — gitignored going forward but present on disk.
