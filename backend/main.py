import os
import time
import uuid
import io
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Optional

import hmac

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import aiofiles

import db
from worker import app as celery_app, extract_srt_task, render_from_segments, detect_subs_task, preview_audio
from pipeline.translator import translate_segments, translate_to_english
from pipeline.tts import synthesize_single, list_voices
from pipeline import voice_clone
from pipeline.srt_parser import parse_srt, build_srt

UPLOAD_DIR     = Path(os.environ.get("UPLOAD_DIR",    "/app/uploads"))
OUTPUT_DIR     = Path(os.environ.get("OUTPUT_DIR",    "/app/outputs"))
RESULTS_DIR    = Path(os.environ.get("RESULTS_DIR",   str(OUTPUT_DIR / "results")))
MAX_UPLOAD_MB  = int(os.environ.get("MAX_UPLOAD_MB",  "500"))
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "30"))
CORS_ORIGINS   = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")]
# Shared access token. When set, every /api/* request (except the exemptions in
# _auth_middleware) must present it via the Authorization: Bearer header, an
# X-API-Token header, a ?token= query param (for media/download links opened by
# the browser), or the dai_token cookie. Left empty → auth disabled (local dev).
APP_TOKEN      = os.environ.get("APP_TOKEN", "").strip()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

api = FastAPI(title="DAI DUBBER — Chinese to Khmer")
api.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

# ── auth gate ─────────────────────────────────────────────────────────────────
# Endpoints reachable before the client has a token: the app shell (so the UI
# can load and prompt for one), static assets, the health check, and CORS
# preflight. Everything else under /api requires APP_TOKEN when it is set.
_AUTH_EXEMPT_PATHS = {"/", "/api/health", "/favicon.ico"}


def _present_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (
        request.headers.get("x-api-token", "").strip()
        or request.query_params.get("token", "").strip()
        or request.cookies.get("dai_token", "").strip()
    )


@api.middleware("http")
async def _auth_middleware(request: Request, call_next):
    if APP_TOKEN and request.method != "OPTIONS":
        path = request.url.path
        exempt = path in _AUTH_EXEMPT_PATHS or path.startswith("/static/")
        if not exempt and not hmac.compare_digest(_present_token(request), APP_TOKEN):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)

# ── rate limiting (simple in-memory sliding window, per IP per endpoint) ─────

_hits: dict = defaultdict(deque)

def _rate_limit(request: Request, key: str, limit: int, window: float = 60.0):
    ip = request.client.host if request.client else "?"
    q = _hits[f"{key}:{ip}"]
    now = time.monotonic()
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, f"Rate limit exceeded ({limit}/{int(window)}s) for {key}")
    q.append(now)


def _resolve_within(directory: Path, filename: str) -> Path:
    """Resolve a client-supplied filename strictly inside `directory`.

    Strips any path components with Path(...).name, then verifies the resolved
    path stays within the directory (guards against symlinks pointing out of it).
    """
    safe = Path(filename).name
    base = directory.resolve()
    path = (base / safe).resolve()
    if not (path == base or base in path.parents):
        raise HTTPException(403, "Invalid path")
    return path


def _safe_upload_path(filename: str) -> Path:
    """Sanitize a client-supplied filename and resolve it inside UPLOAD_DIR."""
    path = _resolve_within(UPLOAD_DIR, filename)
    if not path.exists():
        raise HTTPException(404, f"Uploaded file not found: {Path(filename).name}")
    return path


def _cleanup_old_files(days: int) -> dict:
    """Delete uploads/outputs older than `days`. Returns counts and bytes freed."""
    cutoff = time.time() - days * 86400
    removed, freed = 0, 0
    for d in (UPLOAD_DIR, OUTPUT_DIR):
        for f in d.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                freed += f.stat().st_size
                f.unlink()
                removed += 1
    return {"removed": removed, "freed_mb": round(freed / 1024 / 1024, 1)}


# ── helpers ──────────────────────────────────────────────────────────────────

async def _save_upload(file: UploadFile, dest: Path):
    size = 0
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                await f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB)")
            await f.write(chunk)


# ── UPLOAD VIDEO ──────────────────────────────────────────────────────────────

@api.post("/api/upload-video")
async def upload_video(request: Request, file: UploadFile = File(...)):
    _rate_limit(request, "upload", limit=10)
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO:
        raise HTTPException(400, f"Unsupported type: {ext}")
    if RETENTION_DAYS > 0:
        _cleanup_old_files(RETENTION_DAYS)
    job_id   = str(uuid.uuid4())
    filename = f"{job_id}{ext}"
    await _save_upload(file, UPLOAD_DIR / filename)
    return {"job_id": job_id, "filename": filename}


# ── EXTRACT SRT FROM VIDEO (Whisper) ─────────────────────────────────────────

@api.post("/api/extract-srt")
async def extract_srt(request: Request, filename: str = Form(...), mode: str = Form("zh")):
    _rate_limit(request, "extract", limit=10)
    if mode not in ("zh", "en"):
        mode = "zh"
    safe = _safe_upload_path(filename).name
    job_id = str(uuid.uuid4())
    task   = extract_srt_task.apply_async(args=[job_id, safe, mode], task_id=job_id)
    return {"job_id": job_id}


# ── DETECT BURNED-IN SUBTITLES FROM VIDEO (OCR) ──────────────────────────────

@api.post("/api/detect-subs")
async def detect_subs(request: Request, filename: str = Form(...), lang: str = Form("en")):
    """
    Read on-screen (burned-in) subtitles from the video via OCR, instead of
    transcribing the audio. The result is English text → translate English → Khmer.
    """
    _rate_limit(request, "detect", limit=10)
    if lang not in ("en", "zh", "ch"):
        lang = "en"
    safe = _safe_upload_path(filename).name
    job_id = str(uuid.uuid4())
    detect_subs_task.apply_async(args=[job_id, safe, lang], task_id=job_id)
    return {"job_id": job_id}


# ── IMPORT SRT FILE ───────────────────────────────────────────────────────────

@api.post("/api/import-srt")
async def import_srt(file: UploadFile = File(...)):
    raw  = await file.read()
    text = raw.decode("utf-8", errors="replace")
    segs = parse_srt(text)
    if not segs:
        raise HTTPException(400, "No valid subtitle blocks found in SRT")
    return {"segments": segs}


# ── TRANSLATE SEGMENTS ────────────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    segments: List[dict]
    source_lang: str = "zh"

@api.post("/api/translate")
def translate(req: TranslateRequest):
    translated = translate_segments(req.segments, req.source_lang)
    return {"segments": translated}


# ── FULL ENGLISH SUBTITLE (whole-video preview) ───────────────────────────────

@api.post("/api/translate-english")
def translate_english(req: TranslateRequest, request: Request):
    """Translate all segments to English and return both per-segment and a
    single concatenated full-video subtitle string."""
    _rate_limit(request, "translate_en", limit=20)
    translated = translate_to_english(req.segments, req.source_lang)
    full_text = "\n".join(s.get("english", "").strip()
                          for s in translated if s.get("english", "").strip())
    return {"segments": translated, "full_text": full_text}


# ── GENERATE SINGLE TTS (per row preview) ────────────────────────────────────

class TTSRequest(BaseModel):
    text:   str
    gender: str  = "female"
    speed:  float = 1.0
    voice:  str  = ""

@api.post("/api/tts")
def generate_tts(req: TTSRequest, request: Request):
    _rate_limit(request, "tts", limit=60)
    try:
        wav = synthesize_single(req.text, req.speed, req.gender, req.voice)
    except Exception as e:
        # Surface a readable message (e.g. provider rate limits) instead of a
        # bare 500 "Internal Server Error" so the UI can show what went wrong.
        raise HTTPException(status_code=502, detail=str(e))
    return StreamingResponse(io.BytesIO(wav), media_type="audio/wav")


@api.get("/api/voices")
def voices():
    return {"voices": list_voices()}


# ── VOICE CLONING (record a reference clip) ───────────────────────────────────

@api.post("/api/voices/clone")
async def clone_voice(
    request: Request,
    file:        UploadFile = File(...),
    label:       str = Form(""),
    gender:      str = Form("female"),
    prompt_text: str = Form(""),
):
    """
    Register a cloned voice from a recorded reference clip. The clip is
    normalized to a 16 kHz mono WAV and stored with its transcript; under the
    VoxCPM engine it's sent inline as the cloning reference (see tts.py).
    """
    _rate_limit(request, "clone", limit=20)
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Recording too large (max {MAX_UPLOAD_MB} MB)")
    if not raw:
        raise HTTPException(400, "Empty recording")
    try:
        voice = voice_clone.save_clone(raw, label, gender, prompt_text)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"voice": voice}


class CloneUpdate(BaseModel):
    gender: str | None = None
    label:  str | None = None


@api.patch("/api/voices/clone/{voice_id}")
def update_cloned_voice(voice_id: str, body: CloneUpdate):
    """Fix a cloned voice's gender (or label) after it was recorded."""
    voice = voice_clone.update_clone(
        Path(voice_id).name, gender=body.gender, label=body.label
    )
    if voice is None:
        raise HTTPException(404, "Cloned voice not found")
    return {"voice": voice}


@api.delete("/api/voices/clone/{voice_id}")
def remove_cloned_voice(voice_id: str):
    if not voice_clone.delete_clone(Path(voice_id).name):
        raise HTTPException(404, "Cloned voice not found")
    return {"deleted": voice_id}


# ── RENDER FINAL VIDEO ────────────────────────────────────────────────────────

class SubtitleStyle(BaseModel):
    font_name:  str = "Khmer OS"
    font_size:  int = 22
    font_color: str = "FFFFFF"
    outline:    int = 2
    margin_v:   int = 40   # distance from bottom edge (px); higher = subtitle sits higher up

class RenderRequest(BaseModel):
    job_id:    str
    filename:  str
    segments:  List[dict]
    burn_subs: bool = False
    sub_style: Optional[SubtitleStyle] = None
    remove_subs: bool = False
    remove_region: Optional[dict] = None
    remove_color: str = "white"
    remove_audio: bool = True

@api.post("/api/render")
def render(req: RenderRequest, request: Request):
    _rate_limit(request, "render", limit=6)
    safe = _safe_upload_path(req.filename).name
    render_id = str(uuid.uuid4())
    style = req.sub_style.dict() if req.sub_style else None
    task = render_from_segments.apply_async(
        args=[render_id, safe, req.segments, req.burn_subs, style,
              req.remove_subs, req.remove_audio,
              req.remove_region, req.remove_color],
        task_id=render_id,
    )
    return {"render_id": render_id}


# ── PREVIEW (audio-only dub, no video re-encode) ──────────────────────────────

class PreviewRequest(BaseModel):
    filename:     str
    segments:     List[dict]
    remove_audio: bool = True

@api.post("/api/preview")
def preview(req: PreviewRequest, request: Request):
    _rate_limit(request, "preview", limit=12)
    safe = _safe_upload_path(req.filename).name
    preview_id = str(uuid.uuid4())
    preview_audio.apply_async(
        args=[preview_id, safe, req.segments, req.remove_audio],
        task_id=preview_id,
    )
    return {"preview_id": preview_id}


# ── JOB STATUS ────────────────────────────────────────────────────────────────

@api.get("/api/status/{job_id}")
def job_status(job_id: str):
    result = celery_app.AsyncResult(job_id)
    if result.state == "PENDING":
        return {"state": "pending",     "stage": "Queued", "pct": 0}
    if result.state == "PROGRESS":
        meta = result.info or {}
        return {"state": "processing",  "stage": meta.get("stage",""), "pct": meta.get("pct",0)}
    if result.state == "SUCCESS":
        return {"state": "done",        **result.result}
    if result.state == "FAILURE":
        return {"state": "error",       "error": str(result.info)}
    return {"state": result.state.lower()}


# ── PROJECTS (persistence) ────────────────────────────────────────────────────

class ProjectSave(BaseModel):
    id:       str
    name:     str = ""
    filename: str = ""
    job_id:   str = ""
    segments: List[dict] = []

@api.post("/api/projects")
def save_project(req: ProjectSave):
    db.save_project(req.id, req.name, req.filename, req.job_id, req.segments)
    return {"ok": True}

@api.get("/api/projects")
def list_projects():
    return {"projects": db.list_projects()}

@api.get("/api/projects/{project_id}")
def get_project(project_id: str):
    proj = db.get_project(project_id)
    if not proj:
        raise HTTPException(404, "Project not found")
    return proj

@api.delete("/api/projects/{project_id}")
def delete_project(project_id: str):
    if not db.delete_project(project_id):
        raise HTTPException(404, "Project not found")
    return {"ok": True}


# ── HEALTH / MAINTENANCE ──────────────────────────────────────────────────────

@api.get("/api/health")
def health():
    return {"status": "ok"}

@api.post("/api/cleanup")
def cleanup(days: int = 7):
    if days < 1:
        raise HTTPException(400, "days must be >= 1")
    return _cleanup_old_files(days)


# ── SERVE UPLOADED VIDEO (for restored projects) ─────────────────────────────

@api.get("/api/video/{filename}")
def serve_video(filename: str):
    path = _safe_upload_path(filename)
    return FileResponse(path, media_type="video/mp4")


# ── DOWNLOAD ──────────────────────────────────────────────────────────────────

@api.get("/api/download/{filename}")
def download(filename: str):
    safe = Path(filename).name
    # Final videos live in outputs/results/; intermediate files (.srt) in OUTPUT_DIR.
    path = _resolve_within(RESULTS_DIR, safe)
    if not path.exists():
        path = _resolve_within(OUTPUT_DIR, safe)
    if not path.exists():
        raise HTTPException(404, "File not found")
    if safe.endswith(".srt"):
        media = "text/plain"
    elif safe.endswith(".wav"):
        media = "audio/wav"
    else:
        media = "video/mp4"
    return FileResponse(path, media_type=media, filename=safe)


# ── FRONTEND ──────────────────────────────────────────────────────────────────

frontend_dir = Path(__file__).parent / "frontend"

@api.get("/")
def serve_index():
    idx = frontend_dir / "index.html"
    if not idx.exists():
        raise HTTPException(404, "Frontend not found")
    return FileResponse(str(idx))

if frontend_dir.exists():
    api.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
