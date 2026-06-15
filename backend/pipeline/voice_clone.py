"""
Cloned voice store.

A "cloned voice" is a short reference recording of a real speaker plus the
transcript of what they said. VoxCPM2 uses the (reference audio + reference
text) pair to clone that speaker for new Khmer speech — see `tts.py`, which
sends the pair inline with each /v1/audio/speech request.

Each voice is persisted under DATA_DIR/voices as two files that survive
restarts (DATA_DIR is a bind-mounted volume):
    <id>.wav   — reference clip, normalized to 16 kHz mono
    <id>.json  — {id, label, gender, prompt_text, created}

Ids are namespaced `clone_<hex>` so they never collide with the built-in
gTTS / VoxCPM voice ids.
"""
import base64
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR   = Path(os.environ.get("DATA_DIR", "/app/data"))
VOICES_DIR = DATA_DIR / "voices"

# Reference clips are normalized to this format (VoxCPM-friendly) and capped in
# length — a few seconds of clean speech is enough to clone from.
REF_SAMPLE_RATE = 16000
MAX_REF_SECONDS = float(os.environ.get("VOICE_CLONE_MAX_SECONDS", "30"))


def _ensure_dir() -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)


def _meta_path(voice_id: str) -> Path:
    return VOICES_DIR / f"{voice_id}.json"


def _wav_path(voice_id: str) -> Path:
    return VOICES_DIR / f"{voice_id}.wav"


def _is_clone_id(voice_id: str) -> bool:
    return bool(voice_id) and voice_id.startswith("clone_")


def _normalize_to_wav(raw_audio: bytes, dest: Path) -> None:
    """
    Decode whatever the browser recorded (webm/ogg/mp4/wav) into a trimmed,
    16 kHz mono WAV reference clip via ffmpeg.
    """
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.audio")
        with open(src, "wb") as f:
            f.write(raw_audio)
        subprocess.run(
            ["ffmpeg", "-y", "-i", src,
             "-t", str(MAX_REF_SECONDS),
             "-ac", "1", "-ar", str(REF_SAMPLE_RATE),
             str(dest)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )


def save_clone(raw_audio: bytes, label: str, gender: str, prompt_text: str) -> Dict[str, Any]:
    """
    Persist a new cloned voice from a raw recording. Returns its catalog entry.
    Raises RuntimeError if the audio can't be decoded.
    """
    _ensure_dir()
    gender = gender if gender in ("male", "female") else "female"
    label = (label or "").strip() or "My voice"
    voice_id = f"clone_{uuid.uuid4().hex[:8]}"

    try:
        _normalize_to_wav(raw_audio, _wav_path(voice_id))
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", "replace")[-300:]
        raise RuntimeError(f"could not decode recording: {err}")

    meta = {
        "id": voice_id,
        "label": label,
        "gender": gender,
        "prompt_text": (prompt_text or "").strip(),
        "created": int(time.time()),
    }
    _meta_path(voice_id).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return _public(meta)


def _public(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Catalog shape the frontend/voice list consumes."""
    return {
        "id": meta["id"],
        "label": meta["label"],
        "gender": meta["gender"],
        "cloned": True,
    }


def list_clones() -> List[Dict[str, Any]]:
    if not VOICES_DIR.exists():
        return []
    out: List[Dict[str, Any]] = []
    for meta_file in sorted(VOICES_DIR.glob("clone_*.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _wav_path(meta.get("id", "")).exists():
            out.append(_public(meta))
    return out


def get_clone(voice_id: str) -> Optional[Dict[str, Any]]:
    """Full record incl. wav_path + prompt_text, or None if unknown/missing."""
    if not _is_clone_id(voice_id):
        return None
    meta_file = _meta_path(voice_id)
    wav = _wav_path(voice_id)
    if not (meta_file.exists() and wav.exists()):
        return None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    meta["wav_path"] = str(wav)
    return meta


def clone_ids() -> set:
    return {v["id"] for v in list_clones()}


def delete_clone(voice_id: str) -> bool:
    if not _is_clone_id(voice_id):
        return False
    removed = False
    for p in (_meta_path(voice_id), _wav_path(voice_id)):
        if p.exists():
            p.unlink()
            removed = True
    return removed


def reference_b64(voice_id: str) -> Optional[Dict[str, str]]:
    """
    {audio_b64, prompt_text} for inline cloning, or None if the voice is
    unknown. Used by tts.py to attach the reference to a VoxCPM request.
    """
    rec = get_clone(voice_id)
    if not rec:
        return None
    audio = Path(rec["wav_path"]).read_bytes()
    return {
        "audio_b64": base64.b64encode(audio).decode("ascii"),
        "prompt_text": rec.get("prompt_text", ""),
    }
