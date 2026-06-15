"""
In-process VoxCPM2 TTS — runs the model locally instead of calling a remote
VOXCPM_BASE_URL endpoint. Intended for a machine with an NVIDIA GPU.

VoxCPM is a voice-CLONING model: it has no pre-registered named speakers. Every
voice it produces comes from a reference clip + that clip's transcript. So this
engine produces voices in two ways:

  • cloned voices (clone_*) — reference clip the user recorded in the browser
    (pipeline/voice_clone.py), passed straight into the model.
  • two built-in defaults (male/female) — bootstrapped from gTTS: we synthesize
    one short Khmer sentence with gTTS (a real Khmer voice), cache it as a
    reference WAV, and clone from it. No bundled audio files, and the prompt
    transcript is an exact match for the audio by construction.

The model is loaded once (module-level singleton, like the Whisper cache) and
kept resident in VRAM. The exact VoxCPM Python API is isolated in
`_generate_wav` so it's a one-line change if the installed package differs.
"""
import os
import io
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional, Tuple

DATA_DIR        = Path(os.environ.get("DATA_DIR", "/app/data"))
REF_DIR         = DATA_DIR / "voices"               # shared with voice_clone
VOXCPM_LOCAL_MODEL  = os.environ.get("VOXCPM_LOCAL_MODEL", "openbmb/VoxCPM2")
VOXCPM_LOCAL_DEVICE = os.environ.get("VOXCPM_LOCAL_DEVICE", "").strip()  # "", "cuda", "cpu"

# The Khmer sentence gTTS speaks to make the default reference clips. Because
# gTTS reads exactly this text, it doubles as the reference transcript.
_DEFAULT_REF_TEXT = os.environ.get(
    "VOXCPM_LOCAL_REF_TEXT",
    "សួស្តី តើអ្នកសុខសប្បាយជាទេ ថ្ងៃនេះអាកាសធាតុល្អណាស់។",
)

# Built-in voice catalog for this engine: one default per gender + clones.
LOCAL_VOICES = [
    {"id": "local_female", "label": "Khmer — Female (default)", "gender": "female"},
    {"id": "local_male",   "label": "Khmer — Male (default)",   "gender": "male"},
]
DEFAULT_LOCAL_VOICE = {"male": "local_male", "female": "local_female"}

_model = None


def _resolve_device() -> str:
    if VOXCPM_LOCAL_DEVICE:
        return VOXCPM_LOCAL_DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    print("[voxcpm_local] WARNING: no CUDA device found — running on CPU. "
          "A 2B model on CPU is very slow (tens of seconds+ per segment).")
    return "cpu"


def _get_model():
    """Load VoxCPM2 once and cache it (resident in VRAM)."""
    global _model
    if _model is None:
        from voxcpm import VoxCPM  # heavy import; deferred until first use
        device = _resolve_device()
        print(f"[voxcpm_local] loading {VOXCPM_LOCAL_MODEL} on {device} …")
        try:
            _model = VoxCPM.from_pretrained(VOXCPM_LOCAL_MODEL, device=device)
        except TypeError:
            # Older builds infer the device themselves (no `device` kwarg).
            _model = VoxCPM.from_pretrained(VOXCPM_LOCAL_MODEL)
        print("[voxcpm_local] model ready.")
    return _model


def _generate_wav(text: str, ref_wav_path: str, ref_text: str) -> bytes:
    """
    Run the model and return WAV bytes. ISOLATED so the VoxCPM API call is the
    only thing to adjust if the installed package's signature differs.
    """
    import numpy as np
    model = _get_model()
    wav = model.generate(
        text=text,
        prompt_wav_path=ref_wav_path or None,
        prompt_text=ref_text or None,
    )
    # `generate` returns a float waveform (numpy/torch) at model.sample_rate.
    sr = int(getattr(model, "sample_rate", 16000))
    if hasattr(wav, "detach"):          # torch tensor → numpy
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


# ── Default reference clips (bootstrapped from gTTS) ─────────────────────────────

def _default_ref_path(gender: str) -> Path:
    return REF_DIR / f"_local_ref_{gender}.wav"


def _ensure_default_ref(gender: str) -> Tuple[str, str]:
    """
    Return (ref_wav_path, ref_text) for a built-in default voice, generating &
    caching the gTTS-derived reference clip on first use. Male is pitch-shifted
    down so the two defaults are audibly distinct.
    """
    REF_DIR.mkdir(parents=True, exist_ok=True)
    path = _default_ref_path(gender)
    if not path.exists():
        from .tts import _gtts_mp3, _to_wav, VOICES, DEFAULT_GTTS_VOICE
        mp3 = _gtts_mp3(_DEFAULT_REF_TEXT)
        profile = VOICES[DEFAULT_GTTS_VOICE.get(gender, "female_1")]
        wav_bytes = _to_wav(mp3, profile)        # apply gender pitch/EQ, no stretch
        path.write_bytes(wav_bytes)
        print(f"[voxcpm_local] bootstrapped default {gender} reference → {path}")
    return str(path), _DEFAULT_REF_TEXT


def synthesize(text: str, voice_id: str, gender: str = "female") -> bytes:
    """
    Synthesize `text` for the given voice and return WAV bytes (neutral profile;
    no pitch/EQ post-processing — VoxCPM produces a real voice).

    voice_id may be a clone_* id (uses the recorded reference) or a built-in
    local_male / local_female (uses the gTTS-bootstrapped default reference).
    """
    from . import voice_clone
    clone = voice_clone.get_clone(voice_id)
    if clone:
        ref_wav, ref_text = clone["wav_path"], clone.get("prompt_text", "")
        g = clone.get("gender", gender)
    else:
        g = gender if gender in ("male", "female") else "female"
        if voice_id == "local_male":
            g = "male"
        elif voice_id == "local_female":
            g = "female"
        ref_wav, ref_text = _ensure_default_ref(g)
    return _generate_wav(text, ref_wav, ref_text)
