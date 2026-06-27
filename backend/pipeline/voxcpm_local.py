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
        except Exception:
            # Some builds infer device themselves or need .to() after loading.
            _model = VoxCPM.from_pretrained(VOXCPM_LOCAL_MODEL)
            try:
                import torch
                if device == "cuda" and torch.cuda.is_available():
                    _model = _model.to(device)
            except Exception:
                pass
        print("[voxcpm_local] model ready.")
    return _model


def _collect_wav(result) -> "np.ndarray":
    """
    Normalise model output to a 1-D float32 numpy array.

    VoxCPM2/CosyVoice-family models may return:
      - a single torch Tensor or numpy array
      - a generator of chunks, each a raw Tensor/array or a dict
        like {'tts_speech': Tensor} (CosyVoice streaming style)
    """
    import numpy as np

    try:
        import torch
        _is_tensor = lambda x: isinstance(x, torch.Tensor)
        _to_np = lambda x: x.detach().cpu().numpy()
    except ImportError:
        _is_tensor = lambda x: False
        _to_np = lambda x: x

    def _as_np(chunk):
        if isinstance(chunk, dict):
            for key in ("tts_speech", "wav", "audio"):
                if key in chunk:
                    chunk = chunk[key]
                    break
            else:
                chunk = next(iter(chunk.values()))
        if _is_tensor(chunk):
            chunk = _to_np(chunk)
        return np.asarray(chunk, dtype=np.float32).squeeze()

    # A plain Tensor or dict must go through _as_np directly — both are
    # technically iterable but should NOT be consumed as generators.
    if _is_tensor(result) or isinstance(result, (dict, np.ndarray, bytes, bytearray)):
        return _as_np(result)

    # Generator / iterator (CosyVoice streaming style)
    if hasattr(result, "__next__") or hasattr(result, "__iter__"):
        chunks = [_as_np(c) for c in result]
        if not chunks:
            raise RuntimeError("[voxcpm_local] model returned empty generator")
        return np.concatenate(chunks)

    return _as_np(result)


def _generate_wav(text: str, ref_wav_path: str, ref_text: str) -> bytes:
    """
    Run the model and return WAV bytes. ISOLATED so the VoxCPM API call is the
    only thing to adjust if the installed package's signature differs.

    Tries the file-path API first; if the build doesn't accept prompt_wav_path,
    loads the reference audio as a tensor and retries with prompt_speech_16k
    (the CosyVoice convention used by some VoxCPM2 builds).
    """
    import numpy as np
    model = _get_model()
    sr = int(getattr(model, "sample_rate", 22050))

    ref_path = ref_wav_path or None
    ref_txt  = ref_text or None

    # Wrap only model.generate() in the TypeError guard so that errors from
    # _collect_wav do not incorrectly trigger the fallback API path.
    try:
        result = model.generate(
            text=text,
            prompt_wav_path=ref_path,
            prompt_text=ref_txt,
        )
    except TypeError:
        # Build uses prompt_speech_16k (CosyVoice convention) instead of a path.
        ref_tensor = None
        if ref_path:
            try:
                import soundfile as sf
                import torch
                ref_audio, _ = sf.read(ref_path, dtype="float32")
                # Move to the model's device — required when model is on CUDA.
                model_device = next(model.parameters()).device
                ref_tensor = torch.from_numpy(ref_audio).unsqueeze(0).to(model_device)
            except Exception:
                pass
        result = model.generate(
            text=text,
            prompt_speech_16k=ref_tensor,
            prompt_text=ref_txt,
        )
    wav = _collect_wav(result)

    if wav.size == 0:
        raise RuntimeError("[voxcpm_local] model generated zero-length audio")

    # NaN/Inf can appear from GPU computation; replace with silence.
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

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
        print(f"[voxcpm_local] bootstrapped default {gender} reference -> {path}")
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
