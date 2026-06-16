import json
import os
import subprocess
import tempfile
from pathlib import Path
from pydub import AudioSegment
from typing import List, Dict, Any, Tuple

# ── Engine selection ───────────────────────────────────────────────────────────
# "gtts"   — default, free; one real Khmer voice, distinct voices faked via pitch/EQ
# "voxcpm" — self-hosted VoxCPM2 behind a vLLM-Omni OpenAI-compatible
#            /v1/audio/speech endpoint; produces genuinely distinct voices
# "voxcpm_local" — VoxCPM2 loaded in-process (no endpoint); needs an NVIDIA GPU.
#            Real voice cloning; built-in male/female defaults bootstrapped from
#            gTTS. See pipeline/voxcpm_local.py.
# "azure"  — Microsoft Azure AI Speech; real Khmer neural voices (km-KH),
#            needs AZURE_SPEECH_KEY + AZURE_SPEECH_REGION
# "gemini" — Google AI Studio / Gemini native TTS; 30 prebuilt voices that
#            speak Khmer. Reuses the existing GEMINI_API_KEY (same key as the
#            translator) and has a free tier — note its low requests-per-minute
#            limit, so many segments synthesize slowly.
TTS_ENGINE       = os.environ.get("TTS_ENGINE", "gtts").lower()
TTS_LANG         = os.environ.get("TTS_LANG", "km")
VOXCPM_BASE_URL  = os.environ.get("VOXCPM_BASE_URL", "").rstrip("/")
VOXCPM_MODEL     = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")

# Microsoft Azure AI Speech (neural TTS). AZURE_SPEECH_ENDPOINT optionally
# overrides the region-derived endpoint (e.g. a custom/private endpoint).
AZURE_SPEECH_KEY      = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION   = os.environ.get("AZURE_SPEECH_REGION", "southeastasia")
AZURE_SPEECH_ENDPOINT = os.environ.get("AZURE_SPEECH_ENDPOINT", "").rstrip("/")

# Google AI Studio / Gemini native TTS. Reuses GEMINI_API_KEY (the translator's
# key). The TTS model returns raw PCM (24kHz/16-bit/mono) base64 in inlineData.
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
# Free-tier Gemini TTS is rate-limited (HTTP 429). Retry with exponential
# backoff before giving up so a single segment doesn't fail the whole render.
GEMINI_TTS_MAX_RETRIES = int(os.environ.get("GEMINI_TTS_MAX_RETRIES", "5"))
GEMINI_TTS_BACKOFF     = float(os.environ.get("GEMINI_TTS_BACKOFF", "8"))

# Inline voice-cloning: field names the VoxCPM server expects for the reference
# audio (base64 wav) and its transcript. Configurable so this works with
# whatever vLLM-Omni build you end up hosting. See pipeline/voice_clone.py.
VOXCPM_REF_AUDIO_FIELD = os.environ.get("VOXCPM_REF_AUDIO_FIELD", "reference_audio")
VOXCPM_REF_TEXT_FIELD  = os.environ.get("VOXCPM_REF_TEXT_FIELD",  "reference_text")

# Hybrid cloning on a GPU box: run VoxCPM2 IN-PROCESS only for cloned segments
# while keeping TTS_ENGINE (e.g. gtts) for every other voice. Needs an NVIDIA GPU
# and `pip install -r backend/requirements-voxcpm.txt`. No endpoint required.
VOXCPM_LOCAL_CLONE = os.environ.get("VOXCPM_LOCAL_CLONE", "").strip().lower() in ("1", "true", "yes")

from . import voice_clone


def _is_voxcpm() -> bool:
    return TTS_ENGINE == "voxcpm"


def _is_voxcpm_local() -> bool:
    return TTS_ENGINE == "voxcpm_local"


def _is_azure() -> bool:
    return TTS_ENGINE == "azure"


def _is_gemini() -> bool:
    return TTS_ENGINE == "gemini"


def _real_voice_engine() -> bool:
    """Engines that synthesize genuinely distinct voices (no pitch/EQ faking)."""
    return _is_voxcpm() or _is_voxcpm_local() or _is_azure() or _is_gemini()


# ── gTTS voice profiles ─────────────────────────────────────────────────────────
# gTTS has a single Khmer voice, so distinct voices are made by true pitch
# shifting (asetrate+atempo) plus EQ. pitch <1 = deeper, >1 = higher.
# MALE_PITCH tunes the default male voice (lower = deeper).
MALE_PITCH = float(os.environ.get("MALE_PITCH", "0.80"))

# How far a TTS clip may be slowed down to fill its subtitle window. A clip
# shorter than its slot is stretched to fill it; this is the smallest atempo
# ratio allowed (0.1 = up to 10× slower). The old floor of 0.2 (5×) left short
# clips ending well before long subtitle windows. Lower = fills longer windows
# (sounds slower); raise toward 1.0 to keep speech closer to natural pace.
TTS_MIN_TEMPO = float(os.environ.get("TTS_MIN_TEMPO", "0.1"))
# Largest speed-up applied to fit a long clip into its subtitle slot. Lower =
# clearer voice (less rushed); clips that still don't fit run slightly past the
# slot rather than being compressed into chipmunk speech. Was effectively 3.9.
TTS_MAX_TEMPO = float(os.environ.get("TTS_MAX_TEMPO", "1.5"))

VOICES: Dict[str, Dict[str, Any]] = {
    "male_1":   {"label": "Male — Deep",       "gender": "male",   "pitch": MALE_PITCH, "bass": 6},
    "male_2":   {"label": "Male — Extra Deep", "gender": "male",   "pitch": 0.72,       "bass": 8},
    "male_3":   {"label": "Male — Light",      "gender": "male",   "pitch": 0.88,       "bass": 3},
    "female_1": {"label": "Female — Normal",   "gender": "female", "pitch": 1.00,       "bass": 0},
    "female_2": {"label": "Female — Soft",     "gender": "female", "pitch": 0.93,       "bass": 0},
    "female_3": {"label": "Female — Bright",   "gender": "female", "pitch": 1.08,       "bass": 0},
}
DEFAULT_GTTS_VOICE = {"male": "male_1", "female": "female_1"}

# ── VoxCPM2 voice catalog ────────────────────────────────────────────────────────
# VoxCPM is a voice-cloning model: each "voice" is a reference speaker the server
# is configured to serve. The `id` here is sent verbatim as the `voice` field to
# the endpoint, so it MUST match a speaker name your VoxCPM2 server knows.
# Override the whole catalog via VOXCPM_VOICES — a JSON list of
# {"id","label","gender"} objects — to track your server's actual speaker set.
_DEFAULT_VOXCPM_VOICES: List[Dict[str, str]] = [
    {"id": "km_male_warm",     "label": "Male — Warm",      "gender": "male"},
    {"id": "km_male_deep",     "label": "Male — Deep",      "gender": "male"},
    {"id": "km_male_narrator", "label": "Male — Narrator",  "gender": "male"},
    {"id": "km_female_warm",   "label": "Female — Warm",    "gender": "female"},
    {"id": "km_female_bright", "label": "Female — Bright",  "gender": "female"},
    {"id": "km_female_narrator","label": "Female — Narrator","gender": "female"},
]


def _load_voxcpm_voices() -> List[Dict[str, str]]:
    raw = os.environ.get("VOXCPM_VOICES", "").strip()
    if not raw:
        return _DEFAULT_VOXCPM_VOICES
    try:
        data = json.loads(raw)
        voices = [v for v in data if v.get("id") and v.get("gender")]
        return voices or _DEFAULT_VOXCPM_VOICES
    except Exception as e:
        print(f"[tts] invalid VOXCPM_VOICES, using defaults: {e}")
        return _DEFAULT_VOXCPM_VOICES


VOXCPM_VOICES = _load_voxcpm_voices()
DEFAULT_VOXCPM_VOICE = {
    "male":   next((v["id"] for v in VOXCPM_VOICES if v["gender"] == "male"), ""),
    "female": next((v["id"] for v in VOXCPM_VOICES if v["gender"] == "female"), ""),
}

# ── Azure AI Speech voice catalog ────────────────────────────────────────────────
# Khmer (km-KH) neural voices. `id` is the Azure voice name sent in the SSML.
# Override with AZURE_VOICES — a JSON list of {"id","label","gender"} — to use
# other Azure voices (e.g. multilingual ones).
_DEFAULT_AZURE_VOICES: List[Dict[str, str]] = [
    {"id": "km-KH-SreymomNeural", "label": "Sreymom — Female", "gender": "female"},
    {"id": "km-KH-PisethNeural",  "label": "Piseth — Male",    "gender": "male"},
]


def _load_azure_voices() -> List[Dict[str, str]]:
    raw = os.environ.get("AZURE_VOICES", "").strip()
    if not raw:
        return _DEFAULT_AZURE_VOICES
    try:
        data = json.loads(raw)
        voices = [v for v in data if v.get("id") and v.get("gender")]
        return voices or _DEFAULT_AZURE_VOICES
    except Exception as e:
        print(f"[tts] invalid AZURE_VOICES, using defaults: {e}")
        return _DEFAULT_AZURE_VOICES


AZURE_VOICES = _load_azure_voices()
DEFAULT_AZURE_VOICE = {
    "male":   next((v["id"] for v in AZURE_VOICES if v["gender"] == "male"), ""),
    "female": next((v["id"] for v in AZURE_VOICES if v["gender"] == "female"), ""),
}

# ── Gemini (Google AI Studio) voice catalog ──────────────────────────────────────
# `id` is the Gemini prebuilt voice name sent as prebuiltVoiceConfig.voiceName.
# Gemini doesn't officially label voice gender; the genders below are the
# commonly-perceived ones for a curated subset. Override the whole catalog with
# GEMINI_VOICES — a JSON list of {"id","label","gender"} — to use any of the 30
# voices (Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, ... etc.).
_DEFAULT_GEMINI_VOICES: List[Dict[str, str]] = [
    {"id": "Kore",     "label": "Kore — Firm (F)",       "gender": "female"},
    {"id": "Aoede",    "label": "Aoede — Breezy (F)",    "gender": "female"},
    {"id": "Leda",     "label": "Leda — Youthful (F)",   "gender": "female"},
    {"id": "Zephyr",   "label": "Zephyr — Bright (F)",   "gender": "female"},
    {"id": "Puck",     "label": "Puck — Upbeat (M)",     "gender": "male"},
    {"id": "Charon",   "label": "Charon — Informative (M)", "gender": "male"},
    {"id": "Orus",     "label": "Orus — Firm (M)",       "gender": "male"},
    {"id": "Fenrir",   "label": "Fenrir — Excitable (M)", "gender": "male"},
]


def _load_gemini_voices() -> List[Dict[str, str]]:
    raw = os.environ.get("GEMINI_VOICES", "").strip()
    if not raw:
        return _DEFAULT_GEMINI_VOICES
    try:
        data = json.loads(raw)
        voices = [v for v in data if v.get("id") and v.get("gender")]
        return voices or _DEFAULT_GEMINI_VOICES
    except Exception as e:
        print(f"[tts] invalid GEMINI_VOICES, using defaults: {e}")
        return _DEFAULT_GEMINI_VOICES


GEMINI_VOICES = _load_gemini_voices()
DEFAULT_GEMINI_VOICE = {
    "male":   next((v["id"] for v in GEMINI_VOICES if v["gender"] == "male"), ""),
    "female": next((v["id"] for v in GEMINI_VOICES if v["gender"] == "female"), ""),
}


# ── Voice resolution (engine-aware) ──────────────────────────────────────────────

def list_voices() -> List[Dict[str, str]]:
    """
    Voice catalog for the active engine — VoxCPM speakers or gTTS profiles —
    plus any cloned voices the user has recorded. Cloned voices only truly
    clone under the VoxCPM engine; under gTTS they fall back to a built-in
    voice of the same gender.
    """
    if _is_voxcpm():
        base = [{"id": v["id"], "label": v["label"], "gender": v["gender"]} for v in VOXCPM_VOICES]
    elif _is_voxcpm_local():
        from . import voxcpm_local
        base = [dict(v) for v in voxcpm_local.LOCAL_VOICES]
    elif _is_azure():
        base = [{"id": v["id"], "label": v["label"], "gender": v["gender"]} for v in AZURE_VOICES]
    elif _is_gemini():
        base = [{"id": v["id"], "label": v["label"], "gender": v["gender"]} for v in GEMINI_VOICES]
    else:
        base = [{"id": vid, "label": v["label"], "gender": v["gender"]} for vid, v in VOICES.items()]
    # List cloned voices under every engine so they can be selected/applied;
    # real cloning only happens under VoxCPM, other engines fall back to a
    # same-gender voice (mirrors _active_ids() and the synthesis path).
    base += voice_clone.list_clones()
    return base


def _active_ids() -> set:
    """Set of valid voice ids for the active engine (incl. cloned voices)."""
    if _is_voxcpm():
        base = {v["id"] for v in VOXCPM_VOICES}
    elif _is_voxcpm_local():
        from . import voxcpm_local
        base = {v["id"] for v in voxcpm_local.LOCAL_VOICES}
    elif _is_azure():
        base = {v["id"] for v in AZURE_VOICES}
    elif _is_gemini():
        base = {v["id"] for v in GEMINI_VOICES}
    else:
        base = set(VOICES.keys())
    # Keep clone ids valid regardless of engine so a saved selection survives an
    # engine switch; non-VoxCPM engines fall back to a same-gender voice.
    return base | voice_clone.clone_ids()


def _default_voice_id(gender: str) -> str:
    g = gender if gender in ("male", "female") else "female"
    if _is_voxcpm():
        return DEFAULT_VOXCPM_VOICE.get(g) or DEFAULT_VOXCPM_VOICE["female"]
    if _is_voxcpm_local():
        from . import voxcpm_local
        return voxcpm_local.DEFAULT_LOCAL_VOICE.get(g, "local_female")
    if _is_azure():
        return DEFAULT_AZURE_VOICE.get(g) or DEFAULT_AZURE_VOICE["female"]
    if _is_gemini():
        return DEFAULT_GEMINI_VOICE.get(g) or DEFAULT_GEMINI_VOICE["female"]
    return DEFAULT_GTTS_VOICE.get(g, "female_1")


def _resolve_voice(voice_id: str, gender: str) -> Dict[str, Any]:
    """
    Return the post-processing profile for a voice. VoxCPM produces real voices,
    so no pitch/EQ is applied — a neutral profile is returned. For gTTS this is
    the pitch/bass profile used to fake the chosen voice.
    """
    if _real_voice_engine():
        return {"pitch": 1.0, "bass": 0}
    if voice_id in VOICES:
        return VOICES[voice_id]
    return VOICES[DEFAULT_GTTS_VOICE.get(gender, "female_1")]


def _gtts_mp3(text: str) -> bytes:
    from gtts import gTTS
    with tempfile.TemporaryDirectory() as tmp:
        mp3 = os.path.join(tmp, "out.mp3")
        gTTS(text=text, lang=TTS_LANG).save(mp3)
        with open(mp3, "rb") as f:
            return f.read()


def _voxcpm_audio(text: str, voice_name: str) -> bytes:
    """
    Call a self-hosted VoxCPM2 OpenAI-compatible speech endpoint.

    For a cloned voice (`clone_*`) the speaker isn't pre-registered on the
    server, so we send the reference clip + its transcript inline and omit the
    named `voice`. The server clones from the reference.
    """
    import urllib.request
    if not VOXCPM_BASE_URL:
        raise RuntimeError("TTS_ENGINE=voxcpm but VOXCPM_BASE_URL is not set")

    body: Dict[str, Any] = {
        "model": VOXCPM_MODEL,
        "input": text,
        "response_format": "wav",
    }
    ref = voice_clone.reference_b64(voice_name)
    if ref:
        body[VOXCPM_REF_AUDIO_FIELD] = ref["audio_b64"]
        body[VOXCPM_REF_TEXT_FIELD]  = ref["prompt_text"]
    else:
        body["voice"] = voice_name

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{VOXCPM_BASE_URL}/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _azure_audio(text: str, voice_name: str) -> bytes:
    """Call Microsoft Azure AI Speech (neural TTS) via the REST endpoint."""
    import urllib.request
    from xml.sax.saxutils import escape
    if not AZURE_SPEECH_KEY:
        raise RuntimeError("TTS_ENGINE=azure but AZURE_SPEECH_KEY is not set")
    endpoint = AZURE_SPEECH_ENDPOINT or (
        f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    )
    voice_name = voice_name or _default_voice_id("female")
    ssml = (
        "<speak version='1.0' xml:lang='km-KH'>"
        f"<voice name='{escape(voice_name, {chr(39): '&apos;'})}'>{escape(text)}</voice>"
        "</speak>"
    )
    req = urllib.request.Request(
        endpoint,
        data=ssml.encode("utf-8"),
        headers={
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "Content-Type": "application/ssml+xml",
            # 22050 Hz mono WAV — matches the pipeline's working sample rate.
            "X-Microsoft-OutputFormat": "riff-22050hz-16bit-mono-pcm",
            "User-Agent": "dai-dubber",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _gemini_audio(text: str, voice_name: str) -> bytes:
    """
    Call Google's Gemini native TTS (AI Studio key) and return WAV bytes.

    The API returns raw PCM (24kHz, 16-bit, mono) base64-encoded in
    inlineData, so we wrap it in a WAV container before handing it to _to_wav.
    """
    import urllib.request
    import urllib.error
    import base64
    import io
    import time
    import wave
    if not GEMINI_API_KEY:
        raise RuntimeError("TTS_ENGINE=gemini but GEMINI_API_KEY is not set")

    voice_name = voice_name or _default_voice_id("female")
    body = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice_name}
                }
            },
        },
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_TTS_MODEL}:generateContent"
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
    )

    # Gemini's free tier returns 429 for two very different reasons:
    #   • per-MINUTE rate limit  → transient, worth retrying with backoff
    #   • per-DAY request quota   → exhausted for the day, retrying is futile
    # Inspect the quota violation so we retry the first but fail fast on the
    # second with an actionable message instead of a 2-minute hang.
    data = None
    for attempt in range(GEMINI_TTS_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise RuntimeError(f"Gemini TTS HTTP {e.code}: {e.read()[:200]!r}") from e

            err_body = e.read().decode("utf-8", "replace")
            per_day = "PerDay" in err_body or "RequestsPerDay" in err_body
            if per_day:
                raise RuntimeError(
                    "Gemini TTS daily free-tier quota exhausted (only ~10 "
                    "requests/day on the free tier). It resets ~midnight "
                    "Pacific. To keep working now, set TTS_ENGINE=gtts (free, "
                    "unlimited) or TTS_ENGINE=azure (real Khmer voices, needs "
                    "AZURE_SPEECH_KEY), then restart."
                ) from e
            if attempt < GEMINI_TTS_MAX_RETRIES - 1:
                retry_after = e.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() \
                    else GEMINI_TTS_BACKOFF * (2 ** attempt)
                print(f"[tts] Gemini per-minute 429; retrying in {wait:.0f}s "
                      f"(attempt {attempt + 1}/{GEMINI_TTS_MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise RuntimeError(
                "Gemini TTS per-minute rate limit hit repeatedly (HTTP 429). "
                "Wait a moment and retry, or set a different TTS_ENGINE."
            ) from e

    try:
        b64 = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Gemini TTS returned no audio: {str(data)[:200]}") from e
    pcm = base64.b64decode(b64)

    # Wrap raw PCM (24kHz/16-bit/mono) into a WAV container.
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    return buf.getvalue()


def _engine_audio(text: str, voice_id: str, voice: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    """
    Generate raw speech audio. Returns (audio_bytes, post_processing_voice):
    VoxCPM2 and Azure produce real voices, so no pitch/EQ post-processing is
    applied; gTTS has one voice, so the profile's pitch/bass are applied after.
    """
    # Hybrid cloning: a cloned voice (clone_*) is reproduced by VoxCPM from its
    # reference clip. So even when the configured engine can't clone (gTTS /
    # Azure / Gemini), route ONLY the cloned segments to VoxCPM — a remote
    # endpoint (VOXCPM_BASE_URL) or the in-process model (VOXCPM_LOCAL_CLONE, for
    # a GPU box) — while normal voices stay on the main engine. The dedicated
    # voxcpm / voxcpm_local engines keep their own branches below.
    if not _is_voxcpm() and not _is_voxcpm_local():
        clone = voice_clone.get_clone(voice_id)
        if clone and VOXCPM_BASE_URL:
            return _voxcpm_audio(text, voice_id), {"pitch": 1.0, "bass": 0}
        if clone and VOXCPM_LOCAL_CLONE:
            from . import voxcpm_local
            wav = voxcpm_local.synthesize(text, voice_id, clone.get("gender", "female"))
            return wav, {"pitch": 1.0, "bass": 0}
    if _is_voxcpm():
        return _voxcpm_audio(text, voice_id or _default_voice_id("female")), {"pitch": 1.0, "bass": 0}
    if _is_voxcpm_local():
        from . import voxcpm_local
        gender = voice.get("gender", "female") if isinstance(voice, dict) else "female"
        wav = voxcpm_local.synthesize(text, voice_id or _default_voice_id("female"), gender)
        return wav, {"pitch": 1.0, "bass": 0}
    if _is_azure():
        # A cloned voice can't be used by Azure — fall back to a same-gender voice.
        if voice_id in {v["id"] for v in AZURE_VOICES}:
            az_id = voice_id
        else:
            clone = voice_clone.get_clone(voice_id)
            az_id = _default_voice_id(clone["gender"] if clone else "female")
        return _azure_audio(text, az_id), {"pitch": 1.0, "bass": 0}
    if _is_gemini():
        # A cloned voice can't be used by Gemini — fall back to a same-gender voice.
        if voice_id in {v["id"] for v in GEMINI_VOICES}:
            gem_id = voice_id
        else:
            clone = voice_clone.get_clone(voice_id)
            gem_id = _default_voice_id(clone["gender"] if clone else "female")
        return _gemini_audio(text, gem_id), {"pitch": 1.0, "bass": 0}
    return _gtts_mp3(text), voice


def _atempo_chain(ratio: float) -> str:
    """
    Build an ffmpeg atempo filter chain for an arbitrary speed `ratio`.

    A single atempo is limited to [0.5, 2.0], so large/small ratios are split
    across stages: ratio>1 speeds the clip up (shortens it), ratio<1 slows it
    down (lengthens it). This lets a short TTS clip be stretched as much as
    needed to fill its whole subtitle window — e.g. ratio 0.25 → "0.5,0.5".
    """
    factors = []
    r = ratio
    while r > 2.0:
        factors.append(2.0)
        r /= 2.0
    while r < 0.5:
        factors.append(0.5)
        r /= 0.5
    factors.append(r)
    return ",".join(f"atempo={f:.5f}" for f in factors)


def _probe_duration(path: str) -> float:
    """Duration of an audio file in seconds (0.0 if it can't be read)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def _to_wav(
    audio_bytes: bytes,
    voice: Dict[str, Any],
    target_duration: float = 0.0,
) -> bytes:
    """
    Convert speech audio → WAV with, in a SINGLE ffmpeg pass:
    - true pitch shift per voice profile (asetrate + atempo keeps duration)
    - bass boost per voice profile
    - time-stretch via atempo to hit target_duration exactly

    This was previously up to 4 separate ffmpeg invocations per segment; one
    chained `-af` filtergraph is markedly faster and avoids re-encoding the
    intermediate WAVs. Pitch shift and bass boost both preserve duration, so the
    stretch ratio can be derived from the source clip's own duration up front.
    """
    pitch = float(voice.get("pitch", 1.0))
    bass  = int(voice.get("bass", 0))

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.audio")
        out = os.path.join(tmp, "out.wav")
        with open(src, "wb") as f:
            f.write(audio_bytes)

        filters: List[str] = []

        # ── Pitch shift ────────────────────────────────────────────────
        # asetrate drops/raises pitch AND changes speed; atempo restores the
        # original duration so only the pitch changes.
        if abs(pitch - 1.0) > 0.01:
            shifted = int(22050 * pitch)
            filters += [f"asetrate={shifted}", "aresample=22050",
                        f"atempo={1.0 / pitch:.5f}"]

        # ── Bass boost (deep voices) ───────────────────────────────────
        if bass > 0:
            filters.append(f"equalizer=f=100:width_type=o:width=2:g={bass}")

        # ── Time-stretch to match target_duration exactly ──────────────
        # Pitch/bass leave duration unchanged, so the source duration is the
        # clip duration the stretch ratio is computed against.
        if target_duration > 0.1:
            clip_dur = _probe_duration(src)
            if clip_dur > 0.05:
                ratio = clip_dur / target_duration          # >1 = speed up, <1 = slow down
                # Allow heavy slow-down so a short clip fills its whole subtitle
                # window (voice shouldn't end before the subtitle does). Floor is
                # TTS_MIN_TEMPO (default 0.1 = up to 10× stretch), a guard against
                # absurd stretches on tiny clips. The speed-up is capped at
                # TTS_MAX_TEMPO (default 1.5×) so a long Khmer line is never
                # crushed into fast/garbled speech — it just runs a bit past its
                # slot instead.
                ratio = max(TTS_MIN_TEMPO, min(TTS_MAX_TEMPO, ratio))
                filters.append(_atempo_chain(ratio))

        cmd = ["ffmpeg", "-y", "-i", src, "-ar", "22050", "-ac", "1"]
        if filters:
            cmd += ["-af", ",".join(filters)]
        cmd.append(out)
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(out, "rb") as f:
            return f.read()


def _detect_gender_from_segment(audio_path: str, start: float, end: float) -> str:
    try:
        import librosa, numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            seg = os.path.join(tmp, "seg.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path,
                 "-ss", str(start), "-to", str(end),
                 "-ac", "1", "-ar", "22050", seg],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            y, sr = librosa.load(seg, sr=None, mono=True)
            f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
            f0 = f0[~np.isnan(f0)]
            if len(f0) == 0:
                return "female"
            return "male" if np.mean(f0) < 160 else "female"
    except Exception:
        return "female"


def synthesize_single(text: str, speed: float = 1.0, gender: str = "male",
                      voice_id: str = "") -> bytes:
    if not voice_id or voice_id == "auto" or voice_id not in _active_ids():
        voice_id = _default_voice_id(gender)
    voice = _resolve_voice(voice_id, gender)
    audio, post_voice = _engine_audio(text, voice_id, voice)
    return _to_wav(audio, post_voice)


def _tts_concurrency() -> int:
    """
    How many segments to synthesize at once. TTS is network- and ffmpeg-bound
    (both release the GIL), so threads give real speedup. Gemini's free tier is
    request-per-minute limited, so it defaults to a low concurrency to avoid
    hammering into 429s; everything else defaults higher. Override with
    TTS_CONCURRENCY.
    """
    raw = os.environ.get("TTS_CONCURRENCY", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    # A single local GPU can't run model calls in parallel — serialize them.
    if _is_voxcpm_local():
        return 1
    return 3 if _is_gemini() else 8


def synthesize_segments(
    segments: List[Dict[str, Any]],
    output_dir: str,
    job_id: str,
    original_audio_path: str = "",
) -> List[Dict[str, Any]]:
    out_dir = Path(output_dir)

    def synth_one(i: int, seg: Dict[str, Any]) -> Dict[str, Any]:
        khmer = seg.get("khmer", "")
        if not khmer:
            return {**seg, "tts_path": None, "tts_duration": 0.0}

        # Resolve gender (used for auto-detection and as voice fallback)
        gender = seg.get("gender", "male")
        voice_id = seg.get("voice", "")
        if gender == "auto" or voice_id == "auto":
            if original_audio_path and os.path.exists(original_audio_path):
                gender = _detect_gender_from_segment(
                    original_audio_path, seg["start"], seg["end"]
                )
            else:
                gender = "male"
            voice_id = _default_voice_id(gender)
        # Voice from another engine (e.g. a saved gTTS id under VoxCPM) → fall back
        elif voice_id not in _active_ids():
            voice_id = _default_voice_id(gender)
        voice = _resolve_voice(voice_id, gender)

        slot = max(0.1, seg["end"] - seg["start"])
        wav_path = str(out_dir / f"{job_id}_seg{i:04d}.wav")

        try:
            audio_bytes, post_voice = _engine_audio(khmer, voice_id, voice)
            wav_bytes = _to_wav(audio_bytes, post_voice, target_duration=slot)
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
            duration = len(AudioSegment.from_wav(wav_path)) / 1000.0
            vname = voice_id if voice_id in _active_ids() else _default_voice_id(gender)
            print(f"[tts] seg {i:03d} | {vname:9s} | slot={slot:.2f}s | tts={duration:.2f}s")
            return {**seg, "tts_path": wav_path, "tts_duration": duration,
                    "gender": gender, "voice": vname}
        except Exception as e:
            print(f"[tts] seg {i} FAILED: {e}")
            return {**seg, "tts_path": None, "tts_duration": 0.0, "gender": gender}

    workers = max(1, min(_tts_concurrency(), len(segments) or 1))
    if workers == 1 or len(segments) <= 1:
        return [synth_one(i, seg) for i, seg in enumerate(segments)]

    from concurrent.futures import ThreadPoolExecutor
    result: List[Any] = [None] * len(segments)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(synth_one, i, seg): i
                   for i, seg in enumerate(segments)}
        for fut in futures:
            result[futures[fut]] = fut.result()
    return result
