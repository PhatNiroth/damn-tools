import os
import whisper
from typing import List, Dict, Any


WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

_models: Dict[str, whisper.Whisper] = {}


def _load_model(model_size: str) -> whisper.Whisper:
    if model_size not in _models:
        try:
            _models[model_size] = whisper.load_model(model_size)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Whisper model '{model_size}': {exc}") from exc
    return _models[model_size]


def transcribe(audio_path: str, model_size: str = "",
               task: str = "transcribe", language: str = "zh") -> List[Dict[str, Any]]:
    """
    Run Whisper on the audio and return segments with timestamps.
    Each segment: {start, end, text}

    task="transcribe" (default) keeps the spoken language (Chinese) in `text`.
    task="translate" makes Whisper emit ENGLISH directly from the Chinese audio
    (Whisper only translates *into* English) — used by the English-pivot flow,
    which then translates English → Khmer (a higher-quality pair than zh → km).
    `language` is the SOURCE language of the audio, kept as "zh" for both tasks.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _load_model(model_size or WHISPER_MODEL)
    result = model.transcribe(
        audio_path,
        language=language,
        task=task,
        word_timestamps=False,
    )

    segments = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if not text:
            continue
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": text,
        })

    return segments
