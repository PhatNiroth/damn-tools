import subprocess
import os
from pathlib import Path


def extract_audio(video_path: str, output_dir: str) -> str:
    """Extract audio from video as 16kHz mono WAV for Whisper.

    The output name is derived from the source video stem, so the extract /
    preview / render phases of the same upload all resolve to the same WAV.
    If that WAV already exists and is at least as new as the source video, the
    ffmpeg pass is skipped and the cached file is reused (the audio is identical
    across phases). Set EXTRACT_AUDIO_NOCACHE=1 to always re-extract.
    """
    video_path = Path(video_path)
    audio_path = Path(output_dir) / f"{video_path.stem}_audio.wav"

    nocache = os.environ.get("EXTRACT_AUDIO_NOCACHE", "").strip() in ("1", "true", "yes")
    if not nocache and audio_path.exists():
        try:
            if (audio_path.stat().st_size > 0
                    and audio_path.stat().st_mtime >= video_path.stat().st_mtime):
                return str(audio_path)
        except OSError:
            pass  # fall through and re-extract

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return str(audio_path)
