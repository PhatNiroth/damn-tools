import os
import shutil
from celery import Celery
from pathlib import Path

from pipeline.extractor import extract_audio
from pipeline.transcriber import transcribe
from pipeline.diarizer import assign_speakers
from pipeline.sub_ocr import detect_subtitles
from pipeline.translator import translate_segments
from pipeline.tts import synthesize_segments
from pipeline.syncer import build_audio_track
from pipeline.lipsync import finalize_video
from pipeline.srt_parser import build_srt

REDIS_URL  = os.environ.get("REDIS_URL",  "redis://localhost:6379/0")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/app/uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")
# Final rendered videos are moved here — only the finished full video, not the
# intermediate .wav/.srt working files, which stay in OUTPUT_DIR.
RESULTS_DIR = os.environ.get("RESULTS_DIR", str(Path(OUTPUT_DIR) / "results"))

app = Celery("worker", broker=REDIS_URL, backend=REDIS_URL)
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]


@app.task(bind=True, name="extract_srt")
def extract_srt_task(self, job_id: str, video_filename: str, mode: str = "zh"):
    """
    Whisper-extract a video into segments (no Khmer translation here).

    mode="zh" (default): transcribe the Chinese speech as-is → `text` is Chinese.
    mode="en": run Whisper's translate task → `text` is English. The English-pivot
    flow then translates English → Khmer, which is a higher-quality language pair.
    """
    def update(stage, pct):
        self.update_state(state="PROGRESS", meta={"stage": stage, "pct": pct})

    task = "translate" if mode == "en" else "transcribe"
    video_path = str(Path(UPLOAD_DIR) / video_filename)
    try:
        update("Extracting audio", 10)
        audio_path = extract_audio(video_path, OUTPUT_DIR)

        label = "Transcribing → English (Whisper)" if mode == "en" else "Transcribing speech (Whisper)"
        update(label, 50)
        segments = transcribe(audio_path, task=task)

        # Tag each segment with a speaker (Man 1/2/3 / Girl) so the UI can assign
        # one voice per speaker. Best-effort: degrades to gender-only on failure.
        update("Detecting speakers", 85)
        segments = assign_speakers(audio_path, segments)

        update("Done", 100)
        return {"status": "done", "segments": segments, "source_lang": mode}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise


@app.task(bind=True, name="detect_subs")
def detect_subs_task(self, job_id: str, video_filename: str, lang: str = "en"):
    """
    Read burned-in (on-screen) subtitles from a video via OCR — no audio
    transcription. Returns segments whose `text` is the on-screen subtitle.

    Default lang="en": the source videos carry English subtitles, so the
    downstream flow translates English -> Khmer (a higher-quality pair than
    transcribing Chinese audio and translating zh -> km).
    """
    def update(stage, pct):
        self.update_state(state="PROGRESS", meta={"stage": stage, "pct": pct})

    video_path = str(Path(UPLOAD_DIR) / video_filename)
    try:
        update("Scanning frames for subtitles", 5)

        def on_progress(frac):
            update("Reading subtitles (OCR)", 5 + int(frac * 85))

        segments = detect_subtitles(video_path, lang=lang, progress=on_progress)

        update("Done", 100)
        return {"status": "done", "segments": segments, "source_lang": "en"}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise


@app.task(bind=True, name="preview_audio")
def preview_audio(self, job_id: str, video_filename: str, segments: list,
                  remove_audio: bool = True,
                  bgm_volume: float = 1.0, voice_volume: float = 1.0):
    """
    Quick quality preview: generate the Khmer TTS and mix the final audio track,
    but skip the expensive video render (mux / sub-cover / sub-burn / letterbox).
    Returns just the mixed WAV so the frontend can play it (in sync) over the
    original video — letting the user judge voice/timing/ducking before
    committing to a full render. The audio file stays in OUTPUT_DIR.
    """
    def update(stage, pct):
        self.update_state(state="PROGRESS", meta={"stage": stage, "pct": pct})

    video_path = str(Path(UPLOAD_DIR) / video_filename)
    try:
        update("Extracting original audio", 10)
        audio_path = extract_audio(video_path, OUTPUT_DIR)

        update("Generating Khmer voice (timed to subtitles)", 45)
        segments = synthesize_segments(
            segments, OUTPUT_DIR, job_id,
            original_audio_path=audio_path,
        )

        update("Mixing audio track", 85)
        khmer_audio = build_audio_track(segments, audio_path, OUTPUT_DIR, job_id,
                                        remove_original=remove_audio,
                                        bgm_volume=bgm_volume,
                                        voice_volume=voice_volume)

        update("Done", 100)
        return {
            "status": "done",
            "audio": Path(khmer_audio).name,
        }
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise


@app.task(bind=True, name="render_from_segments")
def render_from_segments(self, job_id: str, video_filename: str, segments: list,
                         burn_subs: bool = False, sub_style: dict = None,
                         remove_subs: bool = False, remove_audio: bool = True,
                         remove_region: dict = None, remove_color: str = "white",
                         remove_mode: str = "bar",
                         bgm_volume: float = 1.0, voice_volume: float = 1.0):
    """
    Given pre-translated segments (with 'khmer', 'start', 'end', 'gender'),
    generate Khmer TTS timed to each subtitle window and mux into the video.
    """
    def update(stage, pct):
        self.update_state(state="PROGRESS", meta={"stage": stage, "pct": pct})

    video_path = str(Path(UPLOAD_DIR) / video_filename)
    try:
        update("Extracting original audio", 10)
        audio_path = extract_audio(video_path, OUTPUT_DIR)

        update("Generating Khmer voice (timed to subtitles)", 40)
        segments = synthesize_segments(
            segments, OUTPUT_DIR, job_id,
            original_audio_path=audio_path,
        )

        update("Mixing audio track", 80)
        khmer_audio = build_audio_track(segments, audio_path, OUTPUT_DIR, job_id,
                                        remove_original=remove_audio,
                                        bgm_volume=bgm_volume,
                                        voice_volume=voice_volume)

        # Save Khmer SRT (written before the render so it can be burned in the
        # single combined encode below).
        srt_path = str(Path(OUTPUT_DIR) / f"{job_id}_khmer.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(build_srt(segments, use_khmer=True))

        # Single combined encode: swap in the Khmer audio, optionally cover the
        # original burned-in subs, optionally burn the Khmer SRT, and letterbox
        # to horizontal — all in one ffmpeg pass (was up to 4 re-encodes).
        update("Rendering final video", 90)
        output_video = finalize_video(
            video_path, khmer_audio, OUTPUT_DIR, job_id,
            remove_subs=remove_subs,
            remove_region=remove_region,
            remove_color=remove_color,
            remove_mode=remove_mode,
            srt_path=srt_path,
            burn_subs=burn_subs,
            sub_style=sub_style,
        )

        # Move only the finished full video into the results folder.
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        final_path = Path(RESULTS_DIR) / Path(output_video).name
        shutil.move(output_video, str(final_path))

        update("Done", 100)
        return {
            "status": "done",
            "output": final_path.name,
            "srt":    Path(srt_path).name,
        }
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise
