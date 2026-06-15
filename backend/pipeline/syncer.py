import os
from pathlib import Path
from pydub import AudioSegment
from typing import List, Dict, Any, Tuple

# Background (original Chinese audio) levels, in dB attenuation
BG_IDLE_DB   = float(os.environ.get("BG_IDLE_DB",   "8"))   # in gaps between speech
BG_SPEECH_DB = float(os.environ.get("BG_SPEECH_DB", "18"))  # under Khmer TTS (ducked)
DUCK_PAD_MS  = 150   # ducking starts slightly before / ends after each clip
FADE_MS      = 120   # fade length at each duck boundary


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def build_audio_track(
    segments: List[Dict[str, Any]],
    original_audio_path: str,
    output_dir: str,
    job_id: str,
    remove_original: bool = False,
) -> str:
    """
    Build the final mixed audio track:
    - Original Chinese audio kept as background, ducked under each TTS clip
      (BG_IDLE_DB in gaps, BG_SPEECH_DB during speech, with short fades)
    - Khmer TTS clips placed at exact original timestamps
    Each TTS clip was already time-stretched to fit its slot in tts.py,
    so no further speed adjustment is needed here.

    If remove_original is True, the original audio is dropped entirely and the
    track contains only the Khmer TTS clips over silence.
    """
    if not os.path.exists(original_audio_path):
        raise FileNotFoundError(f"Original audio not found: {original_audio_path}")

    original = AudioSegment.from_wav(original_audio_path)
    total_ms = len(original)

    khmer_track = AudioSegment.silent(duration=total_ms)
    speech_iv: List[Tuple[int, int]] = []

    placed = truncated = 0
    for seg in segments:
        tts_path = seg.get("tts_path")
        if not tts_path or not os.path.exists(tts_path):
            continue

        start_ms = int(seg["start"] * 1000)
        clip     = AudioSegment.from_wav(tts_path)

        # Clip must not overflow total duration
        if start_ms + len(clip) > total_ms:
            cut = start_ms + len(clip) - total_ms
            print(f"[syncer] WARNING: clip at {seg['start']:.2f}s truncated by {cut}ms")
            clip = clip[: total_ms - start_ms]
            truncated += 1

        khmer_track = khmer_track.overlay(clip, position=start_ms)
        speech_iv.append((
            max(0, start_ms - DUCK_PAD_MS),
            min(total_ms, start_ms + len(clip) + DUCK_PAD_MS),
        ))
        placed += 1

    print(f"[syncer] placed {placed}/{len(segments)} TTS clips"
          + (f", {truncated} truncated" if truncated else ""))

    # Drop the original audio entirely: export only the Khmer TTS over silence.
    if remove_original:
        print("[syncer] removing original audio — Khmer voice-over only")
        out_path = str(Path(output_dir) / f"{job_id}_khmer_audio.wav")
        khmer_track.export(out_path, format="wav")
        return out_path

    # Build the ducked background: louder in gaps, quieter under speech,
    # short edge fades on each region to avoid clicks.
    bg = AudioSegment.silent(duration=total_ms)
    cursor = 0
    regions = []  # (start, end, attenuation)
    for a, b in _merge_intervals(speech_iv):
        if a > cursor:
            regions.append((cursor, a, BG_IDLE_DB))
        regions.append((a, b, BG_SPEECH_DB))
        cursor = b
    if cursor < total_ms:
        regions.append((cursor, total_ms, BG_IDLE_DB))

    for a, b, att in regions:
        piece = original[a:b] - att
        fade = min(FADE_MS, max(0, (b - a) // 2))
        if fade:
            piece = piece.fade_in(fade).fade_out(fade)
        bg = bg.overlay(piece, position=a)

    mixed    = bg.overlay(khmer_track)
    out_path = str(Path(output_dir) / f"{job_id}_khmer_audio.wav")
    mixed.export(out_path, format="wav")
    return out_path
