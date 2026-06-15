"""
Detect burned-in (hardcoded) subtitles from a video via OCR.

Most of the source videos here are Chinese-language with English subtitles
*painted into the picture* (no soft subtitle stream — verified with ffprobe).
Whisper can only guess at the speech; reading the on-screen English text is far
more faithful. This stage scans frames in the subtitle band, OCRs them with
PaddleOCR, and collapses repeated frames into timed segments — the same
{start, end, text} dicts the rest of the pipeline already consumes.

The output feeds the existing English -> Khmer translator (source_lang="en"),
which is a higher-quality pair than Chinese -> Khmer.
"""
import os
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional, Any

import cv2

# PaddleOCR is heavy to construct (loads detection + recognition models), so
# build one instance per language and reuse it across calls.
_ocr_cache: dict = {}

# Map our mode codes -> PaddleOCR language codes.
_LANG = {"en": "en", "zh": "ch", "ch": "ch"}


def _get_ocr(lang: str = "en"):
    code = _LANG.get(lang, "en")
    if code not in _ocr_cache:
        from paddleocr import PaddleOCR
        # use_angle_cls=False: subtitles are horizontal, skip the angle classifier.
        _ocr_cache[code] = PaddleOCR(use_angle_cls=False, lang=code, show_log=False)
    return _ocr_cache[code]


def _ocr_text(ocr, image, min_conf: float) -> str:
    """OCR a cropped frame, return the detected text joined top-to-bottom."""
    try:
        result = ocr.ocr(image, cls=False)
    except Exception:
        return ""
    if not result or not result[0]:
        return ""
    lines = []
    for line in result[0]:
        try:
            box, (txt, conf) = line[0], line[1]
        except (ValueError, TypeError):
            continue
        if conf is None or conf < min_conf:
            continue
        txt = (txt or "").strip()
        if not txt:
            continue
        y_top = min(p[1] for p in box)  # order multi-line subs by vertical position
        lines.append((y_top, txt))
    lines.sort(key=lambda t: t[0])
    return " ".join(t for _, t in lines)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _similar(a: str, b: str, threshold: float) -> bool:
    if not a or not b:
        return False
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio() >= threshold


def _spans_from_samples(samples: List, interval: float, duration: float,
                        threshold: float) -> List[Dict[str, Any]]:
    """Collapse a time-ordered list of (t, text) samples into timed segments.

    Consecutive samples whose text is identical (or fuzzily similar, to absorb
    OCR jitter) are merged into a single span; empty samples close the span.
    """
    segs: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for t, text in samples:
        if not text:
            if cur:
                segs.append(cur)
                cur = None
            continue
        if cur and _similar(cur["text"], text, threshold):
            cur["end"] = t + interval
            # Keep the longest observed variant — usually the most complete read.
            if len(text) > len(cur["text"]):
                cur["text"] = text
        else:
            if cur:
                segs.append(cur)
            cur = {"start": t, "end": t + interval, "text": text}
    if cur:
        segs.append(cur)

    out: List[Dict[str, Any]] = []
    for s in segs:
        start = round(s["start"], 2)
        end = s["end"]
        if duration:
            end = min(end, duration)
        end = round(end, 2)
        if end <= start:
            end = round(start + interval, 2)
        out.append({"start": start, "end": end, "text": s["text"]})
    return out


def detect_subtitles(
    video_path: str,
    lang: str = "en",
    sample_interval: float = 0.5,
    crop_top: float = 0.70,
    crop_bottom: float = 1.0,
    min_conf: float = 0.5,
    similarity: float = 0.8,
    progress: Optional[Callable[[float], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Scan a video for burned-in subtitles and return timed segments.

    sample_interval: seconds between sampled frames (smaller = finer timing, slower).
    crop_top/crop_bottom: vertical band (fraction of height) to OCR — defaults to
        the bottom 30%, where subtitles normally sit, which also speeds OCR up and
        avoids grabbing unrelated on-screen text.
    min_conf: drop OCR results below this confidence.
    similarity: fuzzy-match ratio for treating two reads as the same subtitle.
    progress: optional callback receiving a 0..1 fraction.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (frame_count / fps) if fps else 0.0

    ocr = _get_ocr(lang)

    samples: List = []
    t = 0.0
    try:
        while duration == 0 or t <= duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            y0, y1 = int(h * crop_top), int(h * crop_bottom)
            crop = frame[y0:y1, 0:w]
            samples.append((t, _ocr_text(ocr, crop, min_conf)))
            if progress and duration:
                progress(min(t / duration, 1.0))
            t += sample_interval
    finally:
        cap.release()

    return _spans_from_samples(samples, sample_interval, duration, similarity)
