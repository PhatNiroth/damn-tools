import os
import re
import json
import time
import urllib.parse
import urllib.request
from typing import List, Dict, Any


# Map our mode codes → the source-language labels each backend expects.
_SRC_LABEL  = {"zh": "Chinese (Mandarin)", "en": "English"}   # for the Gemini prompt
_SRC_GOOGLE = {"zh": "zh-CN",              "en": "en"}         # for the Google `sl` param

# Gemini model used for translation. 2.5-flash is markedly better than 2.0 at
# low-resource languages like Khmer; override with GEMINI_TRANSLATE_MODEL.
_GEMINI_MODEL = os.environ.get("GEMINI_TRANSLATE_MODEL", "gemini-2.5-flash")

# How many segments we ask the model to translate per request. Keeping this
# bounded prevents the *output* from being truncated on long videos (the old
# code sent everything at once, so the tail silently fell back to Chinese).
_CHUNK = int(os.environ.get("GEMINI_TRANSLATE_CHUNK", "40"))
# Extra neighbouring source lines shown as read-only context so the model can
# translate Whisper's mid-sentence fragments coherently.
_CTX_BEFORE = 6
_CTX_AFTER  = 3

# Khmer Unicode block — used to sanity-check that we actually got Khmer back.
_KHMER_RE = re.compile(r"[ក-៿]")


def _is_khmer(text: str) -> bool:
    return bool(text) and _KHMER_RE.search(text) is not None


def translate_segments(segments: List[Dict[str, Any]],
                       source_lang: str = "zh") -> List[Dict[str, Any]]:
    """
    Translate source-language segments to Khmer.
    source_lang: "zh" (Chinese, default) or "en" (English-pivot flow).
    Priority: Gemini API → Google Translate free endpoint → passthrough.
    Whichever backend runs, any segment that still lacks real Khmer is retried
    individually rather than silently left as the source text.
    """
    if source_lang not in _SRC_LABEL:
        source_lang = "zh"

    result = [dict(seg) for seg in segments]

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        try:
            _translate_gemini(result, api_key, source_lang, key="khmer")
        except Exception as e:
            print(f"[translator] Gemini failed: {e}")

    # Fill any gaps (Gemini absent, errored, or returned non-Khmer for a row)
    # with the Google free endpoint, then per-segment as a last resort.
    if _needs_fill(result, "khmer"):
        try:
            _translate_google_batch(result, source_lang, key="khmer", tl="km")
        except Exception as e:
            print(f"[translator] Google free failed: {e}")

    if _needs_fill(result, "khmer"):
        _translate_google_simple(result, source_lang, key="khmer", tl="km")

    # Anything still untranslated: mark it instead of passing Chinese off as Khmer.
    for seg in result:
        if not _is_khmer(seg.get("khmer", "")):
            seg["khmer"] = seg.get("khmer") or seg.get("text", "")
            seg["translation_failed"] = True
    return result


def translate_to_english(segments: List[Dict[str, Any]],
                         source_lang: str = "zh") -> List[Dict[str, Any]]:
    """
    Translate source-language segments to English (for a full-video English
    subtitle preview). Adds an `english` key to each segment.
    """
    if source_lang not in _SRC_LABEL:
        source_lang = "zh"

    # If the source is already English, just mirror the text across.
    if source_lang == "en":
        return [{**seg, "english": seg.get("text", "")} for seg in segments]

    result = [dict(seg) for seg in segments]

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        try:
            _translate_gemini(result, api_key, source_lang, key="english")
        except Exception as e:
            print(f"[translator] Gemini (EN) failed: {e}")

    if _needs_fill(result, "english"):
        try:
            _translate_google_batch(result, source_lang, key="english", tl="en")
        except Exception as e:
            print(f"[translator] Google free (EN) failed: {e}")

    for seg in result:
        seg.setdefault("english", seg.get("text", ""))
    return result


def _needs_fill(segments: List[Dict[str, Any]], key: str) -> bool:
    if key == "khmer":
        return any(not _is_khmer(s.get("khmer", "")) for s in segments)
    return any(not s.get(key, "").strip() for s in segments)


# ── Gemini ────────────────────────────────────────────────────────────────────

def _translate_gemini(segments, api_key, source_lang, key="khmer"):
    """Translate in chunks, mutating `segments` in place. Uses JSON output keyed
    by segment index so a single malformed line can't shift every translation,
    and only writes back rows that came back in the target language."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(_GEMINI_MODEL)

    src     = _SRC_LABEL.get(source_lang, "Chinese (Mandarin)")
    target  = "Khmer (ភាសាខ្មែរ)" if key == "khmer" else "English"
    n = len(segments)

    for start in range(0, n, _CHUNK):
        end = min(start + _CHUNK, n)
        out = _gemini_chunk(model, segments, start, end, src, target, key)
        for idx, value in out.items():
            if 0 <= idx < n and value:
                if key == "khmer" and not _is_khmer(value):
                    continue
                segments[idx][key] = value


def _gemini_chunk(model, segments, start, end, src, target, key):
    # Read-only context window so fragments split mid-sentence translate coherently.
    ctx_lo = max(0, start - _CTX_BEFORE)
    ctx_hi = min(len(segments), end + _CTX_AFTER)
    context_lines = []
    for i in range(ctx_lo, ctx_hi):
        tag = "TRANSLATE" if start <= i < end else "context"
        context_lines.append(f"[{i}] ({tag}) {segments[i].get('text', '')}")
    block = "\n".join(context_lines)

    natural = ("natural, conversational, spoken Khmer as a Cambodian would say it"
               if key == "khmer" else "natural English")
    extra = ("Do NOT transliterate names or words into Khmer letter-by-letter; "
             "render meaning. Keep it idiomatic, not word-for-word. "
             if key == "khmer" else "")

    prompt = (
        f"You are a professional subtitle translator, {src} → {target}.\n"
        f"The lines below are consecutive subtitle segments from one video, so "
        f"use the surrounding lines as context — sentences are often split across "
        f"several segments. Translate ONLY the lines marked (TRANSLATE) into "
        f"{natural}. Preserve the speaker's tone and meaning; {extra}"
        f"if a segment is just a fragment, translate it as the fitting part of "
        f"the larger sentence.\n\n"
        f"Return ONLY a JSON object mapping each (TRANSLATE) line's number to its "
        f'translation, e.g. {{"{start}": "...", "{start+1}": "..."}}.\n\n'
        f"{block}"
    )

    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json",
                           "temperature": 0.3},
    )
    return _parse_json_map(response.text)


def _parse_json_map(text: str) -> Dict[int, str]:
    """Parse the model's JSON reply into {index: translation}. Tolerant of code
    fences and stray prose around the JSON object."""
    text = (text or "").strip()
    if not text:
        return {}
    # Strip ```json ... ``` fences if present.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except Exception:
        return {}
    out: Dict[int, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                out[int(k)] = str(v).strip()
            except (ValueError, TypeError):
                continue
    return out


# ── Google Translate free endpoint ─────────────────────────────────────────────

def _google_call(text: str, sl: str, tl: str, timeout: int = 15) -> str:
    encoded = urllib.parse.quote(text)
    url = (
        "https://translate.googleapis.com/translate_a/single"
        f"?client=gtx&sl={sl}&tl={tl}&dt=t&q={encoded}"
    )
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return "".join(part[0] for part in data[0] if part and part[0])


def _translate_google_batch(segments, source_lang, key, tl):
    """Fill missing rows via the free endpoint. We translate each still-missing
    segment on its own line in a small batch using a newline join (Google keeps
    newlines), avoiding the old `||||` delimiter that the translator often ate."""
    sl = _SRC_GOOGLE.get(source_lang, "zh-CN")
    todo = [i for i, s in enumerate(segments) if _row_missing(s, key)]
    for batch_start in range(0, len(todo), 20):
        idxs = todo[batch_start:batch_start + 20]
        combined = "\n".join(segments[i].get("text", "") for i in idxs)
        out = _google_call(combined, sl, tl)
        parts = out.split("\n")
        if len(parts) == len(idxs):
            for j, i in enumerate(idxs):
                val = parts[j].strip()
                if val and (key != "khmer" or _is_khmer(val)):
                    segments[i][key] = val


def _translate_google_simple(segments, source_lang, key, tl):
    """Last resort: one request per still-missing segment."""
    sl = _SRC_GOOGLE.get(source_lang, "zh-CN")
    for i, seg in enumerate(segments):
        if not _row_missing(seg, key):
            continue
        text = seg.get("text", "")
        try:
            val = _google_call(text, sl, tl, timeout=10).strip()
            if val:
                segments[i][key] = val
            time.sleep(0.1)
        except Exception as e:
            print(f"[translator] segment '{text[:20]}' failed: {e}")


def _row_missing(seg: Dict[str, Any], key: str) -> bool:
    if key == "khmer":
        return not _is_khmer(seg.get("khmer", ""))
    return not seg.get(key, "").strip()
