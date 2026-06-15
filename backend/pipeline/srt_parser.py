import re
from typing import List, Dict, Any


def parse_srt(text: str) -> List[Dict[str, Any]]:
    """Parse SRT content into list of segments."""
    segments = []
    blocks = re.split(r'\n\s*\n', text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        time_match = re.match(
            r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})',
            lines[1]
        )
        if not time_match:
            continue
        start = _srt_to_sec(time_match.group(1))
        end   = _srt_to_sec(time_match.group(2))
        text_content = ' '.join(lines[2:]).strip()
        segments.append({
            'start':   start,
            'end':     end,
            'text':    text_content,
            'khmer':   '',
            'gender':  'auto',
        })
    return segments


def build_srt(segments: List[Dict[str, Any]], use_khmer: bool = True) -> str:
    """Build SRT string from segments."""
    lines = []
    for i, seg in enumerate(segments, 1):
        content = seg.get('khmer', '') if use_khmer else seg.get('text', '')
        lines.append(f"{i}")
        lines.append(f"{_sec_to_srt(seg['start'])} --> {_sec_to_srt(seg['end'])}")
        lines.append(content)
        lines.append('')
    return '\n'.join(lines)


def _srt_to_sec(t: str) -> float:
    t = t.replace('.', ',')
    hms, ms = t.split(',')
    h, m, s = hms.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _sec_to_srt(s: float) -> str:
    s = max(0.0, s)
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    sc = int(s % 60)
    ms = int(round((s % 1) * 1000))
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"
