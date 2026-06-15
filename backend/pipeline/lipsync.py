import subprocess
import os
from pathlib import Path


WAV2LIP_DIR = os.environ.get("WAV2LIP_DIR", "/app/Wav2Lip")
CHECKPOINT = os.environ.get(
    "WAV2LIP_CHECKPOINT",
    "/app/Wav2Lip/checkpoints/wav2lip_gan.pth",
)


def run_lipsync(
    video_path: str,
    audio_path: str,
    output_dir: str,
    job_id: str,
) -> str:
    """
    Run Wav2Lip on the input video + Khmer audio track.
    Returns path to the lip-synced output video.
    """
    out_path = str(Path(output_dir) / f"{job_id}_lipsync.mp4")

    inference_script = str(Path(WAV2LIP_DIR) / "inference.py")

    cmd = [
        "python", inference_script,
        "--checkpoint_path", CHECKPOINT,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", out_path,
        "--nosmooth",
    ]

    result = subprocess.run(
        cmd,
        cwd=WAV2LIP_DIR,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Wav2Lip failed:\n{result.stderr}")

    return out_path


def mux_audio_only(
    video_path: str,
    audio_path: str,
    output_dir: str,
    job_id: str,
) -> str:
    """
    Fallback: replace the audio track without lip sync (pure ffmpeg mux).
    Used when Wav2Lip is unavailable or fails.
    """
    out_path = str(Path(output_dir) / f"{job_id}_dubbed.mp4")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-ac", "2",
            "-ar", "44100",
            "-c:a", "aac",
            "-b:a", "192k",
            out_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return out_path


def _probe_dimensions(video_path: str) -> tuple:
    """Return (width, height) of the video's first video stream."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            video_path,
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def _ff_color(color: str) -> str:
    """Normalise a color for ffmpeg's `drawbox`: '#RRGGBB' → '0xRRGGBB',
    named colors ('white', 'black') and existing '0x...' pass through."""
    c = (color or "white").strip()
    if c.startswith("#"):
        return "0x" + c[1:]
    return c


def remove_subtitles(
    video_path: str,
    output_dir: str,
    job_id: str,
    region: dict = None,
    spans: list = None,
    color: str = "white",
) -> str:
    """
    Hide burned-in (hardcoded) subtitles by painting a solid, fully-opaque bar
    over the subtitle band with ffmpeg's `drawbox` filter (t=fill). Unlike
    interpolation (delogo), which smears and often leaves the text legible, a
    filled bar cleanly covers the original Chinese/English CC.

    region: fractions of the frame {x, y, w, h}; defaults to a full-width bottom
        band (where drama subtitles sit). drawbox needs no edge margin, so the
        bar can run to the frame edges for full coverage.
    spans: optional list of (start, end) seconds — the bar is only drawn during
        these windows (timeline-gated). None draws it across the whole video,
        which guarantees any CC outside the subtitle windows is also covered.
    color: bar fill color — an ffmpeg color name or '#RRGGBB' hex.
    """
    out_path = str(Path(output_dir) / f"{job_id}_nosub.mp4")
    W, H = _probe_dimensions(video_path)

    region = region or {}
    fx = float(region.get("x", 0.0))
    fy = float(region.get("y", 0.82))
    fw = float(region.get("w", 1.0))
    fh = float(region.get("h", 0.16))

    x = max(0, int(W * fx))
    y = max(0, int(H * fy))
    w = int(W * fw)
    h = int(H * fh)
    if x + w > W:
        w = W - x
    if y + h > H:
        h = H - y

    vf = f"drawbox=x={x}:y={y}:w={w}:h={h}:color={_ff_color(color)}:t=fill"
    if spans:
        enable = "+".join(f"between(t,{s:.2f},{e:.2f})" for s, e in spans)
        vf += f":enable='{enable}'"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "copy",
            out_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return out_path


def to_horizontal(
    video_path: str,
    output_dir: str,
    job_id: str,
    width: int = 1920,
    height: int = 1080,
) -> str:
    """
    Convert any video into a horizontal 16:9 frame (default 1920x1080).

    Vertical/portrait source (common for the drama clips) is letterboxed onto a
    landscape canvas: the original is scaled to fit fully inside the frame and
    centred over a blurred, zoomed copy of itself so the side bars aren't dead
    black. Already-landscape source is just normalised to the target size.
    Audio is copied through untouched.
    """
    out_path = str(Path(output_dir) / f"{job_id}_horizontal.mp4")

    W, H = _probe_dimensions(video_path)
    # Already horizontal at (or wider than) 16:9 and not larger than target:
    # leave the picture geometry alone, just remux/normalise the container.
    if W >= H:
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
        )
    else:
        # Blurred fill background + the upright clip centred on top.
        vf = (
            f"split[bg][fg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=20:5[bg];"
            f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "copy",
            out_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return out_path


def _subtitles_filter(srt_path: str, style: dict = None) -> str:
    """Build the ffmpeg `subtitles=...` filter string for burning an SRT."""
    style = style or {}
    font    = style.get("font_name") or "Khmer OS"
    size    = int(style.get("font_size") or 22)
    color   = _hex_to_ass(style.get("font_color") or "FFFFFF")
    outline = int(style.get("outline") or 2)
    margin  = int(style.get("margin_v") or 40)
    force_style = (
        f"FontName={font},FontSize={size},PrimaryColour={color},"
        f"OutlineColour=&H00000000,Outline={outline},Shadow=1,Alignment=2,"
        f"MarginV={margin}"
    )
    escaped = srt_path.replace("'", r"\'").replace(":", r"\:")
    return f"subtitles='{escaped}':force_style='{force_style}'"


def _drawbox_filter(W: int, H: int, region: dict = None, color: str = "white") -> str:
    """Build the `drawbox=...` filter that covers the original burned-in subs."""
    region = region or {}
    fx = float(region.get("x", 0.0))
    fy = float(region.get("y", 0.82))
    fw = float(region.get("w", 1.0))
    fh = float(region.get("h", 0.16))

    x = max(0, int(W * fx))
    y = max(0, int(H * fy))
    w = int(W * fw)
    h = int(H * fh)
    if x + w > W:
        w = W - x
    if y + h > H:
        h = H - y
    return f"drawbox=x={x}:y={y}:w={w}:h={h}:color={_ff_color(color)}:t=fill"


def finalize_video(
    video_path: str,
    audio_path: str,
    output_dir: str,
    job_id: str,
    remove_subs: bool = False,
    remove_region: dict = None,
    remove_color: str = "white",
    srt_path: str = None,
    burn_subs: bool = False,
    sub_style: dict = None,
    width: int = 1920,
    height: int = 1080,
    preset: str = "fast",
    crf: int = 18,
) -> str:
    """
    Produce the final video in a SINGLE ffmpeg encode: swap in the Khmer audio,
    optionally cover the original burned-in subtitles, optionally burn the Khmer
    SRT, and letterbox into a horizontal 16:9 frame.

    Previously these were 3-4 separate ffmpeg passes, each fully re-encoding the
    video (stacking lossy generations). Collapsing them into one filtergraph is
    both much faster and higher quality (one encode generation, not three). The
    video filters run in the same order as before: cover originals → burn Khmer
    → letterbox, so subtitle geometry/appearance is unchanged.
    """
    out_path = str(Path(output_dir) / f"{job_id}_final.mp4")
    W, H = _probe_dimensions(video_path)

    # Filters applied to the source frame BEFORE letterboxing (so subtitles sit
    # on the real picture, not the blurred side-bars), in legacy order.
    pre = []
    if remove_subs:
        pre.append(_drawbox_filter(W, H, remove_region, remove_color))
    if burn_subs and srt_path:
        pre.append(_subtitles_filter(srt_path, sub_style))

    audio_args = ["-map", "1:a:0", "-ac", "2", "-ar", "44100",
                  "-c:a", "aac", "-b:a", "192k"]
    common = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]

    if W >= H:
        # Already landscape: cover/burn, then normalise into the target frame.
        pre.append(
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
        )
        vf = ",".join(pre)
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
               "-vf", vf, "-map", "0:v:0", *audio_args, *common, out_path]
    else:
        # Portrait: cover/burn on the upright clip, then centre it over a
        # blurred, zoomed copy of itself filling the landscape canvas.
        pre_part = (",".join(pre) + ",") if pre else ""
        fc = (
            f"[0:v]{pre_part}split[bg][fg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=20:5[bg];"
            f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
        )
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
               "-filter_complex", fc, "-map", "[v]", *audio_args, *common, out_path]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def _hex_to_ass(color: str) -> str:
    """'RRGGBB' (or '#RRGGBB') → ASS '&H00BBGGRR' format."""
    c = color.lstrip("#")
    if len(c) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in c):
        c = "FFFFFF"
    return f"&H00{c[4:6]}{c[2:4]}{c[0:2]}".upper()


def burn_subtitles(
    video_path: str,
    srt_path: str,
    output_dir: str,
    job_id: str,
    style: dict = None,
) -> str:
    """Burn Khmer SRT subtitles into the video using ffmpeg subtitles filter."""
    out_path = str(Path(output_dir) / f"{job_id}_burned.mp4")

    style = style or {}
    font    = style.get("font_name") or "Khmer OS"
    size    = int(style.get("font_size") or 22)
    color   = _hex_to_ass(style.get("font_color") or "FFFFFF")
    outline = int(style.get("outline") or 2)
    margin  = int(style.get("margin_v") or 40)
    force_style = (
        f"FontName={font},FontSize={size},PrimaryColour={color},"
        f"OutlineColour=&H00000000,Outline={outline},Shadow=1,Alignment=2,"
        f"MarginV={margin}"
    )

    escaped = srt_path.replace("'", r"\'").replace(":", r"\:")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"subtitles='{escaped}':force_style='{force_style}'",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "copy",
            out_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return out_path
