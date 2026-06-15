#!/usr/bin/env python3
"""
Standalone check for the in-process VoxCPM2 engine (TTS_ENGINE=voxcpm_local).
Run this on the GPU PC BEFORE wiring it into a full render — it confirms the
model loads, the `voxcpm` Python API matches what voxcpm_local.py expects, and
that synthesis + the gTTS-bootstrapped default reference actually work.

    cd backend
    pip install -r requirements.txt
    pip install -r requirements-voxcpm.txt
    cd ..
    python3 test_voxcpm_local.py                      # default female voice
    python3 test_voxcpm_local.py --voice local_male   # default male
    python3 test_voxcpm_local.py --text "ខ្ញុំស្រឡាញ់ភាសាខ្មែរ" --out hello.wav

Writes a WAV you can listen to and prints its duration. Exit code 0 = success.
"""
import argparse
import os
import sys
import wave
from pathlib import Path

# Run from the repo root; make `backend` importable and default DATA_DIR local.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("TTS_ENGINE", "voxcpm_local")
os.environ.setdefault("DATA_DIR", str(ROOT / "data"))


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke-test the local VoxCPM2 TTS engine.")
    p.add_argument("--text", default="សួស្តី នេះគឺជាការសាកល្បងសំឡេង VoxCPM2 ជាភាសាខ្មែរ។",
                   help="Khmer text to synthesize.")
    p.add_argument("--voice", default="local_female",
                   help="Voice id: local_female, local_male, or a clone_* id.")
    p.add_argument("--out", default="voxcpm_local_test.wav", help="Output WAV path.")
    p.add_argument("--list-voices", action="store_true",
                   help="List available voice ids (incl. clones) and exit.")
    p.add_argument("--set-gender", choices=["male", "female"],
                   help="Fix the gender of the clone given by --voice, then exit.")
    args = p.parse_args()

    if args.set_gender:
        from pipeline import voice_clone
        updated = voice_clone.update_clone(args.voice, gender=args.set_gender)
        if updated is None:
            print(f"[test] no such cloned voice: {args.voice}")
            return 1
        print(f"[test] updated {updated['id']} → gender={updated['gender']} "
              f"({updated['label']})")
        return 0

    if args.list_voices:
        from pipeline import tts
        print("[test] available voices for this engine:")
        for v in tts.list_voices():
            tag = " (cloned)" if v.get("cloned") else ""
            print(f"  {v['id']:24s}  {v.get('gender',''):6s}  {v.get('label','')}{tag}")
        return 0

    print(f"[test] TTS_ENGINE = {os.environ['TTS_ENGINE']}")
    print(f"[test] model      = {os.environ.get('VOXCPM_LOCAL_MODEL', 'openbmb/VoxCPM2')}")
    print(f"[test] voice      = {args.voice}")

    try:
        import torch
        print(f"[test] torch {torch.__version__} | CUDA available = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[test] GPU = {torch.cuda.get_device_name(0)}")
        else:
            print("[test] WARNING: no CUDA — this will be very slow on CPU.")
    except Exception as e:
        print(f"[test] could not import torch: {e}")

    # ── voxcpm package diagnostics ───────────────────────────────────────────
    # Print version + public API so a signature mismatch with voxcpm_local.py is
    # obvious in one shot (versions vary; this is the most likely failure point).
    try:
        import importlib.metadata as _md
        import voxcpm as _vox
        try:
            ver = _md.version("voxcpm")
        except Exception:
            ver = getattr(_vox, "__version__", "unknown")
        print(f"[test] voxcpm version = {ver}")
        mod_api = [n for n in dir(_vox) if not n.startswith("_")]
        print(f"[test] voxcpm module exports = {mod_api}")
        cls = getattr(_vox, "VoxCPM", None)
        if cls is None:
            print("[test] WARNING: voxcpm has no `VoxCPM` class — "
                  "update _get_model() in voxcpm_local.py to the real entrypoint.")
        else:
            methods = [n for n in dir(cls) if not n.startswith("_")]
            print(f"[test] VoxCPM methods = {methods}")
            for name in ("from_pretrained", "generate"):
                mark = "OK" if hasattr(cls, name) else "MISSING"
                print(f"[test]   VoxCPM.{name}: {mark}")
    except Exception as e:
        print(f"[test] could not import voxcpm: {type(e).__name__}: {e}")
        print("[test] install it on the GPU PC: "
              "pip install -r backend/requirements-voxcpm.txt")
        return 1

    from pipeline import voxcpm_local

    print("[test] synthesizing (first call also loads the model — may take a while)…")
    gender = "male" if args.voice == "local_male" else "female"
    try:
        wav_bytes = voxcpm_local.synthesize(args.text, args.voice, gender)
    except Exception as e:
        print(f"[test] FAILED during synthesis: {type(e).__name__}: {e}")
        print("[test] If this is an API mismatch, adjust _generate_wav/_get_model "
              "in backend/pipeline/voxcpm_local.py to match the installed voxcpm.")
        return 1

    Path(args.out).write_bytes(wav_bytes)
    with wave.open(args.out, "rb") as w:
        dur = w.getnframes() / float(w.getframerate())
        print(f"[test] wrote {args.out} | {len(wav_bytes)} bytes | "
              f"{w.getframerate()} Hz | {dur:.2f}s")
    print("[test] SUCCESS — listen to the file to confirm the voice sounds right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
