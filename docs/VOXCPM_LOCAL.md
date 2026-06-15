# Running VoxCPM2 locally (`TTS_ENGINE=voxcpm_local`)

This guide explains how to dub videos using **VoxCPM2 running directly on your
own GPU PC** — no remote endpoint, no API keys. It produces real, natural Khmer
voices and supports **voice cloning** from a clip you record in the browser.

> **You need an NVIDIA GPU** (~8 GB VRAM recommended). It will run on CPU but is
> very slow (tens of seconds+ per subtitle segment), so CPU is for testing only.

---

## 1. One-time setup (on the GPU PC)

Run the whole app (API + Celery worker + Redis) on the GPU PC.

```bash
# 1. Get the code and base dependencies
cd backend
pip install -r requirements.txt

# 2. Add the local-VoxCPM dependencies (torch + voxcpm)
pip install -r requirements-voxcpm.txt
```

If the default `torch` wheel doesn't match your GPU's CUDA version, install the
matching one first, e.g. for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-voxcpm.txt
```

The VoxCPM2 weights (~5 GB) download automatically from Hugging Face on first
use and are cached in `~/.cache/huggingface`.

---

## 2. Turn the engine on

Edit `.env` in the repo root:

```env
TTS_ENGINE=voxcpm_local
```

Optional knobs (defaults shown):

| Variable | Default | Meaning |
|---|---|---|
| `VOXCPM_LOCAL_MODEL` | `openbmb/VoxCPM2` | Hugging Face model id |
| `VOXCPM_LOCAL_DEVICE` | auto (`cuda` if available, else `cpu`) | Force `cuda` or `cpu` |
| `TTS_CONCURRENCY` | `1` | Segments synthesized at once. Leave at 1 — a single GPU can't run model calls in parallel |
| `VOXCPM_LOCAL_REF_TEXT` | a built-in Khmer sentence | Text used to bootstrap the default voices |

---

## 3. Verify it works (recommended before a real render)

```bash
python3 test_voxcpm_local.py                       # default female voice
python3 test_voxcpm_local.py --voice local_male    # default male voice
```

You should see torch/CUDA info, the `voxcpm` version + methods, then:

```
[test] SUCCESS — listen to the file to confirm the voice sounds right.
```

It writes `voxcpm_local_test.wav` — play it to hear the voice. If it errors,
the message points at the fix; send the output along and it can be corrected.

---

## 4. Run the app

Three processes (each in its own terminal):

```bash
redis-server                                              # 1. broker
cd backend && celery -A worker worker --loglevel=info     # 2. worker (loads the model)
cd backend && uvicorn main:api --reload                   # 3. API
```

Open <http://localhost:8000>.

> The model loads into VRAM on the **first** synthesis and stays resident, so
> the first segment of the first render is slower; everything after is fast.

---

## 5. Using it in the UI

1. **Upload** a Chinese-language video → it transcribes and translates to Khmer.
2. Edit the subtitle text/timing as usual.
3. **Pick voices.** Under this engine the voice list contains:
   - **`Khmer — Female (default)`** and **`Khmer — Male (default)`** — ready to
     use, no setup. (These are bootstrapped automatically from gTTS the first
     time, so they always work.)
   - **Any voices you cloned** (see below).
4. **Render.** Each segment is synthesized by VoxCPM2, time-stretched to fit its
   subtitle window, and muxed over the ducked original audio.

### Voice cloning (the main reason to use this engine)

1. In **VOICE SETTINGS**, open **"🎙 Clone a voice (record)"**.
2. Record a short clear sample of the target voice in the browser and save it.
3. The cloned voice now appears in the voice list — assign it to segments (or to
   a detected speaker in the **Speaker voices** panel) and render. VoxCPM2 clones
   that voice locally for the Khmer dub.

---

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `could not import voxcpm` | Dependencies not installed → `pip install -r backend/requirements-voxcpm.txt` |
| `VoxCPM.from_pretrained: MISSING` or synthesis throws on first call | Installed `voxcpm` API differs from what the code expects → adjust `_get_model` / `_generate_wav` in `backend/pipeline/voxcpm_local.py` (paste the test output to get the exact fix) |
| Extremely slow, "running on CPU" warning | No CUDA detected → install the CUDA `torch` build and/or set `VOXCPM_LOCAL_DEVICE=cuda` |
| Out-of-memory (CUDA OOM) | GPU has too little VRAM → close other GPU apps, or use a smaller model via `VOXCPM_LOCAL_MODEL` |
| Cloned voice sounds like the default | The clone reference is only used under `voxcpm_local` (and the remote `voxcpm` engine); make sure `TTS_ENGINE=voxcpm_local` and the worker was restarted after the change |

---

## Notes

- Switching engines: set `TTS_ENGINE` back to `gtts` / `azure` / `gemini` and
  restart the worker. Saved per-segment voice selections survive the switch
  (unknown ids fall back to a same-gender default).
- This engine needs **no** API keys (`GEMINI_API_KEY`, `AZURE_SPEECH_KEY`, etc.
  are unused here).
- See `backend/pipeline/voxcpm_local.py` for the implementation and
  `test_voxcpm_local.py` for the standalone check.
