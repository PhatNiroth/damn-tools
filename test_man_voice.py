#!/usr/bin/env python3
"""End-to-end test: dub uploads/test-voice-1.mp4 with forced MALE Khmer voice."""
import json
import sys
import time
import urllib.request
import urllib.parse

BASE = "http://localhost:8000"
FILENAME = sys.argv[1] if len(sys.argv) > 1 else "test-voice-1.mp4"
VOICE    = sys.argv[2] if len(sys.argv) > 2 else ""   # e.g. male_2 (Extra Deep)


def post(path, data=None, form=None):
    if form is not None:
        body = urllib.parse.urlencode(form).encode()
        req = urllib.request.Request(BASE + path, data=body)
    else:
        req = urllib.request.Request(
            BASE + path, data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def poll(job_id, label):
    last = ""
    while True:
        with urllib.request.urlopen(f"{BASE}/api/status/{job_id}", timeout=60) as r:
            s = json.loads(r.read())
        if s["state"] == "done":
            print(f"[{label}] done", flush=True)
            return s
        if s["state"] == "error":
            print(f"[{label}] FAILED: {s.get('error')}", flush=True)
            sys.exit(1)
        msg = f"{s.get('stage','')} {s.get('pct','')}%"
        if msg != last:
            print(f"[{label}] {msg}", flush=True)
            last = msg
        time.sleep(5)


print("=== 1. Transcribe (Whisper) ===", flush=True)
job = post("/api/extract-srt", form={"filename": FILENAME})
res = poll(job["job_id"], "whisper")
segments = res["segments"]
print(f"got {len(segments)} segments", flush=True)

print("=== 2. Translate to Khmer ===", flush=True)
res = post("/api/translate", data={"segments": segments})
segments = res["segments"]
ok = sum(1 for s in segments if s.get("khmer"))
print(f"translated {ok}/{len(segments)} segments", flush=True)

print(f"=== 3. Render with MALE voice {VOICE or '(default)'} ===", flush=True)
for s in segments:
    s["gender"] = "male"
    if VOICE:
        s["voice"] = VOICE
res = post("/api/render", data={
    "job_id": "test-man", "filename": FILENAME,
    "segments": segments, "burn_subs": False,
})
final = poll(res["render_id"], "render")
print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)
print(f"OUTPUT: outputs/{final['output']}", flush=True)
