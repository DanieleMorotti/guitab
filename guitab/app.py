"""Local web app: load or record a recording, transcribe it, then view and play the tab in the browser.

    guitab-app [--port 8765] [--device auto] [--no-browser]      # then open http://127.0.0.1:8765

The page (guitab/web/index.html) sends the audio and the options to POST /api/transcribe; the server converts the audio
with ffmpeg, runs the same pipeline as the `guitab` command (guitab/pipeline.py) and keeps every result in
out/app/<time>_<name>/: audio.wav (the converted audio), <name>.gp5 and result.json (settings and summary). DELETE
/api/tabs/<id> removes one. The tab viewer and player is alphaTab (loaded from a CDN).
Only listens on 127.0.0.1; one transcription runs at a time.
"""

import argparse
import json
import re
import shutil
import subprocess
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .device import DEVICES
from .paths import MODELS, ROOT

WEB = Path(__file__).parent / "web"
OUT = ROOT / "out" / "app"
LOCK = threading.Lock()  # one model run at a time


def available_models():
    """Trained checkpoints with their own default onset threshold (the page's note-sensitivity starting value)."""
    from .inference import checkpoint_decoding  # loads torch on the first call

    return [{"id": k, "label": label, "onset": checkpoint_decoding(path)["onset"]}
            for k, (path, label) in MODELS.items() if path.exists()]


def options(query):
    """Validated transcription settings from the request's query string."""
    q = {k: v[-1] for k, v in parse_qs(query).items()}
    model = q.get("model", "")
    if model not in MODELS or not MODELS[model][0].exists():
        raise ValueError(f"unknown model {model!r}")
    capo = int(q.get("capo", 0))
    if not 0 <= capo <= 12:
        raise ValueError("capo must be between 0 and 12")
    fingering = q.get("fingering", "model")
    if fingering not in ("model", "auto", "low"):
        raise ValueError("fingering must be model, auto or low")
    tempo = q.get("tempo", "auto") or "auto"
    if tempo != "auto" and not 30 <= int(tempo) <= 300:
        raise ValueError("tempo must be auto or 30-300 BPM")
    grid = int(q.get("grid", 240))
    if grid not in (120, 240):
        raise ValueError("grid must be 120 or 240")
    params = {}
    if q.get("onset"):
        params["onset"] = min(max(float(q["onset"]), 0.1), 0.95)
    music_lm = {"lam": 0.3, "beta": 1.0} if q.get("music_lm") == "1" else None
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", q.get("name", "") or "recording").strip("_")[:60] or "recording"
    return {"model": model, "capo": capo, "fingering": fingering, "tempo": tempo, "grid": grid, "params": params,
            "music_lm": music_lm, "name": name}


def history():
    runs = []
    for f in sorted(OUT.glob("*/result.json"), reverse=True)[:30]:
        try:
            runs.append(json.loads(f.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return runs


class Handler(BaseHTTPRequestHandler):
    device = "auto"

    def log_message(self, fmt, *args):  # keep the terminal quiet except for transcriptions
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self.send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/models":
            return self.send(200, available_models())
        if path == "/api/history":
            return self.send(200, history())
        if path.startswith("/files/"):
            f = (OUT / path[len("/files/"):]).resolve()
            if f.is_file() and f.is_relative_to(OUT.resolve()):
                ctype = "application/octet-stream" if f.suffix == ".gp5" else "text/plain; charset=utf-8"
                return self.send(200, f.read_bytes(), ctype)
        self.send(404, {"error": "not found"})

    def do_DELETE(self):
        path = urlparse(self.path).path
        run = (OUT / path[len("/api/tabs/"):]).resolve() if path.startswith("/api/tabs/") else None
        if not run or run.parent != OUT.resolve() or not (run / "result.json").is_file():
            return self.send(404, {"error": "not found"})
        shutil.rmtree(run)
        self.send(200, {"deleted": run.name})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/transcribe":
            return self.send(404, {"error": "not found"})
        try:
            opt = options(url.query)
            audio = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not audio:
                raise ValueError("no audio received")
        except ValueError as e:
            return self.send(400, {"error": str(e)})
        suffix = Path(self.headers.get("X-Filename", "audio.webm")).suffix.lower()[:6] or ".webm"
        run = OUT / f"{datetime.now():%Y%m%d_%H%M%S}_{opt['name']}"
        run.mkdir(parents=True, exist_ok=True)
        src, wav = run / f"input{suffix}", run / "audio.wav"
        src.write_bytes(audio)
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "22050", str(wav)],
                           check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            shutil.rmtree(run)
            return self.send(400, {"error": "could not read the audio: " + e.stderr.decode(errors="replace")[-300:]})
        src.unlink()  # audio.wav is enough to transcribe again
        from .pipeline import transcribe_to_tab  # torch and the models load on the first request

        t0 = time.time()
        print(f"[{datetime.now():%H:%M:%S}] {opt['name']}: {opt['model']}, capo {opt['capo']}, "
              f"{'music LM' if opt['music_lm'] else 'default decoding'}", flush=True)
        try:
            with LOCK:
                res = transcribe_to_tab(wav, run, opt["name"], MODELS[opt["model"]][0], self.device, opt["params"],
                                        opt["music_lm"], opt["capo"], opt["tempo"], opt["fingering"], opt["grid"],
                                        write_tokens=False, log=lambda m: print("   " + m, flush=True))
        except ValueError as e:
            shutil.rmtree(run)
            return self.send(422, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            shutil.rmtree(run)
            return self.send(500, {"error": f"{type(e).__name__}: {e}"})
        result = {"id": run.name, "name": opt["name"], "model": opt["model"], "capo": opt["capo"],
                  "music_lm": bool(opt["music_lm"]), "notes": res["notes"], "tempo": res["tempo"],
                  "beats_per_measure": res["beats_per_measure"], "seconds": round(time.time() - t0, 1),
                  "gp5": f"/files/{run.name}/{res['gp5'].name}",
                  "created": f"{datetime.now():%Y-%m-%d %H:%M}"}
        (run / "result.json").write_text(json.dumps(result, indent=1))
        self.send(200, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto", choices=DEVICES, help="auto = cuda, else mps, else cpu")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page in the browser")
    args = parser.parse_args()
    if not available_models():
        raise SystemExit("No model checkpoint found (see paths.MODELS): download the weights or train a model (README).")
    Handler.device = args.device
    OUT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"guitab app on {url}  (Ctrl+C to stop); results in {OUT}", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
