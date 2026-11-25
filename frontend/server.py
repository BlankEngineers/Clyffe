"""
Offline Local Web Server for ARC-LB-01 Frontend.
Uses Python stdlib http.server + ARCLB01Runtime (no external AI APIs).
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import time

from src.inference.runtime import ARCLB01Runtime


class ServerState:
    def __init__(self, checkpoint_dir: str, active_ckpt: str, tokenizer_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer_dir = tokenizer_dir
        self.active_ckpt = active_ckpt
        self.runtime = ARCLB01Runtime(
            os.path.join(checkpoint_dir, active_ckpt),
            tokenizer_dir=self._tok_for_ckpt(active_ckpt),
            device="cpu",
        )

    def available_checkpoints(self):
        return sorted(
            f for f in os.listdir(self.checkpoint_dir) if f.endswith("_best.pt")
        )

    def _tok_for_ckpt(self, ckpt_name: str) -> str:
        if "clyffe-v0.1.1" in ckpt_name and os.path.exists("checkpoints/clyffe_v0.1.1_tokenizer"):
            return "checkpoints/clyffe_v0.1.1_tokenizer"
        if "clyffe-v0.1" in ckpt_name and os.path.exists("checkpoints/clyffe_v0.1_tokenizer"):
            return "checkpoints/clyffe_v0.1_tokenizer"
        if "V3" in ckpt_name and os.path.exists("checkpoints/tokenizer_v3"):
            return "checkpoints/tokenizer_v3"
        if "V2" in ckpt_name and os.path.exists("checkpoints/tokenizer_v2"):
            return "checkpoints/tokenizer_v2"
        return self.tokenizer_dir

    def switch_checkpoint(self, ckpt_name: str):
        path = os.path.join(self.checkpoint_dir, ckpt_name)
        if not os.path.exists(path):
            raise FileNotFoundError(ckpt_name)
        self.active_ckpt = ckpt_name
        self.runtime = ARCLB01Runtime(path, tokenizer_dir=self._tok_for_ckpt(ckpt_name), device="cpu")


def make_handler(state: ServerState, html_path: str):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj: dict, status: int = 200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(html_path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/status":
                self._send_json(
                    {
                        "active_checkpoint": state.active_ckpt,
                        "available_checkpoints": state.available_checkpoints(),
                        "metadata": state.runtime.metadata,
                    }
                )
            else:
                self._send_json({"error": "Not found"}, status=404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))

            if self.path == "/api/chat":
                prompt = payload.get("prompt", "")
                t0 = time.perf_counter()
                resp = state.runtime.respond(prompt)
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                self._send_json(
                    {
                        "response": resp,
                        "latency_ms": latency_ms,
                        "checkpoint": state.active_ckpt,
                    }
                )
            elif self.path == "/api/switch":
                ckpt = payload.get("checkpoint", "")
                state.switch_checkpoint(ckpt)
                self._send_json(
                    {
                        "active_checkpoint": state.active_ckpt,
                        "metadata": state.runtime.metadata,
                    }
                )
            else:
                self._send_json({"error": "Not found"}, status=404)

        def log_message(self, fmt, *args):
            # Keep stdout clean
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description="ARC-LB-01 Offline Web Frontend")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", type=str, default="clyffe-v0.1.1-C-smoothsign-20261125_best.pt")
    args = parser.parse_args()

    state = ServerState("checkpoints", args.checkpoint, "checkpoints/tokenizer")
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    server = HTTPServer((args.host, args.port), make_handler(state, html_path))
    print(f"ARC-LB-01 Frontend running at http://{args.host}:{args.port} (checkpoint={args.checkpoint})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

