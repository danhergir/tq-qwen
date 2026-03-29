#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock
from typing import Any
from urllib.parse import urlparse

from tq_qwen_common import encode_ndjson
from tq_qwen_runtime import DEFAULT_MODEL, RuntimeConfig, TurboQuantRuntime

STATIC_DIR = Path(__file__).with_name("static")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a local browser chat UI for Qwen3.5 TurboQuant."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bits", type=int, choices=(3, 4), default=4)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


class ChatAppState:
    def __init__(self, runtime: TurboQuantRuntime):
        self.runtime = runtime
        self.messages: list[dict[str, str]] = []
        self.lock = Lock()
        self.stop_event: Event | None = None
        self.busy = False

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "busy": self.busy,
                "model": self.runtime.config.model,
                "model_source": self.runtime.model_source,
                "max_tokens": self.runtime.config.max_tokens,
                "temp": self.runtime.config.temp,
                "bits": self.runtime.config.bits,
                "offline": self.runtime.config.offline,
                "message_count": len(self.messages),
                "messages": [dict(message) for message in self.messages],
            }

    def request_stop(self) -> bool:
        with self.lock:
            if self.stop_event is None:
                return False
            self.stop_event.set()
            return True

    def reset(self) -> bool:
        with self.lock:
            if self.busy:
                return False
            self.messages = []
            return True


class QwenUIHandler(BaseHTTPRequestHandler):
    server_version = "QwenUI/0.1"

    @property
    def app_state(self) -> ChatAppState:
        return self.server.app_state  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_static("index.html")
            return
        if parsed.path == "/api/state":
            self._send_json(HTTPStatus.OK, self.app_state.snapshot())
            return
        if parsed.path.startswith("/assets/"):
            self._serve_static(parsed.path.removeprefix("/assets/"))
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            self._handle_chat()
            return
        if parsed.path == "/api/reset":
            self._handle_reset()
            return
        if parsed.path == "/api/stop":
            self._handle_stop()
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, relative_path: str) -> None:
        candidate = (STATIC_DIR / relative_path).resolve()
        if not str(candidate).startswith(str(STATIC_DIR.resolve())) or not candidate.exists():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        content_type, _ = mimetypes.guess_type(candidate.name)
        data = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", (content_type or "application/octet-stream") + "; charset=utf-8"
        )
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _handle_reset(self) -> None:
        if not self.app_state.reset():
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": "Cannot reset while a response is generating."},
            )
            return
        self._send_json(HTTPStatus.OK, {"ok": True})

    def _handle_stop(self) -> None:
        stopped = self.app_state.request_stop()
        self._send_json(HTTPStatus.OK, {"ok": True, "stopping": stopped})

    def _handle_chat(self) -> None:
        payload = self._read_json()
        user_message = (payload.get("message") or "").strip()
        if not user_message:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Message is required."})
            return

        with self.app_state.lock:
            if self.app_state.busy:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"error": "A response is already being generated."},
                )
                return
            self.app_state.busy = True
            stop_event = Event()
            self.app_state.stop_event = stop_event
            self.app_state.messages.append({"role": "user", "content": user_message})
            working_messages = [dict(message) for message in self.app_state.messages]

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        assistant_text = ""
        stopped = False
        try:
            latest_token_count = 0
            latest_tokens_per_second = 0.0
            for event in self.app_state.runtime.stream_chat_events(
                working_messages,
                stop_event=stop_event,
            ):
                delta = str(event["delta"])
                latest_token_count = int(event["token_count"])
                latest_tokens_per_second = float(event["tokens_per_second"])
                assistant_text += delta
                self.wfile.write(
                    encode_ndjson(
                        {
                            "type": "token",
                            "text": delta,
                            "token_count": latest_token_count,
                            "tokens_per_second": latest_tokens_per_second,
                        }
                    )
                )
                self.wfile.flush()

            stopped = stop_event.is_set()
            if assistant_text:
                with self.app_state.lock:
                    self.app_state.messages.append(
                        {"role": "assistant", "content": assistant_text}
                    )
            self.wfile.write(
                encode_ndjson(
                    {
                        "type": "done",
                        "text": assistant_text,
                        "stopped": stopped,
                        "token_count": latest_token_count,
                        "tokens_per_second": latest_tokens_per_second,
                    }
                )
            )
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            if assistant_text:
                with self.app_state.lock:
                    self.app_state.messages.append(
                        {"role": "assistant", "content": assistant_text}
                    )
        except Exception as exc:
            self.wfile.write(
                encode_ndjson(
                    {
                        "type": "error",
                        "error": str(exc),
                    }
                )
            )
            self.wfile.flush()
        finally:
            with self.app_state.lock:
                self.app_state.busy = False
                self.app_state.stop_event = None


def main() -> int:
    args = parse_args()
    runtime = TurboQuantRuntime(
        RuntimeConfig(
            model=args.model,
            max_tokens=args.max_tokens,
            temp=args.temp,
            seed=args.seed,
            bits=args.bits,
            offline=args.offline,
            verbose=args.verbose,
        )
    )
    server = ThreadingHTTPServer((args.host, args.port), QwenUIHandler)
    server.app_state = ChatAppState(runtime)  # type: ignore[attr-defined]
    address = f"http://{args.host}:{args.port}"
    print(f"Qwen UI ready at {address}")
    print(f"Model: {runtime.config.model}")
    print(f"Source: {runtime.model_source}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
