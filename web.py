"""Minimal web interface for the Aster & Row support agent.

Usage:
    python web.py
"""
from __future__ import annotations

import json
import os
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from dotenv import load_dotenv

from app.agent import SupportAgent


BASE_DIR = os.path.dirname(__file__)
STATIC_INDEX = os.path.join(BASE_DIR, "static", "index.html")


def build_agent() -> SupportAgent:
    return SupportAgent(
        kb_dir=os.path.join(BASE_DIR, "knowledge-base"),
        orders_path=os.path.join(BASE_DIR, "data", "orders.json"),
        log_path=os.path.join(BASE_DIR, "logs", "trace.jsonl"),
    )


class SupportRequestHandler(BaseHTTPRequestHandler):
    agent: SupportAgent

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_index(self) -> None:
        with open(STATIC_INDEX, "rb") as file:
            body = file.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_index()
            return
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"ok": True})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
        except (ValueError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return

        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, str) or not message.strip():
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "message must be a non-empty string"},
            )
            return

        session_id = payload.get("session_id") if isinstance(payload, dict) else None
        if not isinstance(session_id, str) or not session_id.strip():
            session_id = str(uuid.uuid4())

        try:
            result = self.agent.handle_message(session_id, message.strip())
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "agent_error", "detail": str(exc)},
            )
            return

        response = {
            "session_id": session_id,
            "response": result["response"],
            "handoff": result["handoff"],
            "tool_called": bool(result["tool_calls"]),
            "sources": [item["source"] for item in result["retrieved"]],
        }
        self._send_json(HTTPStatus.OK, response)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    load_dotenv()
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "8000"))
    SupportRequestHandler.agent = build_agent()
    server = ThreadingHTTPServer((host, port), SupportRequestHandler)
    print(f"Aster & Row web interface: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
