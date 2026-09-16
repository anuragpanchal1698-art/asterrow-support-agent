"""Structured, append-only JSON-lines logging for observability.

Every turn logs: user message, trimmed history, retrieved chunks + scores,
tool calls + sanitized results, final response, and any handoff/error flag.
Never logs secrets (API keys are never placed in any logged object).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any


class TraceLogger:
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path

    def log_turn(
        self,
        session_id: str,
        user_message: str,
        history_used: list[dict[str, Any]],
        retrieved: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        final_response: str,
        handoff: bool,
        error: str | None = None,
    ) -> None:
        record = {
            "ts": time.time(),
            "session_id": session_id,
            "user_message": user_message,
            "history_used": history_used,
            "retrieved": retrieved,
            "tool_calls": tool_calls,
            "final_response": final_response,
            "handoff": handoff,
            "error": error,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
