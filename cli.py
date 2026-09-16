"""Minimal CLI for the Aster & Row support agent.

Usage:
    python cli.py
"""
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from app.agent import SupportAgent

BASE_DIR = os.path.dirname(__file__)


def main():
    agent = SupportAgent(
        kb_dir=os.path.join(BASE_DIR, "knowledge-base"),
        orders_path=os.path.join(BASE_DIR, "data", "orders.json"),
        log_path=os.path.join(BASE_DIR, "logs", "trace.jsonl"),
    )
    session_id = str(uuid.uuid4())
    print("Aster & Row Support Agent (type 'exit' to quit)")
    print(f"[session: {session_id}]\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        result = agent.handle_message(session_id, user_input)
        print(f"\nAgent: {result['response']}")
        if result["tool_calls"]:
            for tc in result["tool_calls"]:
                print(f"  [tool: {tc['name']}({tc['input']})]")
        if result["handoff"]:
            print("  [-> recommending human handoff]")
        print()


if __name__ == "__main__":
    main()
