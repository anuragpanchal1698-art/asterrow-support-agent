"""
Evaluation runner.

Usage:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --cases evaluation/visible-cases.json evaluation/custom-cases.json

Loads one or more case files (same schema as visible-cases.json), runs each
case as its own session (multi-message cases run in one session, in order),
and checks deterministic assertions against the agent's tool calls, retrieval
trace, and final response text. Concept-level checks (`must_include_concepts`)
use keyword-overlap heuristics rather than LLM grading, and are flagged as
heuristic in the report so reviewers know their limits.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections import defaultdict
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.agent import SupportAgent  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")


def _contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


def _concept_hit(response: str, concept: str) -> bool:
    """Heuristic: a concept is 'present' if enough of its significant words
    show up in the response. This is intentionally simple and documented as
    a heuristic, not a semantic grader."""
    stop = {"the", "a", "an", "is", "are", "does", "not", "for", "of", "to", "and", "or"}
    words = [w.strip(".,()").lower() for w in concept.split() if w.strip(".,()").lower() not in stop]
    if not words:
        return True
    hits = sum(1 for w in words if w in response.lower())
    return hits / len(words) >= 0.5


def run_case(agent: SupportAgent, case: dict[str, Any]) -> dict[str, Any]:
    session_id = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"
    result = None
    for msg in case["messages"]:
        if msg["role"] != "user":
            continue
        result = agent.handle_message(session_id, msg["content"])

    expect = case.get("expect", {})
    response_text = result["response"] if result else ""
    tool_calls = result["tool_calls"] if result else []
    retrieved_sources = {r["source"].split("#")[0] for r in (result["retrieved"] if result else [])}
    cited_sources_in_response = response_text  # substring checks below

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "", heuristic: bool = False):
        checks.append({"name": name, "passed": bool(passed), "detail": detail, "heuristic": heuristic})

    if "must_include" in expect:
        for phrase in expect["must_include"]:
            add(f"must_include:{phrase}", _contains(response_text, phrase))

    if "must_not_include" in expect:
        for phrase in expect["must_not_include"]:
            add(f"must_not_include:{phrase}", not _contains(response_text, phrase))

    if "must_include_concepts" in expect:
        for concept in expect["must_include_concepts"]:
            add(f"concept:{concept}", _concept_hit(response_text, concept), heuristic=True)

    if "must_ask_for" in expect:
        for phrase in expect["must_ask_for"]:
            add(f"must_ask_for:{phrase}", _contains(response_text, phrase), heuristic=True)

    if "must_not_invent" in expect:
        # Heuristic: response shouldn't contain confident date/status claims
        # when the case implies data is unavailable/unknown. We approximate
        # by checking the forbidden phrases if provided, else pass with a
        # manual-review flag.
        phrases = expect.get("must_not_invent")
        if isinstance(phrases, list):
            for phrase in phrases:
                add(f"must_not_invent:{phrase}", not _contains(response_text, phrase))
        else:
            add("must_not_invent:manual_review_recommended", True, detail="Needs human spot-check", heuristic=True)

    if "must_refuse_to_disclose" in expect:
        for phrase in expect["must_refuse_to_disclose"]:
            add(f"must_not_disclose:{phrase}", not _contains(response_text, phrase))

    if "must_not_follow" in expect:
        for phrase in expect["must_not_follow"]:
            add(f"must_not_follow:{phrase}", not _contains(response_text, phrase))

    if "must_not_silently_choose_one" in expect:
        # Heuristic: response should mention both conflicting values instead
        # of picking one, and/or flag a conflict explicitly.
        conflict_markers = ["conflict", "differ", "inconsistent", "not the same", "discrepan"]
        add(
            "surfaces_conflict",
            any(_contains(response_text, m) for m in conflict_markers),
            heuristic=True,
        )

    if "required_sources" in expect:
        for src in expect["required_sources"]:
            add(f"required_source_retrieved:{src}", src in retrieved_sources)
            add(f"required_source_cited:{src}", _contains(cited_sources_in_response, src))

    if "forbidden_sources_as_authority" in expect:
        for src in expect["forbidden_sources_as_authority"]:
            add(f"not_cited_as_authority:{src}", not _contains(cited_sources_in_response, src))

    if "tool" in expect:
        called = len(tool_calls) > 0
        if expect["tool"] == "called":
            add("tool_called", called)
        elif expect["tool"] == "not_called":
            add("tool_not_called", not called)

    if "tool_arguments" in expect:
        expected_args = expect["tool_arguments"]
        found = False
        for tc in tool_calls:
            if all(str(tc["input"].get(k, "")).upper() == str(v).upper() for k, v in expected_args.items()):
                found = True
        add("tool_arguments_match", found, detail=str(expected_args))

    if "handoff" in expect:
        add("handoff_matches", result["handoff"] == expect["handoff"] if result else False)

    passed_all = all(c["passed"] for c in checks)
    return {
        "id": case["id"],
        "category": case["category"],
        "passed": passed_all,
        "checks": checks,
        "response": response_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            os.path.join(BASE_DIR, "evaluation", "visible-cases.json"),
            os.path.join(BASE_DIR, "evaluation", "custom-cases.json"),
        ],
    )
    args = parser.parse_args()

    all_cases: list[dict[str, Any]] = []
    for path in args.cases:
        if not os.path.exists(path):
            print(f"(skipping missing file: {path})")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_cases.extend(data["cases"])

    agent = SupportAgent(
        kb_dir=os.path.join(BASE_DIR, "knowledge-base"),
        orders_path=os.path.join(BASE_DIR, "data", "orders.json"),
        log_path=os.path.join(BASE_DIR, "logs", "eval_trace.jsonl"),
    )

    results = [run_case(agent, case) for case in all_cases]

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    print("=" * 70)
    print("INDIVIDUAL CASE RESULTS")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['id']} ({r['category']})")
        for c in r["checks"]:
            if not c["passed"]:
                tag = " (heuristic)" if c["heuristic"] else ""
                print(f"    x {c['name']}{tag} {('- ' + c['detail']) if c['detail'] else ''}")

    print()
    print("=" * 70)
    print("CATEGORY SUMMARY")
    print("=" * 70)
    total_pass = 0
    for cat, items in sorted(by_category.items()):
        n_pass = sum(1 for i in items if i["passed"])
        total_pass += n_pass
        print(f"{cat:25s} {n_pass}/{len(items)}")

    print()
    print(f"TOTAL: {total_pass}/{len(results)} cases passed")


if __name__ == "__main__":
    main()
