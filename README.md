# Aster & Row Support Agent

A RAG + tool-use support agent for the Aster & Row take-home assignment.

## 1. Setup

```bash
git clone <your-fork-url>
cd asterrow-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

Run the CLI:

```bash
python cli.py
```

Run the evaluation suite:

```bash
python evaluation/run_eval.py
```

## 2. Environment variables

See `.env.example`.

| Variable | Required | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | Your own Anthropic API key. Never committed. |
| `ANTHROPIC_MODEL` | no | Defaults to `claude-sonnet-4-6`. |

## 3. Model / embedding / framework / storage choices

- **LLM**: Claude (Anthropic Messages API), via native tool-calling for the order lookup.
- **Retrieval**: TF-IDF + cosine similarity (scikit-learn) over front-matter-aware
  markdown chunks, no external embedding API required. Chosen for zero cost,
  full determinism, and because the corpus is small (14 docs / ~50 chunks) --
  swapping in a real embedding model is a one-file change (see the
  `EmbeddingRetriever` stub in `app/retriever.py`).
- **Framework**: no agent framework -- a plain tool-calling loop against the
  Anthropic SDK, so every retrieval/tool/prompt decision is visible and
  testable.
- **Storage**: none beyond the filesystem. No vector DB; the TF-IDF index is
  rebuilt in-memory on startup (fast at this corpus size). Sessions are
  in-memory, keyed by `session_id`.

## 4. Architecture

```
knowledge-base/*.md --> app/ingest.py (front-matter + heading chunking)
                              |
                              v
                       app/retriever.py (TF-IDF, precedence boosting)
                              |
data/orders.json --> app/order_tool.py (lookup, redaction, normalization)
                              |
                              v
                        app/agent.py
              (system prompt + tool-calling loop + session state)
                              |
                 -------------+-------------
                 |                         |
              cli.py                evaluation/run_eval.py
                 |                         |
        app/logging_utils.py (structured JSONL trace log for both)
```

Each user turn: retrieve top-k chunks (tagged authoritative / non-authoritative
based on `status` + `policy_authority` + `audience`) -> inject as `<retrieved_context>`
data (never as instructions) -> model responds or calls `lookup_order` -> tool
result fed back as data -> final answer. Every turn is logged to
`logs/trace.jsonl` (or `logs/eval_trace.jsonl` during evaluation).

### Document precedence

Front matter fields (`status`, `policy_authority`, `audience`,
`customer_answering`) are parsed per chunk. Chunks marked `status: active` +
`policy_authority: official` + `audience: customer` are boosted and labeled
`AUTHORITATIVE`; superseded, draft, or internal-only chunks are down-weighted
and labeled `NON-AUTHORITATIVE/CONTEXT-ONLY` in the prompt, with an explicit
instruction never to cite them as current policy. When two *authoritative*
sources disagree, both are retrieved and the system prompt requires the agent
to surface the conflict rather than pick one.

### Privacy / tool safety

`app/order_tool.py` is the only path order data can take into the model. It
allowlists customer-safe fields (never `customer.*` or `internal.*`, even if
asked), normalizes ID casing/whitespace, rejects malformed IDs without
guessing, and blanks stale shipping fields once an order is `cancelled` or
`returned`.

## 5. Running evaluations

```bash
python evaluation/run_eval.py
```

Runs every case in `evaluation/visible-cases.json` plus
`evaluation/custom-cases.json` (7 original cases covering malformed order IDs,
cross-turn order follow-ups, gift-card/final-sale conflicts, missing order ID,
direct prompt-injection, session isolation, and a no-lifetime-warranty
groundedness check). Reports pass/fail per case and a per-category summary.
Assertions on exact strings (`must_include`, `required_sources`, tool
call/arguments, `handoff`) are deterministic; `must_include_concepts` checks
use a keyword-overlap heuristic and are flagged `(heuristic)` in output --
they are not LLM-graded, but they are not exact-match either, so treat
heuristic misses as a prompt to read the actual response, not an automatic
failure.

## 6. Baseline vs. final evaluation results

<!-- TODO after running: paste baseline category breakdown here -->
<!-- TODO after running: paste final category breakdown here -->

| Category | Baseline | Final |
|---|---|---|
| retrieval | | |
| groundedness | | |
| multi-source-grounding | | |
| source-conflict | | |
| tool-use | | |
| tool-reliability | | |
| privacy | | |
| prompt-security | | |
| conversation | | |
| abstention | | |

## 7. Bug diary

<!-- TODO: fill in as you find real failures while testing. Template below. -->

### Bug 1: <title>
- **Reproduction**: <exact input / session that triggered it>
- **Root cause**: <why it happened>
- **Fix**: <what changed>
- **Regression test**: <case id in custom-cases.json>

### Bug 2: <title>
...

### Bug 3: <title>
...

## 8. Known limitations / what I'd improve before production

- TF-IDF retrieval is lexical, not semantic -- paraphrased questions that
  share no vocabulary with the source docs may retrieve poorly. A real
  embedding model would help most here.
- `must_include_concepts` grading in the eval suite is a keyword heuristic,
  not a semantic check -- it can pass/fail incorrectly on edge phrasing.
- Sessions are in-memory only; a restart loses all conversation state.
- No real authentication -- order ID possession is treated as sufficient,
  per the assignment's assumption, which would not be acceptable in
  production.
- No persistent vector store -- fine at this corpus size, would not scale as-is.

## 9. AI coding tools used

<!-- TODO: fill in truthfully, e.g. "Used Claude to scaffold app/retriever.py
and evaluation/run_eval.py; caught and fixed an incorrect assumption that
[X]." Include one concrete example of an AI suggestion that was wrong or
incomplete. -->

## 10. Demo

<!-- TODO: embed a 2-4 min GIF/video here showing:
     1. A knowledge-base question with citations
     2. An order lookup
     3. A multi-turn conversation
     4. A refusal/handoff case
     5. The evaluation suite running -->
