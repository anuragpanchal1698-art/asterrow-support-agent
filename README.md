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

Or run the web interface:

```bash
python web.py
```
Then open http://127.0.0.1:8000 in a browser. (Set `WEB_PORT` / `WEB_HOST` in `.env` to change the bind address.)

Run the evaluation suite:

```bash
python evaluation/run_eval.py
```

## 2. Environment variables

See `.env.example`.

| Variable | Required | Notes |
|---|---|---|
| `LLM_PROVIDER` | no | `anthropic` (default) or `gemini`. |
| `ANTHROPIC_API_KEY` | if using Anthropic | Never committed. |
| `ANTHROPIC_MODEL` | no | Defaults to `claude-sonnet-4-6`. |
| `GEMINI_API_KEY` | if using Gemini | Free tier key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey). |
| `GEMINI_MODEL` | no | Defaults to `gemini-2.0-flash` (see note below -- verify current model names before relying on any default). |

To run on Gemini's free tier instead of Anthropic, set in `.env`:
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
```
`app/providers.py` abstracts the two SDKs behind one interface so the rest of
the agent code (retrieval, tool logic, system prompt) doesn't change between
providers.

## 3. Model / embedding / framework / storage choices

- **LLM**: Claude (Anthropic Messages API), via native tool-calling for the order lookup.
- **Retrieval**: TF-IDF + cosine similarity (scikit-learn) over front-matter-aware
  markdown chunks, no external embedding API required. Chosen for zero cost,
  full determinism, and because the corpus is small (14 docs / ~50 chunks) --
  swapping in a real embedding model is a one-file change (see the
  `EmbeddingRetriever` stub in `app/retriever.py`).
- **Framework**: no agent framework -- a plain tool-calling loop against the
  Anthropic SDK, so every retrieval/tool/prompt decision is visible and
  testable. `app/providers.py` abstracts Anthropic vs. Gemini behind one
  interface, so the agent logic itself is provider-agnostic.
- **Storage**: none beyond the filesystem. No vector DB; the TF-IDF index is
  rebuilt in-memory on startup (fast at this corpus size). Sessions are
  in-memory, keyed by `session_id`.
- **Interface**: a minimal web server (`web.py`) built on Python's built-in
  `http.server` (no extra framework dependency) with a single static HTML/JS
  chat page (`static/index.html`) showing the answer, cited sources, and
  order-lookup/handoff badges -- plus a plain CLI (`cli.py`) for quick
  terminal testing. Both call the same `SupportAgent`.

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

| Category | Baseline (rate-limit noise) | After quota/model fix | After eval-harness + prompt fixes |
|---|---|---|---|
| retrieval | 2/3 | 2/3 | 2/3 |
| groundedness | 0/2 | 1/2 | -- (rerun to confirm) |
| multi-source-grounding | 0/2 | 1/2 | -- |
| source-conflict | 0/1 | 0/1 | -- |
| tool-use | 0/3 | 2/3 | -- |
| tool-reliability | 0/4 | 2/4 | -- |
| privacy | 1/1 | 0/1 (false positive, see Bug 4) | 1/1 (expected) |
| prompt-security | 0/2 | 1/2 | -- |
| conversation | 0/3 | 0/3 | -- |
| abstention | 0/1 | 0/1 | -- |
| **TOTAL** | **3/22** | **9/22** | **-- (rerun after latest fixes)** |

<!-- TODO: rerun `python evaluation/run_eval.py` after the Bug 3 (tool
misuse) and Bug 4 (eval harness false positives) fixes, and fill in the
final column + total. -->

## 7. Bug diary

<!-- TODO: fill in as you find real failures while testing. Template below. -->

### Bug 1: Eval suite exhausted Gemini's per-model daily quota, masking every real result behind a generic fallback
- **Reproduction**: `python evaluation/run_eval.py` on `LLM_PROVIDER=gemini`
  with a fresh free-tier API key, using `gemini-3.6-flash`. First diagnosed
  as per-minute rate limiting (`RESOURCE_EXHAUSTED`, `GenerateRequestsPerMinute...`,
  `limit: 5`) and fixed with request pacing + retry/backoff -- but after
  adding pacing, the *same* generic-fallback failure pattern reappeared.
  Digging into the actual error revealed a second, harder limit:
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `limit: 20`. This
  specific model's free-tier daily allowance (20 requests/day) is far
  smaller than other Flash-family models -- confirmed by checking current
  Gemini pricing/rate-limit docs, which show daily allowances varying from
  ~20/day (`gemini-3.6-flash` and siblings) up to 500+/day
  (`gemini-3.5-flash-lite`) depending on model, despite similar per-minute
  limits. No amount of pacing or retrying fixes a daily cap once it's spent
  -- it only resets on Google's own schedule (midnight Pacific).
- **Root cause**: two distinct, stacked rate limits (per-minute AND
  per-day), each requiring a different mitigation, and a default model
  choice (`gemini-3.6-flash`) with an unusually small daily allowance for
  this kind of iterative testing. `app/agent.py`'s existing
  `except Exception` handler correctly avoided a crash in both cases but
  swallowed the real error into a generic customer-facing message, so both
  failure modes initially looked like ~20 unrelated agent-behavior bugs
  until the raw error was inspected directly.
- **Fix**: three changes:
  1. `app/providers.py` -- `call_with_retry()`, used by both providers'
     `step()`, detects rate-limit errors specifically (never retries a real
     bug), parses a server-suggested `retryDelay` when present, and backs
     off exponentially otherwise. This alone resolves per-minute limits but
     cannot resolve a daily cap.
  2. `evaluation/run_eval.py` -- `_pace_request()` enforces a minimum
     interval between calls (tunable via `EVAL_REQUEST_INTERVAL_SECONDS`)
     so a full suite run doesn't immediately trip per-minute limits. The
     report also now prints the raw error inline next to any case whose
     response is a fallback, so this class of failure is diagnosable from
     the eval output directly.
  3. Switched the default Gemini model to `gemini-3.5-flash-lite`, which
     has a daily quota large enough to actually run this eval suite
     repeatedly during development.
- **Regression test**: no dedicated case (infra/quota issue, not agent
  behavior) -- verified by re-running the full suite on the new model and
  confirming zero generic-fallback responses in the output.

### Bug 2: Turn-logging crashed on Gemini's message format
- **Reproduction**: any message sent while `LLM_PROVIDER=gemini`, once a
  response actually succeeded. Crashed with
  `TypeError: 'Content' object is not subscriptable`.
- **Root cause**: `app/agent.py`'s logging call assumed every message in
  session history was an Anthropic-style dict (`message["role"]`). Gemini's
  SDK returns `google.genai.types.Content` objects, which aren't
  subscriptable.
- **Fix**: added `SupportAgent._message_role()`, a small helper that checks
  `isinstance(message, dict)` and falls back to `getattr(message, "role")`
  for object-shaped messages. Logging now works identically for both
  providers.
- **Regression test**: manually verified against both a dict and a mock
  object with a `.role` attribute (see conversation history / commit for
  the exact check); no automated case exists for this since it's
  provider-plumbing rather than agent behavior an eval case would target.

### Bug 3: Agent called `lookup_order` speculatively on a pure shipping-policy question
- **Reproduction**: `evaluation/run_eval.py`, case `unsupported-country`
  ("Can you ship an Atlas Weekender to Germany?" -- a general shipping
  question naming no order). One run showed `tool_calls: [{'name':
  'lookup_order', 'input': {}, ...}]` -- the tool was called with an empty
  `order_id`, on a question that isn't about any specific order at all.
- **Root cause**: the system prompt told the model it "MUST call
  `lookup_order`" for order questions, but didn't explicitly forbid calling
  it when there's no order context whatsoever -- so on an ambiguous
  "can you ship X" question, the model sometimes reached for the only tool
  available rather than reasoning that this is a policy question with no
  order ID to look up.
- **Fix**: tightened the "Order questions" section of the system prompt in
  `app/agent.py` to explicitly say the tool should be called ONLY when the
  user references a specific order, and never speculatively or with a
  missing `order_id`.
- **Regression test**: `evaluation/custom-cases.json` -- the existing
  `custom-no-order-id-provided` and `unsupported-country` (from
  visible-cases.json) cases both assert `tool: "not_called"` and now serve
  as the regression check for this.

### Bug 4: Eval harness itself had false-positive assertions on correct refusals
- **Reproduction**: two passing agent behaviors were scored as FAIL by the
  eval suite itself:
  1. `custom-warranty-question` -- the agent correctly said *"the Breeze
     Tumbler does **not** come with a lifetime warranty"*, but
     `must_not_include: ["lifetime warranty"]` did a naive substring match
     and flagged the phrase's mere presence, ignoring the negation right
     next to it.
  2. `order-data-privacy` -- the agent correctly refused with *"I cannot
     provide customer email addresses, shipping addresses, internal notes,
     or risk scores"*, but `must_refuse_to_disclose` was checking for the
     literal words "email"/"address"/etc. rather than actual leaked data,
     so a textbook-correct refusal that *names the categories it's
     refusing to share* failed the check.
- **Root cause**: both assertions tested for the presence of a phrase
  without any awareness of negation or refusal context -- adequate for
  catching a real leak, but unable to distinguish "here is your email" from
  "I won't share your email."
- **Fix**: in `evaluation/run_eval.py`:
  1. `_phrase_appears_unnegated()` checks the sentence containing the
     phrase for a nearby negation cue (not/no/n't/cannot/without/etc.)
     before flagging it -- so a correct denial no longer trips the check.
  2. `_check_no_pii_leak()` replaces category-word matching with a check
     against the *actual* PII values on the referenced order (looked up
     from `data/orders.json` by extracting the order ID from the case's
     messages) -- so it only fails when a real email/address/note/score
     value literally appears in the response, not when the response
     correctly discusses the category.
  Both were verified against four cases each (a correct denial, a correct
  refusal, an actual leak, and a genuinely false claim) to confirm the new
  logic passes the first two and still catches the latter two.
- **Regression test**: no dedicated case file entry (this is a harness
  fix, not an agent-behavior target) -- verified via a standalone script
  exercising `_phrase_appears_unnegated()` and `_check_no_pii_leak()`
  directly against both correct and incorrect agent responses.

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
- Gemini's free tier has model-specific daily quotas that vary enormously --
  `gemini-3.6-flash` allows only 20 requests/day on this project's free
  tier, which was exhausted mid-testing (see Bug 1). `gemini-3.5-flash-lite`
  has a much larger daily allowance and is used as the default here for
  that reason. A paid tier or Anthropic would remove this constraint
  entirely.

## 9. AI coding tools used

Claude (Anthropic) was used throughout this project to:
- Scaffold the initial architecture: `app/ingest.py`, `app/retriever.py`,
  `app/order_tool.py`, `app/agent.py`, and `evaluation/run_eval.py`.
- Build the dual-provider abstraction (`app/providers.py`) so the agent can
  run on either Anthropic or Gemini's free tier.
- Debug issues found while wiring up a live provider.

**Example of an AI-generated suggestion that was wrong or incomplete:**
When adding Gemini support, the first `app/providers.py` and `app/agent.py`
pairing worked for generating responses, but the turn-logging code
(`self.logger.log_turn(...)`) still assumed Anthropic's dict-shaped message
history (`message["role"]`). Gemini's SDK returns `types.Content` objects
instead of dicts, so the first real run crashed with
`TypeError: 'Content' object is not subscriptable` the moment a message
actually succeeded. The fix was a small `_message_role()` helper that
branches on whether the message is a dict or an object. This is a good
example of an AI suggestion that looked complete (it ran, imports worked,
the "happy path" code compiled) but hadn't actually been exercised against
a second provider's real object shapes -- a reminder that a provider
abstraction needs testing against *both* providers before it's trustworthy,
not just import-checked.

A second, smaller instance: the initial suggested Gemini model name
(`gemini-2.0-flash`) was already retired by the time it was tested, and the
error surfaced a corrected name (`gemini-3.6-flash`) that had to be looked
up rather than assumed -- a reminder that specific model identifiers from
an AI assistant should always be verified against current provider docs
rather than trusted as-is.

## 10. Demo

https://github.com/user-attachments/assets/ecd951f1-8912-4428-95ee-ddc5932d66aa

The video shows:
1. A knowledge-base question with citations
2. An order lookup
3. A multi-turn conversation
4. A case where the agent refuses to guess / recommends human help
5. The evaluation suite running
