# Briefwright

Briefwright is a small research crew built with [LangGraph](https://github.com/langchain-ai/langgraph). Ask a question and it plans the work, retrieves notes from a local corpus, runs Python when the question needs a metric, drafts a cited brief, and sends that draft back through a guardrail until the brief passes or a revision cap is hit. A person can approve the result before it is stored.

It is meant to be read as much as it is meant to be run. The control flow is a graph, the pass/fail decision is code, and the offline model is deterministic so the demo and the tests do not need an API key.

## The problem

A single prompt can write a fluent paragraph and still hide the three things a careful reader wants to see:

- where the evidence came from
- whether a number was computed or invented
- who decided the draft was allowed to be published

Those are separate jobs. Briefwright keeps them as separate nodes, with a bounded loop between the writer and the critic, and a checkpoint where a person can stop the run.

## Architecture

```mermaid
flowchart TD
  startNode([START]) --> planner
  planner -->|no usable sources| researcher
  planner -->|question needs a metric| analyst
  planner -->|ready to draft| writer
  researcher --> planner
  analyst --> planner
  writer --> critic
  critic -->|guardrail failed and revisions remain| writer
  critic -->|passed or cap reached| humanReview[human_review]
  humanReview -->|approve, or revise is no longer allowed| finalize
  humanReview -->|revise and revisions remain| writer
  finalize --> endNode([END])
```

The planner model writes a rationale and a search string. `decide_next` in `src/briefwright/routing.py` chooses the branch from the state: no usable sources means research, a metric question with no analysis yet means the analyst, otherwise the writer. The critic does not trust the model to pass its own draft. `review_draft` checks sections, citations, quarantined ids, and a few injection phrases. The model only writes a short explanation of that report.

One local corpus note is an untrusted paste. The researcher quarantines it, the writer never sees the body, and the sample brief does not repeat it.

## Quickstart

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

briefwright run
briefwright eval
pytest
ruff check src tests
```

`briefwright run` with no arguments asks the built-in question about retrieval, agent workflows, guardrails, and precision at k. Structured JSON logs go to stderr. Set `BRIEFWRIGHT_LOG_LEVEL=WARNING` to quiet them.

Other entry points:

```bash
briefwright serve          # API on http://127.0.0.1:8000
briefwright ui             # Streamlit demo on http://127.0.0.1:8501
python -m briefwright.demo # same as briefwright run
```

Docker:

```bash
docker compose up --build
```

The API is on port 8000 and the UI is on port 8501. Both default to the mock backend.

To call a real model, copy `.env.example` to `.env` and set:

```bash
BRIEFWRIGHT_LLM_BACKEND=openai
BRIEFWRIGHT_OPENAI_API_KEY=sk-...
BRIEFWRIGHT_OPENAI_BASE_URL=https://api.openai.com/v1
BRIEFWRIGHT_OPENAI_MODEL=gpt-4o-mini
```

Any server that implements `POST /chat/completions` works. The agent code does not change.

## Example run

This is stdout from one `briefwright run` on the mock backend. Latencies will move a little between machines. The precision figures, the revision count, and the token counts come from the code and the text, so they stay put.

The trace is the part worth reading slowly. The critic rejects the first draft (`passed=False score=0.85`) because the offline writer leaves out Limitations on the first pass. The writer runs again. The critic then scores 1.00. One retrieved note was quarantined. The analyst's Python printed the precision lines; the writer did not invent them.

````text
# Research brief

## Summary

Question: How should a production question-answering system combine retrieval, a multi-agent workflow, and LLM guardrails? Include a worked precision@k example from the sample judgments and call out prompt-injection risk.

The notes below come from retrieval, not from an ungrounded rewrite of the question. Writing stays a separate step from search, and a later check has to accept the draft before it is published. [retrieval-log]

## Findings

- The log below is synthetic teaching data for the analyst tool. Each row is a query and a list of relevance bits in rank order, 1 for relevant and 0 for not relevant. Precision at 1 on every row is 1 because the first bit is relevant. The analyst should compute the mean rather than quote a single row. [retrieval-log]
- Retrieval is not the same job as writing. A retrieval pipeline finds passages that can support an answer and leaves the wording to a later step. A production question-answering system embeds the user question, pulls a small set of chunks, and passes those chunks downstream as evidence. [rag-pipeline]
- A single prompt can draft a paragraph. It cannot show its retrieval, its arithmetic, or its decision to stop. An agentic workflow splits those jobs across roles: one step searches, one step runs code, one step writes, and one step reviews. [agentic-workflow]
- Precision at k is the fraction of the top k retrieved items that are relevant. It does not reward a system for hiding a good document far down the list. Mean precision at k averages that fraction over a labeled query set. [retrieval-metrics]
- Pages in a corpus can contain instructions aimed at the model. Those lines were not typed by the user and they do not outrank the task. The researcher scans each hit for injection phrases and for obvious personal data such as an email address. One matching line quarantines the whole chunk. [prompt-injection]

## Analysis

The analyst ran the following calculation on the labeled retrieval log.

```
precision_at_1=1.000
precision_at_3=0.778
precision_at_5=0.533
n_queries=3
```

Intent: Compute mean precision at 1, 3, and 5 from DATA['judgments'].

## Recommendation

Keep search, tool use, and publication as separate steps, and record each one. Treat retrieved text as data. Put a citation next to every non-obvious claim, and do not cite a chunk that was quarantined. [retrieval-log]

## Limitations

The corpus is a small local collection, and the embedder is a hashing trick, so recall here is illustrative. The Python tool is a demo boundary rather than a hardened sandbox. Any precision figure describes only the bundled judgment log. [retrieval-log]

## Sources

- [retrieval-log] Sample retrieval judgments
- [rag-pipeline] What a retrieval pipeline is responsible for
- [agentic-workflow] When one prompt is not enough
- [retrieval-metrics] Precision at k, in plain language
- [prompt-injection] Retrieved text is data, not instructions

## Reviewer note

Auto-approved for an unattended run.

## Run metadata

- Revisions sent back: 1
- Approved: True
- Prompt tokens: 5520
- Completion tokens: 2728
- Estimated cost USD: 0.000000

--- trace ---
planner            0.03 ms  next=research tokens=165+193 cost=0.0
researcher         0.08 ms  retrieved 8; quarantined 1 tokens=1495+543 cost=0.0
planner            0.02 ms  next=analyze tokens=211+193 cost=0.0
analyst            0.03 ms  python ok tokens=180+124 cost=0.0
planner            0.03 ms  next=write tokens=211+193 cost=0.0
writer             0.08 ms  revision=0 chars=2686 tokens=1481+685 cost=0.0
critic             0.02 ms  passed=False score=0.85 send_back=True tokens=100+20 cost=0.0
writer             0.08 ms  revision=1 chars=2957 tokens=1599+754 cost=0.0
critic             0.02 ms  passed=True score=1.00 send_back=False tokens=78+23 cost=0.0
human_review       0.00 ms  approved
finalize           0.53 ms  chars=3152

calls=9 prompt_tokens=5520 completion_tokens=2728 estimated_cost_usd=0.000000
````

A stderr line from that same run:

```json
{"level": "INFO", "logger": "briefwright", "message": "passed=False score=0.85 send_back=True", "node": "critic"}
```

## Evaluation

`briefwright eval` runs three questions through the same graph and scores each brief. The checks are structural: required terms, required sections, citations that resolve to retrieved ids, absence of forbidden strings, and whether the guardrail passed. The full record is `docs/eval_results.json`.

On the mock backend this produced:

| Task | Composite | Guardrail | Revisions | Quarantined |
| --- | --- | --- | --- | --- |
| architecture | 1.0 | passed | 1 | untrusted-paste |
| memory | 1.0 | passed | 1 | none |
| evaluation | 1.0 | passed | 1 | none |

Macro score: **1.0** across 3 tasks. Cost on the mock price table: **0.0 USD**. The architecture task used 5520 prompt tokens and 2728 completion tokens, estimated from text length.

That 1.0 is a real output of the script, and it is a narrow one. The offline writer is deterministic, the corpus is the one the tasks were written against, and the metric does not measure open-domain factuality. It will catch a broken route, a dropped citation, or a leaked payload. It will not tell you that a live model is ready for users. Run it again with `BRIEFWRIGHT_LLM_BACKEND=openai` when you want that second number, and report whatever comes back.

## API

`GET /health` returns the backend and model name.

`POST /v1/research` streams server-sent events: `meta`, `update` for each node, then `done` or `interrupt`.

`POST /v1/research/sync` returns the same payload as JSON.

`POST /v1/research/{thread_id}/resume` with `{"action": "approve", "notes": "..."}` or `"action": "revise"` continues a paused thread. The checkpoint is in process memory, so resume only works on the server that started the thread.

```bash
curl -N -X POST http://127.0.0.1:8000/v1/research \
  -H 'content-type: application/json' \
  -d '{"query": "How should a small team evaluate a RAG assistant?", "auto_approve": true}'
```

Uncheck "Auto-approve" in the Streamlit UI to pause on the human node and approve or send the draft back.

## Configuration

| Variable | Default | Role |
| --- | --- | --- |
| `BRIEFWRIGHT_LLM_BACKEND` | `mock` | `mock` or `openai` |
| `BRIEFWRIGHT_OPENAI_API_KEY` | empty | required for `openai` |
| `BRIEFWRIGHT_OPENAI_BASE_URL` | `https://api.openai.com/v1` | compatible chat endpoint |
| `BRIEFWRIGHT_OPENAI_MODEL` | `gpt-4o-mini` | model name sent to that endpoint |
| `BRIEFWRIGHT_MAX_REVISIONS` | `2` | how many times the critic may send a draft back |
| `BRIEFWRIGHT_SCORE_THRESHOLD` | `0.75` | guardrail score required to pass |
| `BRIEFWRIGHT_MEMORY_PATH` | `data/memory.json` | long-term vector store |
| `BRIEFWRIGHT_LOG_LEVEL` | `INFO` | Python log level |

Per-request `auto_approve` and `max_revisions` on the API override the process defaults for that run.

## Project layout

```text
src/briefwright/
  graph.py            LangGraph wiring
  routing.py          branch decisions, all in code
  state.py            shared state and reducers
  agents/             planner, researcher, analyst, writer, critic, human, finalize
  guardrails/         quarantine and draft checks
  tools/              corpus search and the Python tool
  memory/             hashing embedder and JSON vector store
  llm/                mock model and OpenAI-compatible client
  observability/      JSON logs, spans, price table
  api/                FastAPI app
  ui/streamlit_app.py demo UI
  eval/               task set, metrics, runner
  corpus/documents.json
tests/
docs/PROJECT_EXPLAINER.md
docs/PROJECT_EXPLAINER.pdf
```

## Design decisions

**The graph owns the loop.** A chain is the wrong shape once a critic can send work back, and a free-form tool loop hides the route. LangGraph gives conditional edges, a revision cycle, and an interrupt that checkpoints the thread. See `src/briefwright/graph.py`.

**The model does not pick the next node.** `decide_next` looks at sources, whether the question asks for a metric, and whether a draft exists. The planner can propose a bad route in prose and the edge still does the safe thing. The critic sets `send_back` only when `review_draft` fails and `revision_count` is still under `max_revisions`.

**The guardrail is code.** Required sections, citation ids, quarantined sources, injection phrases, and emails are checked in `src/briefwright/guardrails/checks.py`. The critic model writes a narrative. It cannot flip `passed` to true.

**Retrieved text is untrusted.** `sanitize_untrusted` flags injection lines and obvious personal data. One flag quarantines the whole chunk: the id stays on the trace, the body does not reach the writer.

**Three layers of memory.** The graph state is the working set for one run. `MemorySaver` checkpoints that state under a thread id so a human decision can resume it. The JSON vector store is long-term, and `finalize` writes to it only after the brief is approved and the guardrail passed.

**Numbers come from the tool.** The analyst returns Python, `run_python` executes it against the labeled log, and the stdout is copied into the draft. The sample precision figures are that stdout.

**The mock model is a stand-in, not a hidden network call.** `MockLLM` returns JSON for each role. Token counts are `characters / 4`. The price table rates `mock-research-v1` at zero and `gpt-4o-mini` at the published list price used in `src/briefwright/observability/cost.py`. Swap the client with `create_llm`; the nodes stay the same.

**The first offline draft is missing Limitations on purpose.** That is what makes the revision loop visible with no API key. A live model is asked for the full brief immediately. If it already passes, the loop does not spin. If it does not, the same gate sends it back until the cap.

## Roadmap

- A hosted embedder behind the same `embed` method, kept optional so CI stays offline.
- A durable checkpointer so a paused review survives a process restart.
- A second retriever for the live web, still passed through `sanitize_untrusted`.
- An extra model-graded faithfulness score that cannot override a hard failure.
- Authentication in front of the API, and a real queue instead of the in-process lock.

## License

MIT. See `LICENSE`.
