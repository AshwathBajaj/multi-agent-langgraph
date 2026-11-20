"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from briefwright.config import DEFAULT_QUERY, Settings
from briefwright.corpus import get_corpus
from briefwright.eval.runner import run_eval
from briefwright.service import run_research


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="briefwright", description="Briefwright research crew")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run the crew and print the brief")
    run_parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    run_parser.add_argument("--max-revisions", type=int, default=None)
    run_parser.add_argument("--backend", choices=["mock", "openai"], default=None)
    run_parser.add_argument(
        "--no-auto-approve",
        action="store_true",
        help="Pause at human review and print the thread id",
    )
    run_parser.add_argument("--json", action="store_true", dest="as_json")

    eval_parser = sub.add_parser("eval", help="Score the crew on the bundled tasks")
    eval_parser.add_argument("--output", type=Path, default=None)
    eval_parser.add_argument("--backend", choices=["mock", "openai"], default=None)

    retrieve_parser = sub.add_parser(
        "retrieve-eval",
        help="Precision@k of the MiniLM index against SciFact labels (no LLM)",
    )
    retrieve_parser.add_argument("--corpus-dir", default="")

    serve_parser = sub.add_parser("serve", help="Start the API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    ui_parser = sub.add_parser("ui", help="Start the Streamlit demo")
    ui_parser.add_argument("--port", type=int, default=8501)

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "eval":
        return _eval(args)
    if args.command == "retrieve-eval":
        return _retrieve_eval(args.corpus_dir)
    if args.command == "serve":
        return _serve(args.host, args.port)
    if args.command == "ui":
        return _ui(args.port)
    return 2


def _settings(backend: str | None) -> Settings:
    settings = Settings()
    if backend:
        return settings.model_copy(update={"llm_backend": backend})
    return settings


def _run(args: argparse.Namespace) -> int:
    result = run_research(
        args.query,
        settings=_settings(args.backend),
        auto_approve=not args.no_auto_approve,
        max_revisions=args.max_revisions,
    )
    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    if result.interrupted:
        score = result.critique.get("score")
        print(f"Paused for review. thread_id={result.thread_id}")
        print(f"Guardrail passed={result.critique.get('passed')} score={score}")
        print("The checkpoint lives in this process, so it is gone when the command exits.")
        print("Use `briefwright serve` and POST /v1/research with auto_approve false,")
        print("then POST /v1/research/{thread_id}/resume. The Streamlit UI can pause too.")
        return 0
    print(result.report)
    print("\n--- trace ---")
    for event in result.trace:
        cost = event.get("cost_usd", 0)
        tokens = ""
        if "prompt_tokens" in event:
            tokens = f" tokens={event['prompt_tokens']}+{event['completion_tokens']} cost={cost}"
        print(f"{event['node']:14} {event['latency_ms']:8.2f} ms  {event['detail']}{tokens}")
    print(
        f"\ncalls={len(result.usage)} prompt_tokens={result.prompt_tokens} "
        f"completion_tokens={result.completion_tokens} estimated_cost_usd={result.cost_usd:.6f}"
    )
    return 0


def _eval(args: argparse.Namespace) -> int:
    report = run_eval(settings=_settings(args.backend))
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _retrieve_eval(corpus_dir: str) -> int:
    corpus = get_corpus(corpus_dir)
    scores = corpus.mean_precision()
    payload = {
        "dataset": corpus.meta.get("dataset"),
        "sample": bool(corpus.meta.get("sample")),
        "embedder": corpus.embedder_name,
        "llm": None,
        "n_documents": len(corpus.documents),
        "n_labeled_queries": len(corpus.queries),
        "precision_at": {str(k): round(score, 6) for k, score in scores.items()},
        "note": "Dense cosine over stored MiniLM vectors. No language model was called.",
    }
    print(json.dumps(payload, indent=2))
    return 0


def _serve(host: str, port: int) -> int:
    import uvicorn

    uvicorn.run("briefwright.api.app:app", host=host, port=port)
    return 0


def _ui(port: int) -> int:
    from briefwright.ui import streamlit_app

    script = Path(streamlit_app.__file__)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    return subprocess.call(command)
