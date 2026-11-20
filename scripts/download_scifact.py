"""Download SciFact, embed it with MiniLM on CPU, and refresh the offline sample.

The full corpus and model weights stay in data/cache (gitignored). The committed
sample is a few labeled dev claims plus the documents needed to reproduce their
top-k ranking. CI uses those stored vectors and does not download anything.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from briefwright.retrieval_metrics import mean_precision

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "scifact"
MODEL_CACHE = ROOT / "data" / "cache" / "fastembed"
SAMPLE = ROOT / "src" / "briefwright" / "corpus" / "sample"
TARBALL_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


def download_tarball(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "data.tar.gz"
    if archive.exists() and archive.stat().st_size > 0:
        print(f"cache hit {archive}")
        return archive
    print(f"downloading {TARBALL_URL}")
    urllib.request.urlretrieve(TARBALL_URL, archive)  # noqa: S310 - fixed official URL
    return archive


def extract_tarball(archive: Path, destination: Path) -> Path:
    data_dir = destination / "raw"
    marker = data_dir / "corpus.jsonl"
    if marker.exists():
        print(f"cache hit {data_dir}")
        return data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(data_dir, filter="data")
        except TypeError:
            tar.extractall(data_dir)
    return data_dir


def load_raw(data_dir: Path) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    corpus_path = next(data_dir.rglob("corpus.jsonl"))
    claims_path = next(data_dir.rglob("claims_dev.jsonl"))
    documents = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        abstract = raw.get("abstract") or []
        if isinstance(abstract, list):
            body = " ".join(str(sentence) for sentence in abstract)
        else:
            body = str(abstract)
        documents.append(
            {
                "id": str(raw["doc_id"]),
                "title": str(raw.get("title") or "").strip(),
                "text": body.strip(),
            }
        )
    documents.sort(key=lambda doc: doc["id"])
    queries = []
    qrels: dict[str, list[str]] = {}
    for line in claims_path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        evidence = raw.get("evidence") or {}
        relevant = sorted(str(doc_id) for doc_id in evidence)
        if not relevant:
            continue
        query_id = str(raw["id"])
        queries.append({"id": query_id, "text": str(raw["claim"]).strip()})
        qrels[query_id] = relevant
    queries.sort(key=lambda query: int(query["id"]))
    return documents, queries, qrels


def embed_texts(texts: list[str]) -> np.ndarray:
    from fastembed import TextEmbedding

    model = TextEmbedding(MODEL_NAME, cache_dir=str(MODEL_CACHE))
    rows = [np.asarray(vector, dtype=np.float32) for vector in model.embed(texts, batch_size=64)]
    matrix = np.vstack(rows)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def rankings(doc_matrix: np.ndarray, query_matrix: np.ndarray, k: int) -> np.ndarray:
    scores = query_matrix @ doc_matrix.T
    if k >= scores.shape[1]:
        order = np.argsort(-scores, axis=1)
        return order
    # argpartition then sort the head, so the full dev set stays cheap
    cut = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    head_scores = scores[row, cut]
    head_order = np.argsort(-head_scores, axis=1)
    return cut[row, head_order]


def flags_for(order: np.ndarray, doc_ids: list[str], relevant: set[str], k: int) -> list[int]:
    return [1 if doc_ids[int(index)] in relevant else 0 for index in order[:k]]


def choose_sample(
    queries: list[dict],
    qrels: dict[str, list[str]],
    order: np.ndarray,
    doc_ids: list[str],
    n: int = 3,
) -> list[dict]:
    ranked = []
    for row, query in enumerate(queries):
        relevant = set(qrels[query["id"]])
        if len(relevant) < 2:
            continue
        words = query["text"].split()
        if not 6 <= len(words) <= 28:
            continue
        top = [doc_ids[int(index)] for index in order[row, :5]]
        hits = sum(1 for doc_id in top if doc_id in relevant)
        if hits < 1:
            continue
        ranked.append((-hits, -len(relevant), int(query["id"]), query))
    ranked.sort()
    chosen: list[dict] = []
    used: set[str] = set()
    for *_rest, query in ranked:
        relevant = set(qrels[query["id"]])
        if relevant & used:
            continue
        chosen.append(query)
        used |= relevant
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise SystemExit(f"Only found {len(chosen)} sample claims, wanted {n}.")
    return chosen


def round_vector(vector: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in vector]


def write_sample(
    documents: list[dict],
    queries: list[dict],
    qrels: dict[str, list[str]],
    doc_ids: list[str],
    doc_matrix: np.ndarray,
    query_matrix: np.ndarray,
    order: np.ndarray,
    full_metrics: dict,
) -> None:
    id_to_row = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    query_row = {query["id"]: index for index, query in enumerate(queries)}
    picked = choose_sample(queries, qrels, order, doc_ids)
    keep_docs: set[str] = set()
    for query in picked:
        row = query_row[query["id"]]
        keep_docs.update(doc_ids[int(index)] for index in order[row, :10])
        keep_docs.update(qrels[query["id"]])
    sample_docs = [documents[id_to_row[doc_id]] for doc_id in sorted(keep_docs)]
    sample_queries = picked
    sample_qrels = {query["id"]: qrels[query["id"]] for query in picked}
    vectors = {
        "model": MODEL_NAME,
        "dim": EMBED_DIM,
        "documents": {doc["id"]: round_vector(doc_matrix[id_to_row[doc["id"]]]) for doc in sample_docs},
        "queries": {
            query["id"]: round_vector(query_matrix[query_row[query["id"]]]) for query in sample_queries
        },
    }
    meta = {
        "dataset": "SciFact",
        "split": "dev",
        "source_url": TARBALL_URL,
        "paper": "Wadden et al., Fact or Fiction: Verifying Scientific Claims, EMNLP 2020",
        "paper_url": "https://aclanthology.org/2020.emnlp-main.609/",
        "corpus_origin": "S2ORC abstracts bundled in the SciFact release",
        "embedder": MODEL_NAME,
        "embedder_license": "Apache-2.0",
        "embedding_dim": EMBED_DIM,
        "sample": True,
        "n_documents": len(sample_docs),
        "n_labeled_queries": len(sample_queries),
        "full_dev_documents": len(documents),
        "full_dev_labeled_queries": len(queries),
        "relevance": "A document is relevant when it is an evidence abstract in the SciFact dev claim.",
        "full_dense_precision": full_metrics,
    }
    SAMPLE.mkdir(parents=True, exist_ok=True)
    (SAMPLE / "documents.json").write_text(json.dumps(sample_docs, indent=2), encoding="utf-8")
    (SAMPLE / "queries.json").write_text(json.dumps(sample_queries, indent=2), encoding="utf-8")
    (SAMPLE / "qrels.json").write_text(json.dumps(sample_qrels, indent=2), encoding="utf-8")
    (SAMPLE / "vectors.json").write_text(json.dumps(vectors), encoding="utf-8")
    (SAMPLE / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("sample claims:")
    for query in sample_queries:
        print(f"  {query['id']}: {query['text']}")
    print(f"sample documents: {len(sample_docs)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SciFact and embed it with MiniLM")
    parser.add_argument("--write-sample", action="store_true")
    parser.add_argument("--metrics-out", type=Path, default=None)
    args = parser.parse_args()

    archive = download_tarball(CACHE)
    raw = extract_tarball(archive, CACHE)
    documents, queries, qrels = load_raw(raw)
    print(f"documents={len(documents)} labeled_dev_queries={len(queries)}")

    doc_ids = [doc["id"] for doc in documents]
    doc_texts = [f"{doc['title']}. {doc['text']}".strip() for doc in documents]
    query_texts = [query["text"] for query in queries]
    print("embedding with", MODEL_NAME)
    doc_matrix = embed_texts(doc_texts)
    query_matrix = embed_texts(query_texts)
    if doc_matrix.shape[1] != EMBED_DIM:
        raise SystemExit(f"Unexpected embedding dim {doc_matrix.shape[1]}")

    order = rankings(doc_matrix, query_matrix, k=10)
    rows = [
        flags_for(order[row], doc_ids, set(qrels[query["id"]]), k=10) for row, query in enumerate(queries)
    ]
    scores = mean_precision(rows, ks=(1, 3, 5, 10))
    metrics = {
        "dataset": "SciFact",
        "split": "dev",
        "embedder": MODEL_NAME,
        "retrieval": "dense cosine, title + abstract",
        "n_documents": len(documents),
        "n_labeled_queries": len(queries),
        "precision": {str(k): round(score, 6) for k, score in scores.items()},
        "llm": None,
        "note": "Retrieval only. No language model was used for these figures.",
    }
    print(json.dumps(metrics, indent=2))
    prepared = CACHE / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if args.write_sample:
        write_sample(
            documents, queries, qrels, doc_ids, doc_matrix, query_matrix, order, metrics["precision"]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
