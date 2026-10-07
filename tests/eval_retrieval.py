"""Retrieval-only eval over the docs/medical test set (no LLM, no API calls).

For each question in docs/medical/QUESTIONS.md that has a document answer, the
evidence strings that must appear in the retrieved chunks are listed in GOLD.
Matching is on text, not chunk ids, so any chunking change can be compared.
Each variant is built into a throwaway Chroma store; ./vector_store is untouched.

Run from anywhere:  python tests/eval_retrieval.py
Note: queries use the raw question; the agent rewrites queries and can search
several times, so real behaviour is better than these numbers.
"""

import io
import re
import statistics
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core.vectorstore as vs  # noqa: E402
from core import chunking  # noqa: E402
from core.ingestion import ingest  # noqa: E402
from core.retrieval import retrieve  # noqa: E402
from core.vectorstore import add_chunks  # noqa: E402

DOCS = ROOT / "docs" / "medical"
FILES = [
    "medicare_claims_appeals.pdf",
    "medicare_how_to_file_claim.html",
    "plan_claims_refund_policy.docx",
    "scanned_claim_form.pdf",
    "scanned_claim_form.docx",
]

# question number -> (question, evidence strings that must ALL appear in the retrieved chunks)
GOLD = {
    1: ("What form do I use to file a Medicare claim myself?", ["Patient Request for Medical Payment form (CMS-1490S)"]),
    2: ("Who files claims for me under Original Medicare?", ["the law requires your doctor, provider, or supplier to file Medicare claims"]),
    3: ("What do I need to send with a claim I file myself?", ["The completed claim form", "The itemized bill from your doctor", "A letter explaining why", "Any supporting documents"]),
    4: ("How many levels of appeal are there in Original Medicare?", ["There are 5 levels of appeal in Original Medicare"]),
    5: ("How long does the MAC generally take to decide a Level 1 appeal?", ["within 60 days after they get your appeal"]),
    6: ("After the MAC's decision, how long do I have to ask for a QIC reconsideration?", ["You have 180 days after you get the MAC's decision letter"]),
    7: ("What is the 2026 minimum dollar amount for an OMHA hearing?", ["The minimum dollar amount for 2026 is $200"]),
    8: ("Who can enroll in a Summit Health Supplement Plan?", ["at least 65 years old", "Ohio, Indiana and Michigan"]),
    9: ("What are the premium and deductible for Summit Plan C?", ["Plan C (Premier), Monthly premium: $231, Annual deductible: $0"]),
    10: ("What refund do I get if I cancel a Summit plan within 30 days?", ["full refund of all premiums paid, with no cancellation fee"]),
    11: ("How long does Summit take to refund a billing overpayment?", ["within 45 days of confirming the error", "up to 60 days"]),
    12: ("How long do I have to appeal a denied claim?", ["you have 60 days from the date of the denial notice", "You must file your appeal by the date listed in your MSN"]),
    13: ("What was the total billed on the scanned claim form, and which provider?", ["Total billed: $286.40", "Riverbend Family Clinic"]),
    14: ("Was the scanned claim submitted within Summit's filing window?", ["within 12 months of the date of service", "Date of service: 08/22/2025", "Date submitted: 10/02/2025"]),
}
REFUSAL = {15: "Does Summit cover dental care?", 16: "What dose of metformin should I take?"}


class _Upload(io.BytesIO):
    def __init__(self, path: Path):
        super().__init__(path.read_bytes())
        self.name = path.name


def _norm(s: str) -> str:
    """Case-, whitespace-, quote- and hyphen-insensitive form for matching."""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("‑", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def _build(chunker) -> list:
    vs.PERSIST_DIR = tempfile.mkdtemp(prefix="rag_eval_")  # fresh throwaway store
    vs._collection = None
    chunks = []
    for name in FILES:
        doc_chunks = chunker(ingest(_Upload(DOCS / name)))
        add_chunks(doc_chunks)
        chunks += doc_chunks
    return chunks


def run(chunker, label: str, verbose: bool = True, rerank: bool | None = None) -> dict:
    chunks = _build(chunker)
    sizes = [len(c.text) for c in chunks]
    print(f"\n[{label}] {len(chunks)} chunks | chars: min {min(sizes)}, median {int(statistics.median(sizes))}, max {max(sizes)}")

    hits = {3: 0, 5: 0}
    started = time.perf_counter()
    worst_ranks = []
    per_question = {}
    # Per-question precision@k / recall@k. A chunk is relevant if it contains any of the
    # question's evidence strings; recall is the share of evidence strings found in the
    # top k. Precision divides by the chunks actually returned (the relevance threshold
    # can return fewer than k).
    precision = {3: [], 5: []}
    recall = {3: [], 5: []}
    for q, (question, needles) in GOLD.items():
        texts = [_norm(r.text) for r in retrieve(question, top_k=5, rerank=rerank)]
        found = {}
        for needle in needles:
            ranks = [i + 1 for i, t in enumerate(texts) if _norm(needle) in t]
            found[needle] = ranks[0] if ranks else None
        for k in (3, 5):
            top = texts[:k]
            if top:
                precision[k].append(sum(any(_norm(n) in t for n in needles) for t in top) / len(top))
            else:
                precision[k].append(0.0)
            recall[k].append(sum(v is not None and v <= k for v in found.values()) / len(needles))
        worst = None if any(v is None for v in found.values()) else max(found.values())
        per_question[q] = worst
        for k in (3, 5):
            hits[k] += bool(worst and worst <= k)
        if worst:
            worst_ranks.append(worst)
        if verbose:
            status = "HIT@3" if worst and worst <= 3 else ("hit@5" if worst else "MISS ")
            detail = "; ".join(f"{n[:30]!r}@{v}" for n, v in found.items())
            print(f"  Q{q:<2} {status}  {detail}")

    per_search_ms = (time.perf_counter() - started) / len(GOLD) * 1000
    print(f"  => top-3: {hits[3]}/{len(GOLD)} | top-5: {hits[5]}/{len(GOLD)} | mean worst-evidence rank: {statistics.mean(worst_ranks):.2f} | {per_search_ms:.0f} ms/search")
    print(
        f"  => precision@3 {statistics.mean(precision[3]):.2f} | recall@3 {statistics.mean(recall[3]):.2f}"
        f" | precision@5 {statistics.mean(precision[5]):.2f} | recall@5 {statistics.mean(recall[5]):.2f}"
    )
    for q, question in REFUSAL.items():
        res = retrieve(question, top_k=5, rerank=rerank)
        print(f"  Q{q} (refusal): {len(res)} chunk(s) above threshold" + (f", best {res[0].score:.3f}" if res else ""))
    return {
        "hits": hits,
        "ranks": per_question,
        "chunks": len(chunks),
        "precision": {k: statistics.mean(v) for k, v in precision.items()},
        "recall": {k: statistics.mean(v) for k, v in recall.items()},
    }


if __name__ == "__main__":
    # To compare a chunking variant, pass your own chunker, e.g.
    # run(partial(chunking.chunk_document, chunk_size=800), "800-char chunks")
    run(chunking.chunk_document, "current chunking, cosine only", rerank=False)
    run(chunking.chunk_document, "current chunking, + reranker", rerank=True)
