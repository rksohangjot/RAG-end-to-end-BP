"""End-to-end generation eval over the docs/medical test set (calls the Groq LLM).

Where tests/eval_retrieval.py asks "was the right text retrieved?", this asks "is the
final answer correct?". For each question in docs/medical/QUESTIONS.md it runs the
real agent + verify_answer and checks the answer text with key facts (REQUIRED /
FORBIDDEN below). It does not judge whether a cited chunk supports a claim, and it
cannot tell whether a fact is attached to the right organisation (Q12) -- read the
printed answers for that.

Each run builds a throwaway store from the 5 files (OCR takes about a minute), so
./vector_store is untouched. It uses the Groq free-tier rate limit: a full run is
~16 questions x up to 5 searches each. Run it on demand, not on every change.

Usage (from anywhere):
    python tests/eval_generation.py                 # all 16 questions
    python tests/eval_generation.py --only 12,14    # a subset
    python tests/eval_generation.py --repeat 3      # check run-to-run variance
    python tests/eval_generation.py --show          # print every answer
    python tests/eval_generation.py --real-store    # skip the rebuild, use ./vector_store

Pass rules, answerable questions (Q1-Q14): every REQUIRED item present, no FORBIDDEN
item, not a refusal, and no verify_answer warnings. Refusal questions (Q15-Q16): the
refusal sentence and no [n] citation. API failures count as ERROR, not FAIL.
"""

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_retrieval import GOLD, REFUSAL, _build, _norm  # noqa: E402

from core import chunking  # noqa: E402
from core.agent import run_agent  # noqa: E402
from core.safety import is_refusal, verify_answer  # noqa: E402

# Question number -> required items. Each item is a tuple of accepted alternatives; an
# item is satisfied if ANY alternative appears as a whole word/phrase (after normalising
# case, whitespace, quotes and dashes). All items must be satisfied.
REQUIRED = {
    1: [("cms-1490s",)],
    2: [("doctor", "provider", "supplier")],
    3: [("claim form",), ("itemized bill",), ("letter",), ("supporting documents",)],
    4: [("5", "five"), ("appeal",)],
    5: [("60 days",)],
    6: [("180 days",)],
    7: [("$200",)],
    8: [("65",), ("part a", "parts a"), ("ohio", "oh"), ("indiana",), ("michigan", "mi")],  # no bare "in": it matches any sentence
    9: [("$231",), ("$0",)],
    10: [("full refund",), ("no cancellation fee", "no fee")],
    11: [("45 days",)],
    12: [("summit", "plan claims refund policy", "plan_claims_refund_policy"), ("60 days",), ("medicare",), ("msn", "medicare summary notice")],
    13: [("$286.40",), ("riverbend",)],
    14: [("12 months", "12-month", "twelve months"), ("within", "yes")],
}
# Phrases whose presence means the answer reached the opposite conclusion.
FORBIDDEN = {
    14: [("not within", "outside")],
}
ALL_QUESTIONS = {**{q: question for q, (question, _) in GOLD.items()}, **REFUSAL}
CITATION = re.compile(r"\[\d+\]")


def _text(s: str) -> str:
    """Normalised form for matching (adds en/em dashes and the hyphen variants to _norm)."""
    s = _norm(s)
    for dash in "‐‒–—−":
        s = s.replace(dash, "-")
    return s


def _has(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def _missing(text: str, items: list[tuple]) -> list[str]:
    return [" / ".join(alts) for alts in items if not any(_has(text, a) for a in alts)]


def _found(text: str, items: list[tuple]) -> list[str]:
    return [a for alts in items for a in alts if _has(text, a)]


def check(q: int, answer: str, warnings: list[str]) -> list[str]:
    """Return the reasons this answer fails (empty list = pass)."""
    problems = []
    if q in REFUSAL:
        if not is_refusal(answer):
            problems.append("should have refused but did not")
        if CITATION.search(answer):
            problems.append("refusal contains a [n] citation")
        return problems

    text = _text(answer)
    if is_refusal(answer):
        problems.append("refused to answer")
    problems += [f"missing: {m}" for m in _missing(text, REQUIRED[q])]
    problems += [f"forbidden present: {f}" for f in _found(text, FORBIDDEN.get(q, []))]
    problems += [f"warning: {w}" for w in warnings]
    return problems


def ask(question: str) -> dict:
    started = time.perf_counter()
    result = run_agent(question)
    seconds = time.perf_counter() - started
    if result.failed:
        return {"error": result.answer, "seconds": seconds, "steps": result.steps}
    searched = any(s.tool == "search_documents" for s in result.steps)
    answer, warnings = verify_answer(result.answer, len(result.chunks), searched)
    return {"answer": answer, "warnings": warnings, "seconds": seconds, "steps": result.steps}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", help="comma-separated question numbers, e.g. 12,14")
    parser.add_argument("--repeat", type=int, default=1, help="runs per question (default 1)")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between runs (rate limit)")
    parser.add_argument("--show", action="store_true", help="print every answer, not just failures and Q12")
    parser.add_argument("--real-store", action="store_true", help="use ./vector_store instead of a throwaway store")
    args = parser.parse_args()

    questions = sorted(ALL_QUESTIONS)
    if args.only:
        wanted = {int(n) for n in args.only.split(",")}
        unknown = wanted - set(ALL_QUESTIONS)
        if unknown:
            print(f"Unknown question number(s): {sorted(unknown)}")
            return 1
        questions = sorted(wanted)

    if args.real_store:
        print("Using ./vector_store as is (not rebuilt; results depend on what is stored).")
    else:
        print("Building a throwaway store from docs/medical ...")
        _build(chunking.chunk_document)

    stats = {"pass": 0, "fail": 0, "error": 0}
    answerable_pass = refusal_pass = answerable_total = refusal_total = 0
    searches, seconds, warning_count, runs_done = [], [], 0, 0

    for q in questions:
        is_refusal_q = q in REFUSAL
        outcomes = []
        for run in range(args.repeat):
            if runs_done:
                time.sleep(args.pause)
            runs_done += 1
            out = ask(ALL_QUESTIONS[q])
            seconds.append(out["seconds"])
            searches.append(sum(s.tool == "search_documents" for s in out["steps"]))
            if "error" in out:
                outcomes.append(("ERROR", [out["error"]], out))
                continue
            warning_count += len(out["warnings"])
            problems = check(q, out["answer"], out["warnings"])
            outcomes.append(("PASS" if not problems else "FAIL", problems, out))

        passed = sum(o[0] == "PASS" for o in outcomes)
        errored = sum(o[0] == "ERROR" for o in outcomes)
        status = "ERROR" if errored == len(outcomes) else ("PASS" if passed == len(outcomes) else "FAIL")
        stats[status.lower()] += 1
        if is_refusal_q:
            refusal_total += 1
            refusal_pass += status == "PASS"
        else:
            answerable_total += 1
            answerable_pass += status == "PASS"

        tally = f" ({passed}/{len(outcomes)} runs)" if args.repeat > 1 else ""
        print(f"\nQ{q:<2} {status}{tally}  {ALL_QUESTIONS[q]}")
        for label, problems, out in outcomes:
            for p in problems:
                print(f"     - {p}")
            if "answer" in out and (args.show or label == "FAIL" or q == 12):
                body = out["answer"].strip().replace("\n", "\n       ")
                print(f"     answer ({out['seconds']:.1f}s, {sum(s.tool == 'search_documents' for s in out['steps'])} searches):\n       {body}")

    print("\n" + "=" * 70)
    print(f"answerable correct: {answerable_pass}/{answerable_total} | refusals correct: {refusal_pass}/{refusal_total}"
          f" | errors: {stats['error']}")
    if seconds:
        print(f"avg {sum(searches) / len(searches):.1f} searches and {sum(seconds) / len(seconds):.1f}s per question"
              f" | verify_answer warnings: {warning_count}")
    if stats["error"]:
        print("ERROR = the API call failed (key, rate limit, network); those are not quality failures.")
    print("Q12 needs a manual read: the check cannot tell which organisation each fact belongs to.")
    return 0 if not stats["fail"] and not stats["error"] else 2


if __name__ == "__main__":
    sys.exit(main())
