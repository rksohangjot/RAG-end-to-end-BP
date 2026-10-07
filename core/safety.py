"""Input validation and answer-checking guardrails.

Deterministic checks only (no extra LLM calls): limits on questions and
uploads, and verification that an answer's citations point at real retrieved
chunks. Rejections are returned as readable messages, never raised.
"""

import re

MAX_QUESTION_CHARS = 1000
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_FILES = 10
MAX_DOCUMENT_CHARS = 2_000_000
MAX_OCR_PAGES = 20  # OCR takes a few seconds per page; scanned PDFs over this are rejected

REFUSAL_SENTENCE = "I don't know based on the provided documents."

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# The model sometimes uses its own citation style despite the prompt: full-width
# brackets and/or a line-range suffix, e.g. 【2】 or 【1†L1-L3】. Rewritten to [n]
# before validation so they're checked (and displayed) like any other citation.
_ALT_CITATION = re.compile(r"[【\[]\s*(\d+(?:\s*,\s*\d+)*)\s*†[^】\]]*[】\]]|【\s*(\d+(?:\s*,\s*\d+)*)\s*】")


def sanitize_question(question: str) -> tuple[str, str | None]:
    """Returns (cleaned question, error message or None)."""
    cleaned = _CONTROL_CHARS.sub("", question).strip()
    if not cleaned:
        return "", "Please enter a question."
    if len(cleaned) > MAX_QUESTION_CHARS:
        return "", (
            f"Question is too long ({len(cleaned)} characters). "
            f"The limit is {MAX_QUESTION_CHARS}."
        )
    return cleaned, None


def validate_upload_count(count: int) -> str | None:
    if count > MAX_FILES:
        return f"Too many files ({count}). The limit is {MAX_FILES} per submission."
    return None


def validate_file(file) -> str | None:
    """Returns an error message if the file should be rejected, else None."""
    size = getattr(file, "size", None)
    if size is None:
        file.seek(0, 2)
        size = file.tell()
        file.seek(0)
    if size == 0:
        return "file is empty"
    if size > MAX_FILE_BYTES:
        return f"file is too large ({size / 1024 / 1024:.1f} MB; limit {MAX_FILE_BYTES // 1024 // 1024} MB)"
    return None


def is_refusal(answer: str) -> bool:
    return REFUSAL_SENTENCE.lower() in answer.lower()


def verify_answer(answer: str, num_chunks: int, searched: bool) -> tuple[str, list[str]]:
    """Check an answer against what was actually retrieved.

    Strips citations that don't match a retrieved chunk (the model invented
    them) and returns warnings for answers that look unsupported. This checks
    that a citation exists, not that the cited chunk supports the claim.
    """
    warnings = []
    invalid_found = False
    valid_found = False

    def _fix(match: re.Match) -> str:
        nonlocal invalid_found, valid_found
        numbers = [int(n) for n in re.split(r"\s*,\s*", match.group(1))]
        valid = [n for n in numbers if 1 <= n <= num_chunks]
        if len(valid) != len(numbers):
            invalid_found = True
        if valid:
            valid_found = True
            return "[" + ", ".join(str(n) for n in valid) + "]"
        return ""

    answer = _ALT_CITATION.sub(lambda m: f"[{m.group(1) or m.group(2)}]", answer)
    cleaned = _CITATION.sub(_fix, answer)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)  # tidy gaps left by removals

    if invalid_found:
        warnings.append(
            "The answer referenced sources that don't exist; those citations were removed."
        )

    if not is_refusal(cleaned):
        if not searched:
            warnings.append("This answer was not based on a document search — treat it with caution.")
        elif not valid_found:
            warnings.append("This answer cites no sources — verify it against your documents.")

    return cleaned, warnings
