"""Interaction boundary between the UI and the RAG pipeline.

handle_query validates input, ingests any uploaded files (skipping ones already
stored), then hands the question to the agent, which searches the full
knowledge base as many times as it needs and answers. The answer is checked
against what was actually retrieved before it is returned.
"""

import hashlib
from dataclasses import dataclass, field

from core.agent import AgentStep, run_agent
from core.chunking import chunk_document
from core.ingestion import ingest
from core.logger import logger
from core.safety import (
    is_refusal,
    sanitize_question,
    validate_file,
    validate_upload_count,
    verify_answer,
)
from core.vectorstore import (
    EmbeddingModelMismatch,
    add_chunks,
    delete_ids,
    find_by_hash,
    get_collection,
    ids_for_source,
)


@dataclass
class QueryResult:
    text: str
    steps: list[AgentStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def ingest_and_store(files: list) -> str:
    lines = []
    for f in files:
        error = validate_file(f)
        if error:
            logger.warning("Rejected upload %r: %s", f.name, error)
            lines.append(f"- {f.name}: [rejected: {error}]")
            continue

        doc = ingest(f)
        if "error" in doc.metadata:
            logger.warning("Ingestion failed for %r: %s", doc.name, doc.metadata["error"])
            lines.append(f"- {doc.name}: [error: {doc.metadata['error']}]")
            continue

        content_hash = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
        existing = find_by_hash(content_hash)
        if existing:
            logger.info("Skipped duplicate upload %r (already stored as %r)", doc.name, existing)
            lines.append(f"- {doc.name}: already in the knowledge base (as {existing}) — skipped.")
            continue

        chunks = chunk_document(doc)
        if not chunks:
            lines.append(f"- {doc.name}: 0 chunks (no text extracted)")
            continue

        # Same filename, different content = a new version: swap out the old chunks.
        # The new version is stored first, so a failure never loses the old one.
        old_ids = ids_for_source(doc.name)
        add_chunks(chunks, content_hash=content_hash)
        delete_ids(old_ids)

        if old_ids:
            logger.info("Replaced %d stored chunk(s) of %r with %d new", len(old_ids), doc.name, len(chunks))
            lines.append(
                f"- {doc.name}: updated — {len(chunks)} chunk(s) stored "
                f"(replaced the previous version's {len(old_ids)})."
            )
        else:
            lines.append(f"- {doc.name}: {len(chunks)} chunk(s) embedded & stored.")

    return "\n".join(lines)


def handle_query(files: list, question: str) -> QueryResult:
    question, error = sanitize_question(question)
    if error:
        logger.info("Rejected question: %s", error)
        return QueryResult(error)

    count_error = validate_upload_count(len(files))
    if count_error:
        logger.info("Rejected submission: %s", count_error)
        return QueryResult(count_error)

    try:
        get_collection()
    except EmbeddingModelMismatch as e:
        logger.error("Embedding model mismatch: %s", e)
        return QueryResult(str(e))

    sections = []

    if files:
        sections.append("Uploaded documents:\n" + ingest_and_store(files))

    if get_collection().count() == 0:
        sections.append(
            "No documents in the knowledge base yet. Please upload at least one document."
        )
        return QueryResult("\n\n".join(sections))

    result = run_agent(question)
    if result.failed:
        sections.append(result.answer)
        return QueryResult("\n\n".join(sections), result.steps)

    searched = any(step.tool == "search_documents" for step in result.steps)
    answer, warnings = verify_answer(result.answer, len(result.chunks), searched)
    sections.append(answer)

    if result.chunks and not is_refusal(answer):
        sources_text = "\n".join(
            f"- [{i}] {c.source} #{c.chunk_index} (score={c.score:.3f})"
            for i, c in enumerate(result.chunks, start=1)
        )
        sections.append(f"Sources:\n{sources_text}")

    return QueryResult("\n\n".join(sections), result.steps, warnings)
