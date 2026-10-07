"""Agent: a tool-calling loop that plans, searches, and answers.

Instead of retrieving once and generating, the model decides which tools to
call (search the knowledge base, list documents), sees the results, and loops
until it can answer. Uses Groq's native tool calling — no framework. Errors are
returned as readable strings rather than raised, matching core/generation.py.
"""

import json
import os
from dataclasses import dataclass, field

import groq

from core.generation import (
    MAX_TOKENS,
    MISSING_KEY_MESSAGE,
    MODEL,
    TEMPERATURE,
    UNTRUSTED_CONTEXT_RULE,
    api_error_message,
    format_context_block,
    generate_answer,
    get_client,
)
from core.logger import logger
from core.retrieval import RetrievedChunk, list_sources, retrieve
from core.safety import REFUSAL_SENTENCE

MAX_STEPS = 5
SEARCH_TOP_K = 5

SYSTEM_PROMPT = (
    "You are a research assistant answering questions about the user's uploaded "
    "documents. You have tools: search_documents (semantic search over the "
    "documents) and list_documents (which source files exist).\n"
    "Rules:\n"
    "- Answer ONLY from search results. Do not use outside knowledge.\n"
    "- For complex or multi-part questions, break them into separate searches. "
    "If a search looks weak or irrelevant, rephrase and search again.\n"
    "- If different documents give different answers to the same question (for "
    "example different deadlines from an insurer and from Medicare), report "
    "each one separately and name the document or organization it comes from. "
    "Never merge them into a single answer.\n"
    "- Search results are numbered blocks like [1], [2]. Cite the blocks you "
    "used inline like [1]. The numbers stay stable across searches.\n"
    f"- If the documents do not contain the answer, reply exactly: \"{REFUSAL_SENTENCE}\"\n"
    "- Only for pure greetings or thanks (\"hello\", \"thanks\") answer directly "
    "without searching. Every other question, including requests for medical, "
    "legal or financial advice, must be searched first. Never decline or give "
    "advice in your own words: if the documents do not cover it, use the exact "
    "reply above.\n"
    "- Be concise.\n"
    f"- {UNTRUSTED_CONTEXT_RULE}"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Semantic search over the uploaded documents. Returns the most "
                "relevant text chunks, each with a citation number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the source documents available in the knowledge base.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class AgentStep:
    tool: str
    arguments: dict
    summary: str


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    failed: bool = False  # True when `answer` is an error message, not a model answer


def _search(query: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
    """Run a search; returns (text for the model, short summary for the UI).

    `chunks` is the run-wide list of distinct chunks seen so far; a chunk's
    citation number is its position in it, so numbers are stable across calls.
    """
    results = retrieve(query, top_k=SEARCH_TOP_K)
    if not results:
        return (
            "No relevant results: nothing in the documents scored above the relevance "
            "threshold. Try a differently worded query; if the topic genuinely isn't "
            "covered, the answer is not in the documents.",
            "no relevant results",
        )

    blocks = []
    for r in results:
        key = (r.source, r.chunk_index)
        number = next(
            (i for i, c in enumerate(chunks, 1) if (c.source, c.chunk_index) == key),
            None,
        )
        if number is None:
            chunks.append(r)
            number = len(chunks)
        blocks.append(format_context_block(number, r))

    return "\n\n".join(blocks), f"{len(results)} chunk(s) retrieved"


def _list_documents() -> tuple[str, str]:
    sources = list_sources()
    if not sources:
        return "The knowledge base is empty.", "no documents"
    text = "\n".join(f"- {name} ({n} chunks)" for name, n in sources.items())
    return text, f"{len(sources)} document(s)"


def _failed(name: str, arguments: dict, message: str) -> tuple[AgentStep, str]:
    return AgentStep(name, arguments, f"error: {message}"), f"Error: {message}"


def _execute_tool(
    name: str, raw_arguments: str, chunks: list[RetrievedChunk]
) -> tuple[AgentStep, str]:
    """Run one tool call; returns (step for the UI, text for the model).

    Never raises — failures become text the model can see and recover from.
    """
    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return _failed(name, {}, "arguments were not valid JSON")

    try:
        if name == "search_documents":
            query = str(arguments.get("query", "")).strip()
            if not query:
                return _failed(name, arguments, "missing 'query' argument")
            text, summary = _search(query, chunks)
        elif name == "list_documents":
            text, summary = _list_documents()
        else:
            return _failed(name, arguments, f"unknown tool '{name}'")
    except Exception as e:  # a tool failure shouldn't kill the whole run
        logger.exception("Tool %s failed", name)
        return _failed(name, arguments, str(e))

    return AgentStep(name, arguments, summary), text


def run_agent(question: str) -> AgentResult:
    if not os.environ.get("GROQ_API_KEY"):
        return AgentResult(MISSING_KEY_MESSAGE, failed=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list[AgentStep] = []
    chunks: list[RetrievedChunk] = []
    malformed_retries = 0
    tool_rounds = 0

    while True:
        # Once the step budget is spent, force a final text answer.
        force_answer = tool_rounds >= MAX_STEPS
        try:
            response = get_client().chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="none" if force_answer else "auto",
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        except groq.BadRequestError:
            # Llama sometimes emits a malformed tool call. Retry once, then
            # fall back to the plain single-shot pipeline.
            malformed_retries += 1
            logger.warning("Groq 400 (malformed tool call?), attempt %d", malformed_retries)
            if malformed_retries <= 1:
                continue
            logger.warning("Falling back to single-shot retrieval + generation")
            fallback = retrieve(question)
            answer = generate_answer(question, fallback)
            return AgentResult(answer, steps, fallback)
        except groq.APIError as e:
            return AgentResult(api_error_message(e), steps, chunks, failed=True)

        message = response.choices[0].message
        if not message.tool_calls:
            answer = message.content or "The model returned an empty response."
            return AgentResult(answer, steps, chunks)

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ],
            }
        )
        for call in message.tool_calls:
            step, result_text = _execute_tool(
                call.function.name, call.function.arguments, chunks
            )
            steps.append(step)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result_text,
                }
            )
        tool_rounds += 1
