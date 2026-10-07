# Basic RAG Project

A question-answering app over your own documents. Upload PDF, DOCX, HTML, TXT, CSV or Excel files (scanned pages are OCR'd), ask a question in plain English, and get an answer grounded in those documents, with numbered citations and a list of the sources used. If the documents don't cover the question, the app says so instead of guessing.

Built from scratch with no RAG framework: Streamlit for the UI, `sentence-transformers` + Chroma for retrieval, and a hand-written tool-calling agent on Groq.

## How it works

```
upload ─► ingest ─► chunk ─► embed ─► Chroma (./vector_store)
                                          │
question ─► agent ─► search_documents ────┘
              │         (cosine top-20 ─► cross-encoder rerank ─► top-5)
              ▼
         answer + [n] citations ─► verify_answer ─► UI
```

1. **Ingestion** (`core/ingestion.py`, `core/ocr.py`): parses each file to plain text. Table rows become `col: val, col: val` lines so they stay self-describing. Scanned PDF pages and images inside DOCX files are read with local OCR (RapidOCR).
2. **Chunking** (`core/chunking.py`): about 500 characters with 50 of overlap, splitting at paragraph, then sentence, then word boundaries.
3. **Vector store** (`core/vectorstore.py`): `all-MiniLM-L6-v2` embeddings stored in a persistent Chroma collection (cosine similarity). Identical re-uploads are skipped, and a changed file with the same name replaces the old version.
4. **Retrieval** (`core/retrieval.py`): fetches 20 candidates, drops anything below a cosine score of 0.25, reranks with a cross-encoder (`ms-marco-MiniLM-L-6-v2`) and keeps the best 5.
5. **Agent** (`core/agent.py`): the model (`openai/gpt-oss-120b` on Groq) chooses between `search_documents` and `list_documents`, can rephrase and search again (up to 5 tool rounds), then answers.
6. **Safety** (`core/safety.py`, `core/logger.py`): limits on question and upload size, removal of citations that don't match a retrieved chunk, warnings for answers that cite nothing or weren't based on a search, and prompt-injection hardening of document text. Errors are logged to `app.log` (never document or question text).

The UI (`app.py`) only calls `handle_query()` in `core/interaction.py`, which is the single seam between the UI and the pipeline.

## Setup

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

Get a free API key at [console.groq.com](https://console.groq.com), then create a `.env` file in the project root (see `.env.example`):

```
GROQ_API_KEY=your-key-here
```

On first use the app downloads the embedding model, the reranker (about 90 MB) and the OCR models (about 31 MB), so the first run needs an internet connection and is slower.

## Run

```bash
streamlit run app.py
```

Upload files (optional) and ask a question. Uploaded documents are added to the knowledge base, which persists in `./vector_store/`, so follow-up questions don't need a re-upload.

To load the bundled sample documents without using the UI:

```bash
python ingest_docs.py            # defaults to docs/medical
python ingest_docs.py my_folder  # or any folder
```

It is safe to re-run: unchanged files are skipped and changed ones are replaced.

## Evaluation

Sample documents and 16 test questions are in `docs/medical/` (`QUESTIONS.md`, `SOURCES.md`).

```bash
python tests/eval_retrieval.py                 # no API calls
python tests/eval_generation.py                # calls Groq, uses free-tier rate limit
python tests/eval_generation.py --only 12,14 --repeat 3 --show
```

Results from the last runs:

| Check | Result |
|---|---|
| Retrieval, cosine only | top-3 11/14, top-5 12/14 |
| Retrieval, with reranker | top-3 12/14, top-5 13/14 |
| Generation, answerable questions | 13/14 questions (40/42 runs, 95.2%) |
| Generation, refusals | 2/2 questions (6/6 runs) |

The generation eval checks that key facts appear in the answer, not that the cited chunk supports them. Three samples per question is a small sample.

## Project structure

```
app.py               Streamlit UI
ingest_docs.py       load a folder into the vector store from the command line
core/
  interaction.py     handle_query(): validation, ingestion, agent, verification
  ingestion.py       file parsers -> Document
  ocr.py             RapidOCR wrapper
  chunking.py        text -> Chunk objects
  vectorstore.py     embeddings + Chroma
  retrieval.py       search + rerank
  generation.py      Groq client, prompts, single-shot answer (agent fallback)
  agent.py           tool-calling loop
  safety.py          limits and answer verification
  logger.py          file logging
docs/medical/        sample documents and test questions
tests/               retrieval and generation evals
```

## Limits and known gaps

- **Limits:** question up to 1000 characters, 10 files per submission, 10 MB per file, 2M characters of extracted text per document, 20 OCR pages or images per document.
- **No conversation memory:** each question is answered independently.
- **Citations are checked for existence, not support:** a real but irrelevant `[2]` passes verification.
- **Refusals partly depend on the model:** the 0.25 relevance cutoff is a heuristic, and off-topic questions can still pass it.
- **Not supported:** legacy `.doc`, DOCX headers, footers, text boxes and footnotes, nested DOCX tables, handwriting. Multi-column scans may OCR out of order.
- **Known eval misses:** retrieval for Q12 (second piece of evidence never reaches the top 5), and generation for Q3 (passed 1 of 3 runs; the model leaves out part of the evidence).
- **Changing the embedding model** requires deleting `vector_store/` and re-uploading; there is no automatic re-index.
- **Same-name files are treated as versions of one document.**
- **No authentication:** everyone who can open the app sees the same knowledge base. Don't use it for sensitive documents on a shared server.
- Every query spends Groq free-tier rate limit; multi-search questions spend more.
