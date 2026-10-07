"""Load a folder of documents into the persistent knowledge base (./vector_store).

Usage:  python ingest_docs.py [folder]      (default: docs/medical)

Safe to re-run: unchanged files are skipped, a changed file with the same name
replaces its old version. Files with unsupported extensions (e.g. .md) are ignored.
After this, the app answers from these documents without any upload.
"""

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.ingestion import _PARSERS  # noqa: E402
from core.interaction import ingest_and_store  # noqa: E402
from core.vectorstore import EmbeddingModelMismatch, get_collection  # noqa: E402

DEFAULT_FOLDER = ROOT / "docs" / "medical"


class _File(io.BytesIO):
    def __init__(self, path: Path):
        super().__init__(path.read_bytes())
        self.name = path.name


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FOLDER
    if not folder.is_dir():
        print(f"Folder not found: {folder}")
        return 1

    paths = sorted(p for p in folder.iterdir() if p.suffix.lower().lstrip(".") in _PARSERS)
    if not paths:
        print(f"No supported documents in {folder} (supported: {', '.join(sorted(_PARSERS))})")
        return 1

    try:
        get_collection()
    except EmbeddingModelMismatch as e:
        print(e)
        return 1

    print(ingest_and_store([_File(p) for p in paths]))
    print(f"\nKnowledge base now holds {get_collection().count()} chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
