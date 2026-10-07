"""Application logging to a local file.

Records errors, tool failures, fallbacks, and rejected inputs so problems can
be diagnosed after the fact. Never log document text or question text — only
metadata like names, sizes, and lengths.
"""

import logging
from pathlib import Path

LOG_FILE = str(Path(__file__).resolve().parent.parent / "app.log")

logger = logging.getLogger("rag")

if not logger.handlers:  # Streamlit re-imports modules; don't stack handlers
    logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False
