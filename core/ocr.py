"""OCR for scanned pages and images (local, no API key).

Uses RapidOCR (ONNX models, downloaded once on first use). The text boxes it
returns are regrouped into lines by vertical position, so a form row like
"Total billed:" + "$286.40" comes out as one line instead of two separate blocks.
"""

import logging

import numpy as np

logging.getLogger("RapidOCR").setLevel(logging.WARNING)

_engine = None

# A box joins an existing line if its vertical centre is within this fraction
# of that line's height. Tuned on a single-column form; multi-column scans may
# interleave.
_LINE_TOLERANCE = 0.6


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR

        _engine = RapidOCR()
    return _engine


def image_to_text(image) -> str:
    """Run OCR on a PIL image and return its text, one line per visual row."""
    result = _get_engine()(np.array(image.convert("RGB")))
    if result.boxes is None or len(result.boxes) == 0:
        return ""

    boxes = [(np.array(b), t) for b, t in zip(result.boxes, result.txts)]
    boxes.sort(key=lambda item: item[0][:, 1].mean())

    rows = []
    for box, text in boxes:
        centre = box[:, 1].mean()
        height = box[:, 1].max() - box[:, 1].min()
        for row in rows:
            if abs(row["centre"] - centre) < _LINE_TOLERANCE * row["height"]:
                row["items"].append((box[:, 0].min(), text))
                break
        else:
            rows.append({"centre": centre, "height": height, "items": [(box[:, 0].min(), text)]})

    return "\n".join("   ".join(t for _, t in sorted(row["items"])) for row in rows)
