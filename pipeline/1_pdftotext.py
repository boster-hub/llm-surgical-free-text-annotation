import re
import subprocess

import pandas as pd

from config import (
    ANNOTATION_FILE,
    END_MARKER_PATTERN,
    PDF_DIR,
    PDF_PATTERN,
    START_MARKER,
    TEXT_FILE,
)


def extract_text(pdf_file):
    text = subprocess.check_output(
        [
            "pdftotext",
            "-layout",
            "-enc",
            "UTF-8",
            str(pdf_file),
            "-",
        ],
        text=True,
        encoding="utf-8",
    ).replace("\f", " ")

    if START_MARKER not in text:
        raise ValueError(
            f"Start marker missing: {pdf_file.name}"
        )

    text = text.split(
        START_MARKER,
        maxsplit=1,
    )[1]

    if not re.search(END_MARKER_PATTERN, text):
        raise ValueError(
            f"End marker missing: {pdf_file.name}"
        )

    text = re.split(
        END_MARKER_PATTERN,
        text,
        maxsplit=1,
    )[0].strip()

    return {
        "id": pdf_file.stem.removesuffix("_NP"),
        "text": text,
    }


pdf_files = sorted(
    PDF_DIR.glob(PDF_PATTERN)
)

texts = pd.DataFrame(
    extract_text(pdf_file)
    for pdf_file in pdf_files
)

annotations = pd.read_csv(
    ANNOTATION_FILE,
    dtype={"id": str},
)

assert not texts["id"].duplicated().any()
assert set(texts["id"]) == set(annotations["id"])

texts.sort_values("id").to_csv(
    TEXT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(f"Saved {len(texts)} texts.")
