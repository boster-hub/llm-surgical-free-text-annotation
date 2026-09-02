import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    ANNOTATION_FILE,
    DEVELOPMENT_SIZE,
    RANDOM_SEED,
    SPLIT_DIR,
    SYMPTOMS,
    TEXT_ANALYSIS_FILE,
)


annotations = pd.read_csv(
    ANNOTATION_FILE,
    dtype={"id": str},
)

texts = pd.read_csv(
    TEXT_ANALYSIS_FILE,
    dtype={"id": str},
)

annotations["id"] = annotations["id"].str.strip()
texts["id"] = texts["id"].str.strip()

if texts["id"].duplicated().any():
    raise ValueError(
        "Duplicate IDs in texts_analysis.csv."
    )

analysis_ids = set(texts["id"])

annotations = annotations[
    annotations["id"].isin(analysis_ids)
].copy()

if set(annotations["id"]) != analysis_ids:
    raise ValueError(
        "IDs in texts_analysis.csv and "
        "annotation.csv do not match."
    )

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for symptom in SYMPTOMS:
    develop, validation = train_test_split(
        annotations,
        train_size=DEVELOPMENT_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=annotations[symptom],
    )

    split = pd.concat(
        [
            develop.assign(split="develop"),
            validation.assign(split="validation"),
        ]
    )[["id", "split"]]

    split = split.sort_values(
        "id"
    ).reset_index(
        drop=True
    )

    split.to_csv(
        SPLIT_DIR / f"split_{symptom}.csv",
        index=False,
        encoding="utf-8-sig",
    )
