import pandas as pd

from config import (
    ANNOTATION_FILE,
    ERROR_ANALYSIS_FILE,
    POST_OUTPUT_DIR,
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

files = sorted(
    POST_OUTPUT_DIR.glob(
        "validation_*_validation_*_post.csv"
    )
)

if not files:
    raise ValueError(
        "No LLM validation files found."
    )

validation = pd.concat(
    [
        pd.read_csv(
            file,
            dtype={"id": str},
        )
        for file in files
    ],
    ignore_index=True,
)

validation = validation[
    (validation["analysis"] == "validation")
    & (validation["split"] == "validation")
].copy()

review_rows = []

for symptom in SYMPTOMS:
    data = validation[
        validation["symptom"] == symptom
    ].copy()

    data = (
        data
        .merge(
            annotations[
                ["id", symptom]
            ].rename(
                columns={
                    symptom: "reference"
                }
            ),
            on="id",
            validate="one_to_one",
        )
        .merge(
            texts[
                ["id", "text"]
            ],
            on="id",
            validate="one_to_one",
        )
    )

    data = data[
        data["prediction"]
        != data["reference"]
    ].copy()

    data["discrepancy_type"] = (
        data.apply(
            lambda row: (
                "false_positive"
                if row["prediction"] == "ja"
                else "false_negative"
            ),
            axis=1,
        )
    )

    data["category"] = ""
    data["consensus_note"] = ""

    review_rows.append(
        data[
            [
                "symptom",
                "model",
                "id",
                "reference",
                "prediction",
                "discrepancy_type",
                "raw_response",
                "text",
                "category",
                "consensus_note",
            ]
        ]
    )

review = pd.concat(
    review_rows,
    ignore_index=True,
)

review = review.sort_values(
    [
        "symptom",
        "discrepancy_type",
        "id",
    ]
).reset_index(
    drop=True
)

review.to_csv(
    ERROR_ANALYSIS_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(
    f"Saved {len(review)} discrepancies "
    f"to {ERROR_ANALYSIS_FILE.name}."
)
