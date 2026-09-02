import pandas as pd

from config import (
    EVALUATION_DIR,
    TEMPERATURE_ITERATIONS,
    TEMPERATURE_SUMMARY_FILE,
)


METRICS = [
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "invalid_rate",
    "steady_time_sec",
    "documents_per_sec",
]


files = sorted(
    EVALUATION_DIR.glob(
        "temperature_*_prompt_*_eval.csv"
    )
)

if not files:
    raise ValueError(
        "No prompt-set temperature "
        "evaluation files found."
    )

evaluation = pd.concat(
    [
        pd.read_csv(file)
        for file in files
    ],
    ignore_index=True,
)

evaluation = evaluation[
    (evaluation["analysis"] == "temperature")
    & (evaluation["symptom"] == "nausea")
    & (evaluation["split"] == "prompt")
].copy()

rows = []

for (
    model,
    temperature,
), group in evaluation.groupby(
    [
        "model",
        "temperature",
    ]
):
    if len(group) != TEMPERATURE_ITERATIONS:
        raise ValueError(
            f"Expected {TEMPERATURE_ITERATIONS} iterations for "
            f"{model}, temperature {temperature}; "
            f"found {len(group)}."
        )

    row = {
        "model": model,
        "temperature": temperature,
        "iterations": len(group),
    }

    for metric in METRICS:
        row[f"{metric}_median"] = (
            group[metric].median()
        )

        row[f"{metric}_q1"] = (
            group[metric].quantile(0.25)
        )

        row[f"{metric}_q3"] = (
            group[metric].quantile(0.75)
        )

        row[f"{metric}_iqr"] = (
            row[f"{metric}_q3"]
            - row[f"{metric}_q1"]
        )

    rows.append(row)

summary = pd.DataFrame(
    rows
).sort_values(
    [
        "model",
        "temperature",
    ]
)

summary.to_csv(
    TEMPERATURE_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(
    f"Saved {len(summary)} "
    f"temperature summary rows."
)
