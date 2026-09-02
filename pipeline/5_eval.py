import pandas as pd

from config import (
    ANNOTATION_FILE,
    EVALUATION_DIR,
    POST_OUTPUT_DIR,
)


def divide(a, b):
    if b == 0:
        return float("nan")

    return a / b


def evaluate(data):
    reference = data["reference"].eq("ja")
    prediction = data["prediction"].eq("ja")

    tp = int((reference & prediction).sum())
    tn = int((~reference & ~prediction).sum())
    fp = int((~reference & prediction).sum())
    fn = int((reference & ~prediction).sum())

    n = tp + tn + fp + fn

    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": divide(tp + tn, n),
        "sensitivity": divide(tp, tp + fn),
        "specificity": divide(tn, tn + fp),
        "ppv": divide(tp, tp + fp),
        "npv": divide(tn, tn + fn),
        "f1": divide(
            2 * tp,
            2 * tp + fp + fn,
        ),
    }


annotations = pd.read_csv(
    ANNOTATION_FILE,
    dtype={"id": str},
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for input_file in sorted(
    POST_OUTPUT_DIR.glob("*_post.csv")
):
    output_file = (
        EVALUATION_DIR
        / input_file.name.replace(
            "_post.csv",
            "_eval.csv",
        )
    )

    if output_file.exists():
        print(
            f"Skipping existing: "
            f"{output_file.name}"
        )
        continue

    data = pd.read_csv(
        input_file,
        dtype={"id": str},
    )

    symptom = data["symptom"].iloc[0]

    data = data.merge(
        annotations[
            [
                "id",
                symptom,
            ]
        ].rename(
            columns={
                symptom: "reference"
            }
        ),
        on="id",
        validate="one_to_one",
    )

    metrics = evaluate(
        data
    )

    inference_time = (
        data["prompt_eval_time_sec"]
        + data["generation_time_sec"]
    )

    steady_time = (
        inference_time.sum()
    )

    first = data.iloc[0]

    result = {
        "analysis": first["analysis"],
        "symptom": symptom,
        "split": first["split"],
        "model": first["model"],
        "temperature": first["temperature"],
        "iteration": first["iteration"],
        **metrics,
        "invalid_responses": int(
            data["parsing_status"]
            .eq("invalid_defaulted_to_no")
            .sum()
        ),
        "invalid_rate": divide(
            data["parsing_status"]
            .eq("invalid_defaulted_to_no")
            .sum(),
            len(data),
        ),
        "prompt_tokens": int(
            data["prompt_tokens"].sum()
        ),
        "completion_tokens": int(
            data["completion_tokens"].sum()
        ),
        "steady_time_sec": steady_time,
        "time_per_document_median": (
            inference_time.median()
        ),
        "time_per_document_q1": (
            inference_time.quantile(0.25)
        ),
        "time_per_document_q3": (
            inference_time.quantile(0.75)
        ),
        "documents_per_sec": divide(
            len(data),
            steady_time,
        ),
        "warmup_load_time_sec": (
            first["warmup_load_time_sec"]
        ),
        "warmup_total_time_sec": (
            first["warmup_total_time_sec"]
        ),
    }

    pd.DataFrame(
        [result]
    ).to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved {output_file.name}"
    )
