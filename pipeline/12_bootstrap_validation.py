import numpy as np
import pandas as pd

from config import (
    ANNOTATION_FILE,
    BASELINE_OUTPUT_DIR,
    BOOTSTRAP_FILE,
    N_BOOTSTRAP,
    POST_OUTPUT_DIR,
    RANDOM_SEED,
    SYMPTOMS,
)


METRICS = [
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
]


def divide(a, b):
    if b == 0:
        return np.nan

    return a / b


def evaluate(reference, prediction):
    reference = reference == "ja"
    prediction = prediction == "ja"

    tp = int(
        (reference & prediction).sum()
    )

    tn = int(
        (~reference & ~prediction).sum()
    )

    fp = int(
        (~reference & prediction).sum()
    )

    fn = int(
        (reference & ~prediction).sum()
    )

    return {
        "accuracy": divide(
            tp + tn,
            tp + tn + fp + fn,
        ),
        "sensitivity": divide(
            tp,
            tp + fn,
        ),
        "specificity": divide(
            tn,
            tn + fp,
        ),
        "ppv": divide(
            tp,
            tp + fp,
        ),
        "npv": divide(
            tn,
            tn + fn,
        ),
        "f1": divide(
            2 * tp,
            2 * tp + fp + fn,
        ),
    }


annotations = pd.read_csv(
    ANNOTATION_FILE,
    dtype={"id": str},
)

llm_files = sorted(
    POST_OUTPUT_DIR.glob(
        "validation_*_validation_*_post.csv"
    )
)

if not llm_files:
    raise ValueError(
        "No LLM validation files found."
    )

llm = pd.concat(
    [
        pd.read_csv(
            file,
            dtype={"id": str},
        )
        for file in llm_files
    ],
    ignore_index=True,
)

llm = llm[
    (llm["analysis"] == "validation")
    & (llm["split"] == "validation")
].copy()

baseline_files = sorted(
    BASELINE_OUTPUT_DIR.glob(
        "baseline_*_validation.csv"
    )
)

if not baseline_files:
    raise ValueError(
        "No baseline validation files found."
    )

baseline = pd.concat(
    [
        pd.read_csv(
            file,
            dtype={"id": str},
        )
        for file in baseline_files
    ],
    ignore_index=True,
)

results = []

for symptom in SYMPTOMS:
    llm_symptom = llm[
        llm["symptom"] == symptom
    ][
        [
            "id",
            "model",
            "prediction",
        ]
    ].copy()

    models = (
        llm_symptom["model"]
        .drop_duplicates()
        .tolist()
    )

    if len(models) != 1:
        raise ValueError(
            f"Expected one selected LLM for "
            f"{symptom}; found {len(models)}."
        )

    selected_model = models[0]

    llm_symptom = llm_symptom[
        [
            "id",
            "prediction",
        ]
    ].rename(
        columns={
            "prediction": "llm_prediction"
        }
    )

    baseline_symptom = baseline[
        baseline["symptom"] == symptom
    ][
        [
            "id",
            "prediction",
        ]
    ].rename(
        columns={
            "prediction": (
                "baseline_prediction"
            )
        }
    )

    data = (
        llm_symptom
        .merge(
            baseline_symptom,
            on="id",
            validate="one_to_one",
        )
        .merge(
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
        .sort_values("id")
        .reset_index(drop=True)
    )

    if data["id"].duplicated().any():
        raise ValueError(
            f"Duplicate patient IDs found "
            f"for {symptom}."
        )

    observed_llm = evaluate(
        data["reference"],
        data["llm_prediction"],
    )

    observed_baseline = evaluate(
        data["reference"],
        data["baseline_prediction"],
    )

    bootstrap_llm = {
        metric: []
        for metric in METRICS
    }

    bootstrap_baseline = {
        metric: []
        for metric in METRICS
    }

    bootstrap_f1_difference = []

    # Resetting the generator for each symptom applies the same sampled
    # patient-index structure to all methods within that symptom.
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap_indices = rng.integers(
        0,
        len(data),
        size=(
            N_BOOTSTRAP,
            len(data),
        ),
    )

    for indices in bootstrap_indices:
        sample = (
            data.iloc[indices]
            .reset_index(drop=True)
        )

        llm_metrics = evaluate(
            sample["reference"],
            sample["llm_prediction"],
        )

        baseline_metrics = evaluate(
            sample["reference"],
            sample["baseline_prediction"],
        )

        for metric in METRICS:
            bootstrap_llm[metric].append(
                llm_metrics[metric]
            )

            bootstrap_baseline[metric].append(
                baseline_metrics[metric]
            )

        bootstrap_f1_difference.append(
            llm_metrics["f1"]
            - baseline_metrics["f1"]
        )

    for method, model, observed, bootstrap in [
        (
            "selected_llm",
            selected_model,
            observed_llm,
            bootstrap_llm,
        ),
        (
            "rule_based_baseline",
            "rule_based_baseline",
            observed_baseline,
            bootstrap_baseline,
        ),
    ]:
        for metric in METRICS:
            values = np.asarray(
                bootstrap[metric],
                dtype=float,
            )

            results.append(
                {
                    "symptom": symptom,
                    "comparison": method,
                    "model": model,
                    "metric": metric,
                    "estimate": observed[metric],
                    "ci_lower": (
                        np.nanpercentile(
                            values,
                            2.5,
                        )
                    ),
                    "ci_upper": (
                        np.nanpercentile(
                            values,
                            97.5,
                        )
                    ),
                    "bootstrap_samples": (
                        N_BOOTSTRAP
                    ),
                }
            )

    differences = np.asarray(
        bootstrap_f1_difference,
        dtype=float,
    )

    results.append(
        {
            "symptom": symptom,
            "comparison": (
                "selected_llm_minus_baseline"
            ),
            "model": selected_model,
            "metric": "f1_difference",
            "estimate": (
                observed_llm["f1"]
                - observed_baseline["f1"]
            ),
            "ci_lower": np.nanpercentile(
                differences,
                2.5,
            ),
            "ci_upper": np.nanpercentile(
                differences,
                97.5,
            ),
            "bootstrap_samples": N_BOOTSTRAP,
        }
    )

bootstrap_results = pd.DataFrame(
    results
)

bootstrap_results.to_csv(
    BOOTSTRAP_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(
    f"Saved {len(bootstrap_results)} "
    f"bootstrap results."
)
