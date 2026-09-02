import numpy as np
import pandas as pd

from config import (
    ANNOTATION_FILE,
    CANDIDATE_MODELS,
    DEVELOPMENT_SIZE,
    EVALUATION_DIR,
    N_BOOTSTRAP,
    POST_OUTPUT_DIR,
    RANDOM_SEED,
    SCREENING_SUMMARY_FILE,
    SELECTED_LLMS_FILE,
    SYMPTOMS,
)


def calculate_f1(reference, prediction):
    reference = reference == "ja"
    prediction = prediction == "ja"

    tp = int((reference & prediction).sum())
    fp = int((~reference & prediction).sum())
    fn = int((reference & ~prediction).sum())

    denominator = 2 * tp + fp + fn

    if denominator == 0:
        return np.nan

    return 2 * tp / denominator


evaluation_files = sorted(
    EVALUATION_DIR.glob(
        "screening_*_develop_*_eval.csv"
    )
)

post_files = sorted(
    POST_OUTPUT_DIR.glob(
        "screening_*_develop_*_post.csv"
    )
)

if not evaluation_files:
    raise ValueError(
        "No screening evaluation files found."
    )

if not post_files:
    raise ValueError(
        "No screening prediction files found."
    )

screening = pd.concat(
    [
        pd.read_csv(file)
        for file in evaluation_files
    ],
    ignore_index=True,
)

predictions = pd.concat(
    [
        pd.read_csv(
            file,
            dtype={"id": str},
        )
        for file in post_files
    ],
    ignore_index=True,
)

annotations = pd.read_csv(
    ANNOTATION_FILE,
    dtype={"id": str},
)

screening = screening[
    (screening["analysis"] == "screening")
    & (screening["split"] == "develop")
].copy()

predictions = predictions[
    (predictions["analysis"] == "screening")
    & (predictions["split"] == "develop")
].copy()

if screening[
    ["symptom", "model"]
].duplicated().any():
    raise ValueError(
        "Duplicate screening results found."
    )

selected = []
summary_rows = []

for symptom in SYMPTOMS:
    results = screening[
        screening["symptom"] == symptom
    ].copy()

    if len(results) != len(CANDIDATE_MODELS):
        raise ValueError(
            f"Expected {len(CANDIDATE_MODELS)} models "
            f"for {symptom}; found {len(results)}."
        )

    symptom_predictions = predictions[
        predictions["symptom"] == symptom
    ][
        [
            "id",
            "model",
            "prediction",
        ]
    ]

    prediction_matrix = (
        symptom_predictions
        .pivot(
            index="id",
            columns="model",
            values="prediction",
        )
        .sort_index()
    )

    expected_shape = (
        DEVELOPMENT_SIZE,
        len(CANDIDATE_MODELS),
    )

    if prediction_matrix.shape != expected_shape:
        raise ValueError(
            f"Expected {DEVELOPMENT_SIZE} patients and "
            f"{len(CANDIDATE_MODELS)} models for {symptom}; "
            f"found {prediction_matrix.shape}."
        )

    reference = (
        annotations
        .set_index("id")
        .loc[
            prediction_matrix.index,
            symptom,
        ]
        .to_numpy()
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap_f1 = {
        model: []
        for model in prediction_matrix.columns
    }

    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(
            0,
            len(prediction_matrix),
            size=len(prediction_matrix),
        )

        sampled_reference = reference[
            indices
        ]

        for model in prediction_matrix.columns:
            sampled_prediction = (
                prediction_matrix[model]
                .to_numpy()[indices]
            )

            bootstrap_f1[model].append(
                calculate_f1(
                    sampled_reference,
                    sampled_prediction,
                )
            )

    ci_lower = {}
    ci_upper = {}

    for model, values in bootstrap_f1.items():
        values = np.asarray(
            values,
            dtype=float,
        )

        ci_lower[model] = np.nanpercentile(
            values,
            2.5,
        )

        ci_upper[model] = np.nanpercentile(
            values,
            97.5,
        )

    results["f1_ci_lower"] = (
        results["model"].map(ci_lower)
    )

    results["f1_ci_upper"] = (
        results["model"].map(ci_upper)
    )

    # Models are selected by decreasing observed F1-score.
    # Steady-state inference time is used only as a tiebreaker.
    results = results.sort_values(
        [
            "f1",
            "steady_time_sec",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    results.insert(
        0,
        "rank",
        range(
            1,
            len(results) + 1,
        ),
    )

    summary_rows.append(
        results
    )

    ranking = results[
        [
            "rank",
            "model",
            "f1",
            "f1_ci_lower",
            "f1_ci_upper",
            "steady_time_sec",
        ]
    ]

    print(
        f"\n{symptom.upper()}"
    )

    print(
        ranking.to_string(
            index=False,
            formatters={
                "f1": lambda value: (
                    f"{value:.3f}"
                ),
                "f1_ci_lower": lambda value: (
                    f"{value:.3f}"
                ),
                "f1_ci_upper": lambda value: (
                    f"{value:.3f}"
                ),
                "steady_time_sec": lambda value: (
                    f"{value:.1f}"
                ),
            },
        )
    )

    best = results.iloc[0]

    selected.append(
        {
            "symptom": symptom,
            "model": best["model"],
            "temperature": best["temperature"],
            "f1_development": best["f1"],
            "f1_ci_lower_development": (
                best["f1_ci_lower"]
            ),
            "f1_ci_upper_development": (
                best["f1_ci_upper"]
            ),
            "steady_time_sec_development": (
                best["steady_time_sec"]
            ),
        }
    )

screening_summary = pd.concat(
    summary_rows,
    ignore_index=True,
)

screening_summary.to_csv(
    SCREENING_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(
    selected
).to_csv(
    SELECTED_LLMS_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(
    f"\nSaved {SCREENING_SUMMARY_FILE.name}."
)

print(
    f"Saved {SELECTED_LLMS_FILE.name}."
)
