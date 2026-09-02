import pandas as pd
from ollama import Client

from config import (
    ACTIVE_SPLIT,
    ACTIVE_SYMPTOM,
    ANALYSIS_NAME,
    MODEL,
    N_ITERATIONS,
    OLLAMA_HOST,
    PROMPT_DIR,
    RAW_OUTPUT_DIR,
    SPLIT_DIR,
    SYSTEM_MESSAGE,
    TEMPERATURE,
    TEXT_ANALYSIS_FILE,
    TEXT_PROMPT_FILE,
)


client = Client(
    host=OLLAMA_HOST
)


def ns_to_seconds(value):
    if value is None:
        return None

    return value / 1_000_000_000


# Load data

if ACTIVE_SPLIT == "prompt":
    data = pd.read_csv(
        TEXT_PROMPT_FILE,
        dtype={"id": str},
    )

    data["split"] = "prompt"

else:
    texts = pd.read_csv(
        TEXT_ANALYSIS_FILE,
        dtype={"id": str},
    )

    split = pd.read_csv(
        SPLIT_DIR / f"split_{ACTIVE_SYMPTOM}.csv",
        dtype={"id": str},
    )

    data = texts.merge(
        split,
        on="id",
        validate="one_to_one",
    )

    data = data[
        data["split"] == ACTIVE_SPLIT
    ].copy()


data = data.sort_values(
    "id"
).reset_index(
    drop=True
)


# Load prompt

prompt = (
    PROMPT_DIR
    / f"{ACTIVE_SYMPTOM}.txt"
).read_text(
    encoding="utf-8"
).strip()


# Output

RAW_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

model_safe = (
    MODEL
    .replace(":", "_")
    .replace(".", "p")
)

temperature_safe = (
    str(TEMPERATURE)
    .replace(".", "p")
)


# Run iterations

for iteration in range(
    1,
    N_ITERATIONS + 1,
):
    output_file = (
        RAW_OUTPUT_DIR
        / (
            f"{ANALYSIS_NAME}_"
            f"{ACTIVE_SYMPTOM}_"
            f"{model_safe}_"
            f"temp{temperature_safe}_"
            f"{ACTIVE_SPLIT}_"
            f"rep{iteration:02d}_raw.csv"
        )
    )

    if output_file.exists():
        print(
            f"Skipping existing: "
            f"{output_file.name}"
        )

        continue

    print(
        f"Running {ANALYSIS_NAME}, "
        f"{ACTIVE_SYMPTOM}, "
        f"{MODEL}, "
        f"temperature {TEMPERATURE}, "
        f"iteration {iteration}"
    )

    # Load model before measured inference

    warmup = client.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Antworte nur mit OK.",
            }
        ],
        options={
            "temperature": TEMPERATURE,
        },
        stream=False,
        think=False,
        keep_alive="30m",
    )

    warmup_load_time_sec = ns_to_seconds(
        warmup.load_duration
    )

    warmup_total_time_sec = ns_to_seconds(
        warmup.total_duration
    )

    results = []

    for row in data.itertuples(
        index=False
    ):
        user_message = (
            f"text: {row.text}\n"
            f"{prompt}"
        )

        response = client.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            options={
                "temperature": TEMPERATURE,
            },
            stream=False,
            think=False,
            keep_alive="30m",
        )

        results.append(
            {
                "id": row.id,
                "analysis": ANALYSIS_NAME,
                "symptom": ACTIVE_SYMPTOM,
                "split": ACTIVE_SPLIT,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "iteration": iteration,
                "raw_response": (
                    response.message.content
                ),
                "prompt_tokens": (
                    response.prompt_eval_count
                ),
                "completion_tokens": (
                    response.eval_count
                ),
                "prompt_eval_time_sec": (
                    ns_to_seconds(
                        response.prompt_eval_duration
                    )
                ),
                "generation_time_sec": (
                    ns_to_seconds(
                        response.eval_duration
                    )
                ),
                "total_time_sec": (
                    ns_to_seconds(
                        response.total_duration
                    )
                ),
                "load_time_sec": (
                    ns_to_seconds(
                        response.load_duration
                    )
                ),
                "warmup_load_time_sec": (
                    warmup_load_time_sec
                ),
                "warmup_total_time_sec": (
                    warmup_total_time_sec
                ),
            }
        )

    pd.DataFrame(
        results
    ).to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved {output_file.name}"
    )
