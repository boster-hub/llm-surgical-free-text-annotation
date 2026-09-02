import pandas as pd

from config import (
    POST_OUTPUT_DIR,
    RAW_OUTPUT_DIR,
)


def parse_response(value):
    normalized = (
        str(value)
        .lower()
        .replace("\r", "")
        .replace("\n", "")
        .replace("#", "")
        .replace(" ", "")
    )

    if normalized == "ja":
        return "ja", "valid_yes"

    if normalized == "nein":
        return "nein", "valid_no"

    return "nein", "invalid_defaulted_to_no"


POST_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for input_file in sorted(
    RAW_OUTPUT_DIR.glob("*_raw.csv")
):
    output_file = (
        POST_OUTPUT_DIR
        / input_file.name.replace(
            "_raw.csv",
            "_post.csv",
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

    parsed = data["raw_response"].apply(
        parse_response
    )

    data[
        ["prediction", "parsing_status"]
    ] = pd.DataFrame(
        parsed.tolist(),
        index=data.index,
    )

    data.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved {output_file.name}"
    )
