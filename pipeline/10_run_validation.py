import os
import subprocess
import sys

import pandas as pd

from config import (
    BASE_DIR,
    SELECTED_LLMS_FILE,
)


selected = pd.read_csv(
    SELECTED_LLMS_FILE
)

for row in selected.itertuples(
    index=False
):
    print(
        f"\nRunning validation: "
        f"{row.symptom}, {row.model}"
    )

    environment = os.environ.copy()

    environment.update(
        {
            "ANALYSIS_NAME": "validation",
            "ACTIVE_SYMPTOM": row.symptom,
            "ACTIVE_SPLIT": "validation",
            "MODEL": row.model,
            "TEMPERATURE": str(row.temperature),
            "N_ITERATIONS": "1",
        }
    )

    subprocess.run(
        [
            sys.executable,
            BASE_DIR / "3_llmextract.py",
        ],
        cwd=BASE_DIR,
        env=environment,
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            BASE_DIR / "4_post.py",
        ],
        cwd=BASE_DIR,
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            BASE_DIR / "5_eval.py",
        ],
        cwd=BASE_DIR,
        check=True,
    )

print(
    "\nLLM validation completed."
)
