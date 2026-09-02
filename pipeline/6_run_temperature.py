import os
import subprocess
import sys

from config import (
    BASE_DIR,
    CANDIDATE_MODELS,
)


TEMPERATURES = [
    round(value / 10, 1)
    for value in range(11)
]


for model in CANDIDATE_MODELS:
    for temperature in TEMPERATURES:
        print(
            f"\nRunning temperature analysis: "
            f"{model}, temperature {temperature}"
        )

        environment = os.environ.copy()

        environment.update(
            {
                "ANALYSIS_NAME": "temperature",
                "ACTIVE_SYMPTOM": "nausea",
                "ACTIVE_SPLIT": "prompt",
                "MODEL": model,
                "TEMPERATURE": str(temperature),
                "N_ITERATIONS": "10",
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
    "\nTemperature analysis completed."
)
