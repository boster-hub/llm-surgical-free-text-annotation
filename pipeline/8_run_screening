import os
import subprocess
import sys

from config import (
    BASE_DIR,
    CANDIDATE_MODELS,
    SYMPTOMS,
)


for symptom in SYMPTOMS:
    for model in CANDIDATE_MODELS:
        print(
            f"\nRunning screening: "
            f"{symptom}, {model}"
        )

        environment = os.environ.copy()

        # Model screening was performed once at the fixed temperature of 0.0
        # selected during the preceding temperature analysis.
        environment.update(
            {
                "ANALYSIS_NAME": "screening",
                "ACTIVE_SYMPTOM": symptom,
                "ACTIVE_SPLIT": "develop",
                "MODEL": model,
                "TEMPERATURE": "0.0",
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
    "\nModel screening completed."
)
