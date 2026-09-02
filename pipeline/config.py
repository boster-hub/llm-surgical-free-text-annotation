import os
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

# All paths are resolved relative to the directory containing this file.

BASE_DIR = Path(__file__).resolve().parent

PDF_DIR = BASE_DIR / "PDF"
PROMPT_DIR = BASE_DIR / "prompts"
SPLIT_DIR = BASE_DIR / "splits"
RAW_OUTPUT_DIR = BASE_DIR / "raw_outputs"
POST_OUTPUT_DIR = BASE_DIR / "post_outputs"
EVALUATION_DIR = BASE_DIR / "evaluations"
BASELINE_OUTPUT_DIR = BASE_DIR / "baseline_outputs"

ANNOTATION_FILE = BASE_DIR / "annotation.csv"
TEXT_FILE = BASE_DIR / "texts.csv"
TEXT_ANALYSIS_FILE = BASE_DIR / "texts_analysis.csv"
TEXT_PROMPT_FILE = BASE_DIR / "texts_prompt.csv"
TEMPERATURE_SUMMARY_FILE = BASE_DIR / "temperature_summary.csv"
SELECTED_LLMS_FILE = BASE_DIR / "selected_llms.csv"
BOOTSTRAP_FILE = BASE_DIR / "bootstrap_results.csv"
SCREENING_SUMMARY_FILE = BASE_DIR / "screening_summary.csv"
ERROR_ANALYSIS_FILE = BASE_DIR / "error_analysis.csv"


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

# Filename pattern used to identify the source PDF reports.

PDF_PATTERN = "*_NP.pdf"

# Institution-specific markers delimiting the relevant report section.
# Set both values locally before running the PDF extraction script.

START_MARKER = os.environ.get(
    "START_MARKER",
    "SET_START_MARKER",
)

END_MARKER_PATTERN = os.environ.get(
    "END_MARKER_PATTERN",
    r"SET_END_MARKER_REGEX",
)


# =========================================================
# SYMPTOMS AND CANDIDATE MODELS
# =========================================================

SYMPTOMS = [
    "nausea",
    "vomiting",
    "diarrhea",
    "dysuria",
]

CANDIDATE_MODELS = [
    "gemma2:9b",
    "gemma2:27b",
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3.3:70b",
    "mistral-small:22b",
    "mistral-large:123b",
    "nemotron-mini:4b",
    "nemotron:70b",
]


# =========================================================
# ANALYSIS SETTINGS
# =========================================================

DEVELOPMENT_SIZE = 250

RANDOM_SEED = 42

N_BOOTSTRAP = 2000

TEMPERATURE_ITERATIONS = 10


# =========================================================
# ACTIVE LLM RUN
# =========================================================

# These runtime settings can be overridden using environment variables.
# The orchestration scripts set the required values for each experiment.

ANALYSIS_NAME = os.environ.get(
    "ANALYSIS_NAME",
    "test",
)

ACTIVE_SYMPTOM = os.environ.get(
    "ACTIVE_SYMPTOM",
    "nausea",
)

ACTIVE_SPLIT = os.environ.get(
    "ACTIVE_SPLIT",
    "develop",
)

MODEL = os.environ.get(
    "MODEL",
    "gemma2:9b",
)

TEMPERATURE = float(
    os.environ.get(
        "TEMPERATURE",
        "0.0",
    )
)

N_ITERATIONS = int(
    os.environ.get(
        "N_ITERATIONS",
        "1",
    )
)


# =========================================================
# PROMPT
# =========================================================

# System message applied unchanged to all models and symptoms.

SYSTEM_MESSAGE = (
    "Du hilfst als KI-Assistent aus medizinischen Texten "
    "Informationen zu extrahieren."
)


# =========================================================
# OLLAMA
# =========================================================

# Set OLLAMA_HOST to the URL of the local or network-accessible Ollama server.
#
# Example for a server running on the same computer:
# export OLLAMA_HOST="http://localhost:11434"
#
# Example for a server on the local network:
# export OLLAMA_HOST="http://server-address:11434"

OLLAMA_HOST = os.environ.get(
    "OLLAMA_HOST",
    "http://SET_OLLAMA_HOST:11434",
)
