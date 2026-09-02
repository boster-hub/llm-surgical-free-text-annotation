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

ANALYSIS_NAME = os.environ.get(
    "ANALYSIS_NAME",
    "test",
)


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

# Filename pattern used to identify the source PDF reports.

PDF_PATTERN = "*_NP.pdf"

# Institution-specific markers delimiting the relevant report section.
# Set these variables locally to the exact literal start marker and regular
# expression used for the end marker. 

START_MARKER = os.environ.get(
    "START_MARKER",
    "<SET_START_MARKER>",
)

END_MARKER_PATTERN = os.environ.get(
    "END_MARKER_PATTERN",
    r"<SET_END_MARKER_REGEX>",
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
# DATA SPLIT
# =========================================================

DEVELOPMENT_SIZE = 250

RANDOM_SEED = 42

TEMPERATURE_ITERATIONS = 10

# =========================================================
# ACTIVE LLM RUN
# =========================================================

# Runtime settings can be overridden using environment variables.
# The defaults reproduce the model-run configuration used in the study unless
# a different value is explicitly supplied for a specific pipeline run.

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
    "Du hilfst als KI-Assistent aus medizinischen Texten Informationen zu extrahieren."
)


# =========================================================
# OLLAMA
# =========================================================

# URL of the Ollama server used for local or network-based inference.
# Set OLLAMA_HOST locally before running the inference pipeline.
#
# Example for an Ollama server running on the same computer:
# export OLLAMA_HOST="http://localhost:11434"
#
# Example for a server on the local network:
# export OLLAMA_HOST="http://<server-address>:11434"

OLLAMA_HOST = os.environ.get(
    "OLLAMA_HOST",
    "http://<SET_OLLAMA_HOST>:11434",
)
