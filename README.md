# From Clinical Free Text to Structured Data

This repository contains the reproducible Python pipeline developed for symptom annotation in German emergency department reports using locally deployed open-weight large language models.

The workflow includes:

- extraction of text from computer-generated PDF reports;
- symptom-specific stratified data splitting;
- local LLM inference using Ollama;
- deterministic output parsing;
- model screening and selection;
- evaluation against a negation-aware rule-based baseline;
- patient-level bootstrap analysis; and
- reproducible generation of summary results.

## Legacy code

The original R-based workflow and its associated prompts have been superseded and are retained only for historical transparency:

[`legacy/`](legacy/)

The legacy workflow should not be used to reproduce the results reported in the current manuscript revision.

## License

See [`LICENSE`](LICENSE).
