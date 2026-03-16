# llm-surgical-free-text-annotation
Reproducible benchmarking pipeline for LLM-based annotation of surgical clinical free text.

# Project Overview
Machine learning applications in surgical data science depend on large, well-annotated datasets. However, clinically relevant information is often documented as unstructured free text in electronic health records.
This project evaluates whether locally hosted open-source large language models can reliably annotate clinical free text and thereby support scalable dataset generation for downstream machine learning applications.
The repository provides a reproducible pipeline for:
extracting text from clinical PDF reports
preprocessing clinical free text
prompting locally hosted LLMs
benchmarking annotation performance against expert labels
computing performance metrics including accuracy, sensitivity, specificity, and F1-score

# Pipeline
The analysis workflow consists of three main stages:
1. Data extraction and preprocessing
Emergency department reports are extracted from electronic health records (EHR) as PDF files and converted to plain text using R.
Preprocessing includes:
text extraction from PDF files
normalization of character encoding
segmentation of relevant text sections
2. Model and prompt evaluation
A subset of reports is used to evaluate multiple open-source LLMs with different parameter sizes and temperature settings.
Models evaluated include:
Gemma2 (Google)
Llama 3.1 / 3.3 (Meta)
Mistral (Mistral AI)
Nemotron (NVIDIA)
Each configuration is evaluated across multiple iterations using zero-shot prompting.
3. Final benchmarking
The best-performing model and prompt configuration are applied to the remaining dataset. Predictions are compared against expert annotations using:
confusion matrices
accuracy
sensitivity / specificity
precision
F1-score
processing time per case

# Software Environment
The pipeline was implemented in R using the following key packages:
pdftools
dplyr
stringr
stringi
rollama
LLMs are executed locally using the Ollama framework.

# Reproducibility
The repository contains the full R pipeline required to reproduce the annotation and evaluation workflow.
Due to privacy regulations, the original clinical data cannot be shared. However, the code can be applied to any comparable clinical text dataset.

# Citation
If you use this code or framework in your research, please cite the corresponding study:
Henn J. et al.
Large Language Models Enhance Data Availability for Surgical Data Science: Systematic Assessment of Open-Source Models for Free-Text Annotation.

