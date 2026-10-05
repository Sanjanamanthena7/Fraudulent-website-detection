# Fraudulent Website Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning system that classifies domains as legitimate or fraudulent based on lexical, syntactic, and structural URL features.

---

## Architecture & Workflow

```text
Raw URL String
      │
      ▼
Feature Extraction (Length, IP presence, Subdomains, Hyphens, Sensitive Keywords)
      │
      ▼
Random Forest Classifier
      │
      ▼
Risk Assessment & Streamlit Dashboard Interface