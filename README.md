# ⚡ Cardiometabolic Entity Linkage & Health Retrieval Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/Database-DuckDB-yellow.svg)](https://duckdb.org/)
[![Splink](https://img.shields.io/badge/Linkage-Splink_v3-orange.svg)](https://github.com/moj-analytical-services/splink)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, high-performance data engineering and NLP system that demonstrates **probabilistic record linkage**, **hybrid string matching**, and **Retrieval-Augmented Generation (RAG)** across 5 cycles of CDC NHANES population survey data (2009–2018) combined with clinical guidelines.

---

## 📌 Business & Technical Problem

In healthcare and clinical data science, electronic health records (EHR) frequently suffer from data fragmentation—patients appear across multiple clinic visits, diagnostic labs, and cycles under slightly different names, mistyped dates of birth, or varied clinical measurements. 

Without a universal patient identifier, traditional exact matching SQL queries fail. This repository implements an automated, scalable pipeline that solves two key healthcare challenges:
1. **Entity Resolution & Deduplication:** Probabilistically linking multi-year patient records across dirty clinical datasets without unique identifiers.
2. **Context-Aware Clinical QA:** Extracting clinical entities (lab thresholds, medication names) from unstructured medical literature (e.g., ADA Diabetes Guidelines) to power a grounded RAG engine for cohort risk assessment.

---

## 🏗️ System Architecture
