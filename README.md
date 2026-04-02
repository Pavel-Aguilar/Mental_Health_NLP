# 🧠 Mental Health Status Classification using NLP

![Accuracy](https://img.shields.io/badge/Accuracy-83%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-9cf)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end NLP pipeline that classifies social media comments into **7 mental health categories** (Anxiety, Depression, Stress, Bipolar, PTSD, Personality Disorder, Normal). This project benchmarks classical ML against state-of-the-art Deep Learning (BERT), tackling real challenges like class imbalance and semantic ambiguity.

---

## 🎯 Problem Statement

Mental health analysis in social media is complex due to informal language, sarcasm, and subtle emotional nuances. This project processes **53,000+ real comments** to build a robust multi-class classification pipeline.

**Key Objectives:**
- Process and clean unstructured text data at scale
- Address severe class imbalance (e.g., *Normal* vs *Stress*)
- Benchmark classical NLP approaches against Transfer Learning (BERT)

---

## 🔬 Methodology

### Phase 1 — Classical NLP & Feature Engineering
| Step | Detail |
|---|---|
| Preprocessing | Text cleaning + lemmatization via `spaCy` (`en_core_web_lg`) |
| Vectorization | Document embeddings with `Gensim` FastText |
| Baseline Models | Random Forest, Logistic Regression |
| Optimized Model | LightGBM with `class_weight='balanced'` |

### Phase 2 — Deep Learning (BERT)
| Step | Detail |
|---|---|
| Architecture | Fine-tuned `bert-base-cased` |
| Library | HuggingFace `Transformers` |
| Why BERT? | Bidirectional context captures subtle differences between *Anxiety* and *Stress* |

---

## 📊 Results

| Model | Accuracy | Key Finding |
|:---|:---:|:---|
| Random Forest (Baseline) | 66% | Failed on minority classes — Recall for *Stress*: 0.20 |
| LightGBM (Optimized) | 71% | `class_weight='balanced'` improved minority recall significantly |
| **BERT (Fine-Tuned)** | **83%** | Best performer — F1 for *Stress* jumped from 0.51 → **0.76** |

> BERT bridged the semantic gap that gradient boosting couldn't — understanding context, not just word frequency.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Classical ML | Scikit-learn, LightGBM |
| NLP Preprocessing | spaCy, Gensim (FastText) |
| Deep Learning | HuggingFace Transformers (BERT) |
| Environment | Google Colab (GPU) |

---

## 🚀 How to Run

This project is designed for **Google Colab** (GPU required for BERT fine-tuning).

```bash
git clone https://github.com/Pavel-Aguilar/Mental_Health_NLP.git
cd Mental_Health_NLP
```

Open `mental_health_pln_(bert).py` in Colab and enable GPU runtime:
`Runtime → Change runtime type → T4 GPU`

**Install dependencies:**
```bash
pip install transformers torch scikit-learn lightgbm spacy gensim
python -m spacy download en_core_web_lg
```

---

## 📁 Project Structure

```
Mental_Health_NLP/
├── mental_health_pln_(bert).py   # Full pipeline: preprocessing → BERT fine-tuning
├── .gitignore
└── README.md
```

---

## 👤 About

Built as part of the Master's in AI & Data Analytics at UACJ. Focused on applying NLP to real-world mental health data with a rigorous ML benchmarking approach.

[![GitHub](https://img.shields.io/badge/GitHub-Pavel--Aguilar-181717?logo=github)](https://github.com/Pavel-Aguilar)

---

## 📄 License

[MIT License](LICENSE)
