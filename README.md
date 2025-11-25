# Mental Health Status Classification using NLP

An end-to-end NLP project that classifies social media comments into 7 mental health categories (e.g., Anxiety, Depression, Stress). This project demonstrates the evolution from classical Machine Learning models to state-of-the-art Deep Learning (BERT), addressing challenges like class imbalance and semantic context.

![Accuracy](https://img.shields.io/badge/Accuracy-83%25-green) ![Python](https://img.shields.io/badge/Python-3.10-blue) ![Library](https://img.shields.io/badge/Library-HuggingFace-orange)

---

## Project Overview

Mental health analysis in social media is complex due to informal language, sarcasm, and subtle emotional nuances. This project tackles a dataset of **53,000+ real comments** to build a robust classification pipeline.

**Key Objectives:**
* Process and clean unstructured text data.
* Address severe class imbalance (e.g., *Normal* vs *Stress*).
* Compare the performance of classical NLP approaches vs. Transfer Learning.

---

## Methodology & Tech Stack

The project implements a two-phase approach to solve the classification problem:

### Phase 1: Classical NLP & Feature Engineering
* **Preprocessing:** Text cleaning and lemmatization using `spaCy` (`en_core_web_lg`).
* **Vectorization:** Generating document embeddings using `Gensim` (FastText).
* **Modeling:** Evaluating baseline models (Random Forest, Logistic Regression).
* **Optimization:** Implementing `LightGBM` with `class_weight='balanced'` to improve recall on minority classes.

### Phase 2: Deep Learning (The Solution)
* **Architecture:** Fine-tuning **BERT (bert-base-cased)** using the **Hugging Face Transformers** library.
* **Why BERT?** To capture the bidirectional context of words, crucial for distinguishing between similar sentiments like *Anxiety* and *Stress*.

---

## Results Summary

The project showed a clear progression in performance across the tested architectures:

| Model Strategy | Accuracy | Key Finding |
| :--- | :--- | :--- |
| **Random Forest (Baseline)** | 66% | Failed to detect minority classes (Recall for *Stress* was only 0.20). |
| **LightGBM (Optimized)** | **71%** | `class_weight='balanced'` improved minority recall significantly. |
| **BERT (Fine-Tuned)** | **83%** | **Best Performer.** Successfully captured semantic context, achieving high precision and recall across all 7 classes. |

> **Key Insight:** While LightGBM improved the detection of imbalanced classes through weighting, it lacked semantic understanding. **BERT** bridged this gap, raising the F1-Score for difficult classes like *Stress* from 0.51 to **0.76**.

---

## How to Run this Project

This project is designed to run on **Google Colab** (specifically for the GPU requirements of BERT).

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Pavel-Aguilar/Mental-Health-NLP.git](https://github.com/Pavel-Aguilar/Mental-Health-NLP.git)