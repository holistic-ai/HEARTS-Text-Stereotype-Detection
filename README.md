
# HEARTS-Text-Stereotype-Detection

[![Paper](https://img.shields.io/badge/ArXiv-2409.11579-red)](https://arxiv.org/abs/2409.11579)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

HEARTS introduces explainable, low-carbon models fine-tuned on the **Expanded Multi-Grain Stereotype Dataset (EMGSD)** to tackle challenges in stereotype detection. This repository includes scripts for training, evaluation, and explainability analysis for sentence-level stereotype classification. For details, refer to the [HEARTS research paper](https://arxiv.org/abs/2409.11579).

---

## Features

- **Exploratory Data Analysis (EDA):** Analyze group distributions, text length, and sentiment/regard trends in EMGSD.
- **Model Training & Evaluation:** Train and test models (e.g., BERT, ALBERT-V2, logistic regression) on EMGSD with ablation studies.
- **Explainability:** Generate SHAP and LIME explanations to interpret predictions.
- **LLM Bias Evaluation:** Classify and evaluate bias in LLM outputs using neutral prompts derived from EMGSD.

---

## Quickstart

1. Clone this repository:
   ```bash
   git clone https://github.com/username/HEARTS-Text-Stereotype-Detection.git
   cd HEARTS-Text-Stereotype-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Explore the modules (see details below).

---

## Modules

### 1. Exploratory Data Analysis
Scripts to perform basic analysis on EMGSD, available at [Hugging Face](https://huggingface.co/datasets/holistic-ai/EMGSD).

- **`Initial_EDA`**: Analyze target group distribution, stereotype group distribution, text length, and frequency.
- **`Sentiment_Regard_Analysis`**: Classify sentiment ([RoBERTa Sentiment Model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)) and regard ([Regard v3](https://huggingface.co/sasha/regardv3)) for dataset entries.

### 2. Model Training and Evaluation
Train and evaluate various models on EMGSD, with ablation studies on its three core datasets (MGSD, Augmented WinoQueer, and Augmented SeeGULL).

- **`BERT_Models_Fine_Tuning`**: Fine-tune and evaluate ALBERT-V2, DistilBERT, and BERT models.
- **`Logistic_Regression`**: Train logistic regression models using:
  - TF-IDF vectorization
  - Pre-trained embeddings ([spaCy embeddings](https://spacy.io/models/en))
- **`DistilRoBERTaBias`**: Evaluate an open-source bias detection model ([DistilRoBERTa Bias](https://huggingface.co/valurank/distilroberta-bias)).
- **`GPT4_Models`**: Evaluate GPT-4o and GPT-4o-mini using API prompting (API credentials required).

### 3. Model Explainability
Interpret model predictions using SHAP and LIME. Weights for the fine-tuned ALBERT-V2 model are available at [Hugging Face](https://huggingface.co/holistic-ai/bias_classifier_albertv2).

- **`SHAP_LIME_Analysis`**: Generate SHAP and LIME explanations for selected model predictions and compare their similarity using metrics such as:
  - Cosine similarity
  - Pearson correlation
  - Jensen-Shannon divergence

### 4. LLM Bias Evaluation
Classify and evaluate bias in LLM responses using neutral prompts derived from EMGSD.

- **`LLM_Prompt_Verification`**: Verify neutrality of prompts using the fine-tuned ALBERT-V2 model.
- **`LLM_Bias_Evaluation`**: Classify LLM outputs to compute aggregate bias scores, representing stereotype prevalence.
- **`SHAP_LIME_Analysis_LLM_Outputs`**: Apply SHAP and LIME to interpret predictions on LLM outputs.

---

## Results

Key findings and performance benchmarks from the paper are outlined [here](https://arxiv.org/abs/2409.11579).

---

## Citation

If you use this work, please cite the following paper:

```
@article{hearts2024,
  title={HEARTS: Enhancing Stereotype Detection with Explainable, Low-Carbon Models},
  author={Author Names},
  journal={arXiv preprint arXiv:2409.11579},
  year={2024}
}
```

---

## License

This repository is licensed under the [MIT License](LICENSE).

---

## Contact

For questions or collaborations, contact [Holistic AI](https://www.holisticai.com).
