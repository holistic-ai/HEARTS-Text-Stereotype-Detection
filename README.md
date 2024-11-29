# HEARTS-Text-Stereotype-Detection

# Overview: 
HEARTS enhances stereotype detection with explainable, low-carbon models fine-tuned on a diverse dataset, addressing LLMs' poor accuracy and the subjectivity of stereotypes. See https://arxiv.org/abs/2409.11579. 

This repository provides access to the scripts used to train and evaluate the sentence-level stereotype classification models introduced in the HEARTS research. 

# Resources: 

First run *requirements.txt*, which provides all necessary prerequisites to complete the four modules below. 

**1. Exploratory Data Analysis**

Code to perform basic Exploratory Data Analysis (EDA) on the Expanded Multi-Grain Stereotype Dataset (EMGSD). Dataset is also available at https://huggingface.co/datasets/holistic-ai/EMGSD. 

- **Initial_EDA** - analyses target group distribution, stereotype group distrubtion, text length and frequency analysis.
- **Sentiment_Regard_Analysis** - uses pre-trained models to classify Sentiment (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) and Regard (https://huggingface.co/sasha/regardv3) of each observation in dataset.

**2. Model Training and Evaluation**

Code to train and/or evaluate a series of models on the EMGSD. In each case an ablation study is performed using the three underlying components of EMGSD (MGSD, Augmented WinoQueer Dataset, Augmented SeeGULL Dataset). 

- **BERT_Models_Fine_Tuning** - trains and evaluates performance of ALBERT-V2, DistilBERT and BERT models fine-tuned on EMGSD. 
- **Logistic_Regression** - trains and evaluates performance of logistic regression models on EMGSD using (1) TF-IDF vectorisation and (2) pre-trained embeddings (see https://spacy.io/models/en).
- **DistilRoBERTaBias** - evaluates performance of open-source model for general bias detection (from https://huggingface.co/valurank/distilroberta-bias).
- **GPT4_Models** - evaluates performance of GPT-4o and GPT-4o-mini with prompting through API, requires own credentials to execute.

**3. Model Explainability**

Code to perform SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-Agnostic Explanations) analysis on predictions of the ALBERT-V2 model fine-tuned on EMGSD. The weights of this model are also available at https://huggingface.co/holistic-ai/bias_classifier_albertv2. 

- **SHAP_LIME_Analysis** - calculates SHAP and LIME vectors for chosen sample of model predictions, then compares vectors using set of similarity metrics (cosine similarity, Pearson correlation, Jensen-Shannon divergence) to provide degree of confidence in the explanations provided by the SHAP and LIME methods.

**4. LLM Bias Evaluation Exercise**

Code using fine-tuned ALBERT-V2 model to classify LLM responses, generated through APIs, when given neutral prompts constructed by stemming sentences from the EMGSD. 

- **LLM_Prompt_Verification** - uses fine-tuned ALBERT-V2 model to verify that all prompts fed into the LLMs tested are neutral.
- **LLM_Bias_Evaluation** - uses fine-tuned ALBERT-V2 model to classify LLM outputs to these prompts, to calculate aggregate bias scores for each model, representing stereotype prevalance.
- **SHAP_LIME_Analysis_LLM_Outputs** - applies same explainability framework as Module 3 to provide SHAP and LIME analysis of predictions made by the fine-tuned ALBERT-V2 models on the LLM outputs. 


