from transformers import pipeline
import numpy as np
import pandas as pd
import torch
import shap
from lime.lime_text import LimeTextExplainer
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon

file_path = 'full_results_llm_exercise.csv'
model_path = "holistic-ai/bias_classifier_albertv2"

sampled_data = pd.read_csv(file_path)

# Define function to compute SHAP values
def shap_analysis(sampled_data, model_path):
    pipe = pipeline("text-classification", model=model_path, return_all_scores=True)
    masker = shap.maskers.Text(tokenizer=r'\b\w+\b')  
    explainer = shap.Explainer(pipe, masker)

    results = []
    class_names = ['LABEL_0', 'LABEL_1']
    
    for index, row in sampled_data.iterrows():
        text_input = [row['text']]
        shap_values = explainer(text_input)
        
        print(f"Group: {row['group']} - Predicted Label: {row['predicted_label']} - Model: {row['model']}")
        label_index = class_names.index("LABEL_1")  
        
        specific_shap_values = shap_values[:, :, label_index].values
        
        tokens = re.findall(r'\w+', row['text'])
        for token, value in zip(tokens, specific_shap_values[0]):
            results.append({
                'sentence_id': index,
                'token': token,
                'value': value,
                'sentence': row['text'],
                'group': row['group'],
                'predicted_label': row['predicted_label'],
                'model': row['model']
            })
                
    return pd.DataFrame(results)


shap_results = shap_analysis(sampled_data, model_path)
print(shap_results)

# Define function to compute LIME values 
def custom_tokenizer(text):
    tokens = re.split(r'\W+', text)
    tokens = [token for token in tokens if token]
    return tokens

def lime_analysis(sampled_data, model_path):
    pipe = pipeline("text-classification", model=model_path, return_all_scores=True)
    
    def predict_proba(texts):
        preds = pipe(texts, return_all_scores=True)
        probabilities = np.array([[pred['score'] for pred in preds_single] for preds_single in preds])
        print("Probabilities shape:", probabilities.shape)
        return probabilities    
    
    explainer = LimeTextExplainer(class_names=['LABEL_0', 'LABEL_1'], split_expression=lambda x: custom_tokenizer(x))  
    
    results = []
    
    for index, row in sampled_data.iterrows():
        text_input = row['text']
        tokens = custom_tokenizer(text_input)
        exp = explainer.explain_instance(text_input, predict_proba, num_features=len(tokens), num_samples=100)
        
        print(f"Group: {row['group']} - Predicted Label: {row['predicted_label']} - Model: {row['model']}")

        explanation_list = exp.as_list(label=1)
        
        token_value_dict = {token: value for token, value in explanation_list}

        for token in tokens:
            value = token_value_dict.get(token, 0)  
            results.append({
                'sentence_id': index,
                'token': token,
                'value': value,
                'sentence': text_input,
                'group': row['group'],
                'predicted_label': row['predicted_label'],
                'model': row['model']
            })

    return pd.DataFrame(results)

lime_results = lime_analysis(sampled_data, model_path)
print(lime_results)

lime_results.to_csv('lime_results.csv')
shap_results.to_csv('shap_results.csv')

# Compute similarity scores
shap_df = pd.read_csv('shap_results.csv')
lime_df = pd.read_csv('lime_results.csv')

common_columns = [col for col in shap_df.columns if col != 'value']

merged_df = pd.merge(shap_df, lime_df, on=common_columns, suffixes=('_shap', '_lime'))

print(merged_df.head())

grouped = merged_df.groupby('sentence_id').agg({
    'value_shap': list,
    'value_lime': list
})

def compute_cosine_similarity(row):
    vector_shap = np.array(row['value_shap']).reshape(1, -1)
    vector_lime = np.array(row['value_lime']).reshape(1, -1)
    return cosine_similarity(vector_shap, vector_lime)[0][0]

def compute_pearson_correlation(row):
    vector_shap = np.array(row['value_shap'])
    vector_lime = np.array(row['value_lime'])
    correlation, _ = pearsonr(vector_shap, vector_lime)
    return correlation

def to_probability_distribution(values):
    min_val = np.min(values)
    if min_val < 0:
        values += abs(min_val)
    total = np.sum(values)
    if total > 0:
        values /= total
    return values

def compute_js_divergence(row):
    vector_shap = np.array(row['value_shap'])
    vector_lime = np.array(row['value_lime'])
    prob_shap = to_probability_distribution(vector_shap.copy())
    prob_lime = to_probability_distribution(vector_lime.copy())
    return jensenshannon(prob_shap, prob_lime)   

cosine_similarities = grouped.apply(compute_cosine_similarity, axis=1)
merged_df['cosine_similarity'] = merged_df['sentence_id'].map(cosine_similarities)

pearson_correlations = grouped.apply(compute_pearson_correlation, axis=1)
merged_df['pearson_correlation'] = merged_df['sentence_id'].map(pearson_correlations)

js_divergences = grouped.apply(compute_js_divergence, axis=1)
merged_df['js_divergence'] = merged_df['sentence_id'].map(js_divergences)

merged_df.to_csv('similarity_results.csv')
