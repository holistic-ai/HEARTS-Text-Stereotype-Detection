# Define helper function for loading data
import pandas as pd
from sklearn.model_selection import train_test_split

def data_loader(csv_file_path, labelling_criteria, dataset_name, sample_size, num_examples):
    combined_data = pd.read_csv(csv_file_path, usecols=['text', 'label', 'group'])

    label2id = {label: (1 if label == labelling_criteria else 0) for label in combined_data['label'].unique()}
    combined_data['label'] = combined_data['label'].map(label2id)

    combined_data['data_name'] = dataset_name

    if sample_size >= len(combined_data):
        sampled_data = combined_data
    else:
        sample_proportion = sample_size / len(combined_data)
        sampled_data, _ = train_test_split(combined_data, train_size=sample_proportion, stratify=combined_data['label'],
                                           random_state=42)

    train_data, test_data = train_test_split(sampled_data, test_size=0.2, random_state=42,
                                             stratify=sampled_data['label'])

    print("First few examples from the training data:")
    print(train_data.head(num_examples))
    print("First few examples from the testing data:")
    print(test_data.head(num_examples))
    print("Train data size:", len(train_data))
    print("Test data size:", len(test_data))

    return train_data, test_data

# Define helper function for merging data
def merge_datasets(train_data_candidate, test_data_candidate, train_data_established, test_data_established, num_examples):
    merged_train_data = pd.concat([train_data_candidate, train_data_established], ignore_index=True)
    merged_test_data = pd.concat([test_data_candidate, test_data_established], ignore_index=True)

    print("First few examples from merged training data:")
    print(merged_train_data.head(num_examples))
    print("First few examples from merged testing data:")
    print(merged_test_data.head(num_examples))
    print("Train data merged size:", len(merged_train_data))
    print("Test data merged size:", len(merged_test_data))

    return merged_train_data, merged_test_data

# Define function for evaluating the model
import os
import numpy as np
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

def evaluate_model(test_data, model_name, result_output_base_dir, dataset_name, seed):

    np.random.seed(seed)
    num_labels = len(test_data['label'].unique())
    print(f"Number of unique labels: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,
                                                               ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name.startswith("gpt"):
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)

    tokenized_test = Dataset.from_pandas(test_data).map(tokenize_function, batched=True).map(
        lambda examples: {'labels': examples['label']})
    print("Sample tokenized input from test:", tokenized_test[0])

    result_output_dir = os.path.join(result_output_base_dir, dataset_name)
    os.makedirs(result_output_dir, exist_ok=True)

    pipe = pipeline("text-classification", model= model,tokenizer=tokenizer,device=-1)

    predictions = pipe(test_data['text'].to_list(), return_all_scores=False)
    pred_labels = [1 if pred['label'] == 'BIASED' else 0 for pred in predictions]
    pred_probs = [pred['score'] for pred in predictions]
    y_true = test_data['label'].tolist()

    results_df = pd.DataFrame({
        'text': test_data['text'],
        'predicted_label': pred_labels,
        'predicted_probability': pred_probs,
        'actual_label': y_true,
        'group': test_data['group'],
        'dataset_name': test_data['data_name']
    })

    results_file_path = os.path.join(result_output_dir, "full_results.csv")
    results_df.to_csv(results_file_path, index=False)

    report = classification_report(y_true,pred_labels,output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    result_file_path = os.path.join(result_output_dir, "classification_report.csv")
    df_report.to_csv(result_file_path)

    return df_report

# Load and combine relevant datasets
train_data_winoqueer_gpt_augmentation, test_data_winoqueer_gpt_augmentation = data_loader(csv_file_path='Winoqueer - GPT Augmentation.csv', labelling_criteria='stereotype', dataset_name='Winoqueer - GPT Augmentation', sample_size=1000000, num_examples=5)
train_data_seegull_gpt_augmentation, test_data_seegull_gpt_augmentation = data_loader(csv_file_path='SeeGULL - GPT Augmentation.csv', labelling_criteria='stereotype', dataset_name='SeeGULL - GPT Augmentation', sample_size=1000000, num_examples=5)
train_data_mgsd, test_data_mgsd = data_loader(csv_file_path='MGSD.csv', labelling_criteria='stereotype', dataset_name='MGSD', sample_size=1000000, num_examples=5)
train_data_merged_winoqueer_gpt_augmentation, test_data_merged_winoqueer_gpt_augmentation = merge_datasets(train_data_candidate = train_data_winoqueer_gpt_augmentation, test_data_candidate = test_data_winoqueer_gpt_augmentation, train_data_established = train_data_mgsd, test_data_established = test_data_mgsd, num_examples=5)
train_data_merged_seegull_gpt_augmentation, test_data_merged_seegull_gpt_augmentation = merge_datasets(train_data_candidate = train_data_seegull_gpt_augmentation, test_data_candidate = test_data_seegull_gpt_augmentation, train_data_established = train_data_mgsd, test_data_established = test_data_mgsd, num_examples=5)
train_data_merged_winoqueer_seegull_gpt_augmentation, test_data_merged_winoqueer_seegull_gpt_augmentation = merge_datasets(train_data_candidate = train_data_seegull_gpt_augmentation, test_data_candidate = test_data_seegull_gpt_augmentation, train_data_established = train_data_merged_winoqueer_gpt_augmentation, test_data_established = test_data_merged_winoqueer_gpt_augmentation, num_examples=5)

# Execute full pipeline - DistilRoBERTaBias
evaluate_model(test_data_winoqueer_gpt_augmentation, model_name='valurank/distilroberta-bias', result_output_base_dir='result_output_distilrobertabias', dataset_name='winoqueer_gpt_augmentation', seed=42)
evaluate_model(test_data_seegull_gpt_augmentation, model_name='valurank/distilroberta-bias', result_output_base_dir='result_output_distilrobertabias', dataset_name='seegull_gpt_augmentation', seed=42)
evaluate_model(test_data_mgsd, model_name='valurank/distilroberta-bias', result_output_base_dir='result_output_distilrobertabias', dataset_name='mgsd', seed=42)
evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, model_name='valurank/distilroberta-bias', result_output_base_dir='result_output_distilrobertabias', dataset_name='merged_winoqueer_seegull_gpt_augmentation', seed=42)

