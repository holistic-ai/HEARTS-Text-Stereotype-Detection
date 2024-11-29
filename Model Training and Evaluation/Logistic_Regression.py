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

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import spacy
import spacy.cli
import joblib
from codecarbon import EmissionsTracker
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm import tqdm

# Define function for training the model
def train_model(train_data, model_output_base_dir, dataset_name, feature_type, seed):
    np.random.seed(seed)
    num_labels = len(np.unique(train_data['label']))
    print(f"Number of unique labels: {num_labels}")

    tracker = EmissionsTracker()
    tracker.start()

    nlp = spacy.load("en_core_web_lg")
    vectorizer = None

    if feature_type == 'tfidf':
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(train_data['text'])
    elif feature_type == 'embedding':
        def get_embedding(text):
            doc = nlp(text)
            return doc.vector

        X = np.array([get_embedding(text) for text in tqdm(train_data['text'], desc="Computing embeddings")])

    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

    y = train_data['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    C_values = [0.01, 0.1, 1]
    penalties = ['l1', 'l2', None]
    best_f1_score = 0
    best_model = None
    best_params = {}

    for C in C_values:
        for penalty in penalties:
            model = LogisticRegression(C=C, penalty=penalty, solver='saga', random_state=seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro')

            print(f"Testing C={C}, penalty={penalty} => F1 Score: {f1}")

            if f1 > best_f1_score:
                best_f1_score = f1
                best_model = model
                best_params = {'C': C, 'penalty': penalty}

    model_output_dir = os.path.join(model_output_base_dir, dataset_name)
    os.makedirs(model_output_dir, exist_ok=True)

    model_path = os.path.join(model_output_dir, 'model.pkl')
    vectorizer_path = os.path.join(model_output_dir, 'vectorizer.pkl')

    joblib.dump(best_model, model_path)
    if vectorizer is not None:
        joblib.dump(vectorizer, vectorizer_path)

    print(f"Best model parameters: {best_params}")
    print(f"Model and vectorizer saved to {model_output_dir}")

    emissions = tracker.stop()
    print(f"Estimated total emissions: {emissions} kg CO2")

    return model_output_dir

# Define function for evaluating the model
def evaluate_model(test_data, model_output_dir, result_output_base_dir, dataset_name, feature_type, seed):

    np.random.seed(seed)
    num_labels = len(test_data['label'].unique())
    print(f"Number of unique labels: {num_labels}")

    nlp = spacy.load("en_core_web_lg")

    model_path = os.path.join(model_output_dir, 'model.pkl')
    model = joblib.load(model_path)

    vectorizer_path = os.path.join(model_output_dir, 'vectorizer.pkl')
    vectorizer = None
    if feature_type == 'tfidf':
        vectorizer = joblib.load(vectorizer_path)

    if feature_type == 'tfidf':
        X_test = vectorizer.transform(test_data['text'])
    elif feature_type == 'embedding':
        def get_embedding(text):
            doc = nlp(text)
            return doc.vector

        X_test = np.array([get_embedding(text) for text in tqdm(test_data['text'], desc="Computing embeddings")])

    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

    result_output_dir = os.path.join(result_output_base_dir, dataset_name)
    os.makedirs(result_output_dir, exist_ok=True)

    y_pred_probs = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    pred_labels = y_pred.tolist()
    pred_probs = y_pred_probs.max(axis=1).tolist()
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

    report = classification_report(y_true, pred_labels, output_dict=True)
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

# Execute full pipeline for logistic regression tfidf model
train_model(train_data_mgsd, model_output_base_dir='model_output_LR_tfidf', dataset_name='mgsd_trained', feature_type='tfidf', seed=42)
evaluate_model(test_data_winoqueer_gpt_augmentation, model_output_dir='model_output_LR_tfidf/mgsd_trained', result_output_base_dir='result_output_LR_tfidf/mgsd_trained', dataset_name='winoqueer_gpt_augmentation', feature_type='tfidf', seed=42)
evaluate_model(test_data_seegull_gpt_augmentation, model_output_dir='model_output_LR_tfidf/mgsd_trained', result_output_base_dir='result_output_LR_tfidf/mgsd_trained', dataset_name='seegull_gpt_augmentation', feature_type='tfidf', seed=42)
evaluate_model(test_data_mgsd, model_output_dir='model_output_LR_tfidf/mgsd_trained', result_output_base_dir='result_output_LR_tfidf/mgsd_trained', dataset_name='mgsd', feature_type='tfidf', seed=42)
evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, model_output_dir='model_output_LR_tfidf/mgsd_trained', result_output_base_dir='result_output_LR_tfidf/mgsd_trained', dataset_name='merged_winoqueer_seegull_gpt_augmentation', feature_type='tfidf', seed=42)

train_model(train_data_winoqueer_gpt_augmentation, model_output_base_dir='model_output_LR_tfidf', dataset_name='winoqueer_gpt_augmentation_trained', feature_type='tfidf', seed=42)
evaluate_model(test_data_winoqueer_gpt_augmentation, model_output_dir='model_output_LR_tfidf/winoqueer_gpt_augmentation_trained', result_output_base_dir='result_output_LR_tfidf/winoqueer_gpt_augmentation_trained', dataset_name='winoqueer_gpt_augmentation', feature_type='tfidf', seed=42)
evaluate_model(test_data_seegull_gpt_augmentation, model_output_dir='model_output_LR_tfidf/winoqueer_gpt_augmentation_trained', result_output_base_dir='result_output_LR_tfidf/winoqueer_gpt_augmentation_trained', dataset_name='seegull_gpt_augmentation', feature_type='tfidf', seed=42)
evaluate_model(test_data_mgsd, model_output_dir='model_output_LR_tfidf/winoqueer_gpt_augmentation_trained', result_output_base_dir='result_output_LR_tfidf/winoqueer_gpt_augmentation_trained', dataset_name='mgsd', feature_type='tfidf', seed=42)
evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, model_output_dir='model_output_LR_tfidf/winoqueer_gpt_augmentation_trained', result_output_base_dir='result_output_LR_tfidf/winoqueer_gpt_augmentation_trained', dataset_name='merged_winoqueer_seegull_gpt_augmentation', feature_type='tfidf', seed=42)

train_model(train_data_seegull_gpt_augmentation, model_output_base_dir='model_output_LR_tfidf', dataset_name='seegull_gpt_augmentation_trained', feature_type='tfidf', seed=42)
evaluate_model(test_data_winoqueer_gpt_augmentation, model_output_dir='model_output_LR_tfidf/seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_tfidf/seegull_gpt_augmentation_trained', dataset_name='winoqueer_gpt_augmentation', feature_type='tfidf', seed=42)
evaluate_model(test_data_seegull_gpt_augmentation, model_output_dir='model_output_LR_tfidf/seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_tfidf/seegull_gpt_augmentation_trained', dataset_name='seegull_gpt_augmentation', feature_type='tfidf', seed=42)
evaluate_model(test_data_mgsd, model_output_dir='model_output_LR_tfidf/seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_tfidf/seegull_gpt_augmentation_trained', dataset_name='mgsd', feature_type='tfidf', seed=42)
evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, model_output_dir='model_output_LR_tfidf/seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_tfidf/seegull_gpt_augmentation_trained', dataset_name='merged_winoqueer_seegull_gpt_augmentation', feature_type='tfidf', seed=42)

train_model(train_data_merged_winoqueer_seegull_gpt_augmentation, model_output_base_dir='model_output_LR_tfidf', dataset_name='merged_winoqueer_seegull_gpt_augmentation_trained', feature_type='tfidf', seed=42)
evaluate_model(test_data_winoqueer_gpt_augmentation, model_output_dir='model_output_LR_tfidf/merged_winoqueer_seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_tfidf/merged_winoqueer_seegull_gpt_augmentation_trained', dataset_name='winoqueer_gpt_augmentation', feature_type='tfidf', seed=42)
evaluate_model(test_data_seegull_gpt_augmentation, model_output_dir='model_output_LR_tfidf/merged_winoqueer_seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_tfidf/merged_winoqueer_seegull_gpt_augmentation_trained', dataset_name='seegull_gpt_augmentation', feature_type='tfidf', seed=42)
evaluate_model(test_data_mgsd, model_output_dir='model_output_LR_tfidf/merged_winoqueer_seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_tfidf/merged_winoqueer_seegull_gpt_augmentation_trained', dataset_name='mgsd', feature_type='tfidf', seed=42)
evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, model_output_dir='model_output_LR_tfidf/merged_winoqueer_seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_tfidf/merged_winoqueer_seegull_gpt_augmentation_trained', dataset_name='merged_winoqueer_seegull_gpt_augmentation', feature_type='tfidf', seed=42)

# Execute full pipeline for logistic regression embedding model
train_model(train_data_mgsd, model_output_base_dir='model_output_LR_embedding', dataset_name='mgsd_trained', feature_type='embedding', seed=42)
evaluate_model(test_data_winoqueer_gpt_augmentation, model_output_dir='model_output_LR_embedding/mgsd_trained', result_output_base_dir='result_output_LR_embedding/mgsd_trained', dataset_name='winoqueer_gpt_augmentation', feature_type='embedding', seed=42)
evaluate_model(test_data_seegull_gpt_augmentation, model_output_dir='model_output_LR_embedding/mgsd_trained', result_output_base_dir='result_output_LR_embedding/mgsd_trained', dataset_name='seegull_gpt_augmentation', feature_type='embedding', seed=42)
evaluate_model(test_data_mgsd, model_output_dir='model_output_LR_embedding/mgsd_trained', result_output_base_dir='result_output_LR_embedding/mgsd_trained', dataset_name='mgsd', feature_type='embedding', seed=42)
evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, model_output_dir='model_output_LR_embedding/mgsd_trained', result_output_base_dir='result_output_LR_embedding/mgsd_trained', dataset_name='merged_winoqueer_seegull_gpt_augmentation', feature_type='embedding', seed=42)

train_model(train_data_winoqueer_gpt_augmentation, model_output_base_dir='model_output_LR_embedding', dataset_name='winoqueer_gpt_augmentation_trained', feature_type='embedding', seed=42)
evaluate_model(test_data_winoqueer_gpt_augmentation, model_output_dir='model_output_LR_embedding/winoqueer_gpt_augmentation_trained', result_output_base_dir='result_output_LR_embedding/winoqueer_gpt_augmentation_trained', dataset_name='winoqueer_gpt_augmentation', feature_type='embedding', seed=42)
evaluate_model(test_data_seegull_gpt_augmentation, model_output_dir='model_output_LR_embedding/winoqueer_gpt_augmentation_trained', result_output_base_dir='result_output_LR_embedding/winoqueer_gpt_augmentation_trained', dataset_name='seegull_gpt_augmentation', feature_type='embedding', seed=42)
evaluate_model(test_data_mgsd, model_output_dir='model_output_LR_embedding/winoqueer_gpt_augmentation_trained', result_output_base_dir='result_output_LR_embedding/winoqueer_gpt_augmentation_trained', dataset_name='mgsd', feature_type='embedding', seed=42)
evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, model_output_dir='model_output_LR_embedding/winoqueer_gpt_augmentation_trained', result_output_base_dir='result_output_LR_embedding/winoqueer_gpt_augmentation_trained', dataset_name='merged_winoqueer_seegull_gpt_augmentation', feature_type='embedding', seed=42)

train_model(train_data_seegull_gpt_augmentation, model_output_base_dir='model_output_LR_embedding', dataset_name='seegull_gpt_augmentation_trained', feature_type='embedding', seed=42)
evaluate_model(test_data_winoqueer_gpt_augmentation, model_output_dir='model_output_LR_embedding/seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_embedding/seegull_gpt_augmentation_trained', dataset_name='winoqueer_gpt_augmentation', feature_type='embedding', seed=42)
evaluate_model(test_data_seegull_gpt_augmentation, model_output_dir='model_output_LR_embedding/seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_embedding/seegull_gpt_augmentation_trained', dataset_name='seegull_gpt_augmentation', feature_type='embedding', seed=42)
evaluate_model(test_data_mgsd, model_output_dir='model_output_LR_embedding/seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_embedding/seegull_gpt_augmentation_trained', dataset_name='mgsd', feature_type='embedding', seed=42)
evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, model_output_dir='model_output_LR_embedding/seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_embedding/seegull_gpt_augmentation_trained', dataset_name='merged_winoqueer_seegull_gpt_augmentation', feature_type='embedding', seed=42)

train_model(train_data_merged_winoqueer_seegull_gpt_augmentation, model_output_base_dir='model_output_LR_embedding', dataset_name='merged_winoqueer_seegull_gpt_augmentation_trained', feature_type='embedding', seed=42)
evaluate_model(test_data_winoqueer_gpt_augmentation, model_output_dir='model_output_LR_embedding/merged_winoqueer_seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_embedding/merged_winoqueer_seegull_gpt_augmentation_trained', dataset_name='winoqueer_gpt_augmentation', feature_type='embedding', seed=42)
evaluate_model(test_data_seegull_gpt_augmentation, model_output_dir='model_output_LR_embedding/merged_winoqueer_seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_embedding/merged_winoqueer_seegull_gpt_augmentation_trained', dataset_name='seegull_gpt_augmentation', feature_type='embedding', seed=42)
evaluate_model(test_data_mgsd, model_output_dir='model_output_LR_embedding/merged_winoqueer_seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_embedding/merged_winoqueer_seegull_gpt_augmentation_trained', dataset_name='mgsd', feature_type='embedding', seed=42)
evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, model_output_dir='model_output_LR_embedding/merged_winoqueer_seegull_gpt_augmentation_trained', result_output_base_dir='result_output_LR_embedding/merged_winoqueer_seegull_gpt_augmentation_trained', dataset_name='merged_winoqueer_seegull_gpt_augmentation', feature_type='embedding', seed=42)
