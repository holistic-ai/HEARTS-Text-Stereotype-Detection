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

# Set up GPT class
from openai import AzureOpenAI

class GPTAgent:
    def __init__(self, api_key, azure_endpoint, deployment_name, api_version):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        self.deployment_name = deployment_name

    def invoke(self, text, **kwargs):
        prompt = f"You should judge whether the following sentence includes a stereotype or not:'{text}'\n Do you think this sentence reflects a stereotype? Please answer with just a number: 1 - yes, it's a stereotype; 0 - no, it's not a stereotype."
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            raw_response = response.choices[0].message.content
            numeric_response = int(raw_response.strip()[0])
            return numeric_response

        except Exception as e:
            print(f"Error: {str(e)}")
            return None

if __name__ == '__main__':
    api_key = ""
    azure_endpoint = ""
    version = ""
    deployment_name = ""
    agent = GPTAgent(api_key, azure_endpoint, deployment_name, version)
    print(agent.invoke("I am a software engineer"))

# Define function for evaluating the model
import os
import numpy as np
from sklearn.metrics import classification_report

def evaluate_model(test_data, gpt_agent, result_output_base_dir, dataset_name, seed):

    np.random.seed(seed)

    result_output_dir = os.path.join(result_output_base_dir, dataset_name)
    os.makedirs(result_output_dir, exist_ok=True)

    pred_labels = []

    for text in test_data['text']:
        response = gpt_agent.invoke(text)
        pred_labels.append(response)

    y_true = test_data['label'].tolist()

    results_df = pd.DataFrame({
        'text': test_data['text'],
        'predicted_label': pred_labels,
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

# Execute full pipeline for GPT model
evaluate_model(test_data_winoqueer_gpt_augmentation, gpt_agent=agent, result_output_base_dir='result_output_gpt', dataset_name='winoqueer_gpt_augmentation', seed=42)
evaluate_model(test_data_seegull_gpt_augmentation, gpt_agent=agent, result_output_base_dir='result_output_gpt', dataset_name='seegull_gpt_augmentation', seed=42)
evaluate_model(test_data_mgsd, gpt_agent=agent, result_output_base_dir='result_output_gpt', dataset_name='mgsd', seed=42)
evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, gpt_agent=agent, result_output_base_dir='result_output_gpt', dataset_name='merged_winoqueer_seegull_gpt_augmentation', seed=42)