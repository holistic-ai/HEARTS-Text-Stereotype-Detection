import pandas as pd
from transformers import pipeline

def classify_texts(filename, model_name):
    df = pd.read_csv(filename)
        
    classifier = pipeline("sentiment-analysis", model=model_name)
    
    df['prediction'] = df['prompt'].apply(lambda x: classifier(x)[0]['label'])
    
    output_filename = "Classified_" + filename
    df.to_csv(output_filename, index=False)
    print(f"Classification complete. Results saved to {output_filename}")

classify_texts("LLM Prompts.csv", "holistic-ai/bias_classifier_albertv2")
