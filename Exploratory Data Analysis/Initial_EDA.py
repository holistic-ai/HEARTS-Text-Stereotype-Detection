import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Define function to load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded dataset with {df.shape[0]} observations.")
    return df

# Define function to show target variable distribution (prevalence of stereotypes)
def prepare_target_variable_distribution(df, target_col):
    return df[target_col].value_counts()

# Define function to show stereotype group distributions
def prepare_group_distribution(df, group_col):
    return df[group_col].value_counts()

# Define function to perform text length analysis on each statement
def prepare_text_length_analysis(df, text_col):
    df[text_col] = df[text_col].fillna('').astype(str)
    df['text_length'] = df[text_col].apply(len)
    length_counts = df['text_length'].value_counts().rename('count_of_texts')
    df = df.join(length_counts, on='text_length')
    return df[['text_length', 'count_of_texts']]

# Define function to create a word cloud from text data
def create_word_cloud(df, text_col, output_filename):
    text = " ".join(review for review in df[text_col])
    wordcloud = WordCloud(width=1600, height=800, background_color="white", max_words=200, max_font_size=250).generate(text)
    plt.figure(figsize=(16, 8), dpi=600)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(output_filename, format='png', dpi=600)

# Execute usage - MGSD Expanded
df_mgsd_expanded = load_data(filepath='MGSD - Expanded.csv')
target_dist_mgsd_expanded = prepare_target_variable_distribution(df_mgsd_expanded, target_col='category')
group_dist_mgsd_expanded = prepare_group_distribution(df_mgsd_expanded, group_col='stereotype_type')
text_length_mgsd_expanded = prepare_text_length_analysis(df_mgsd_expanded, text_col='text')
create_word_cloud(df_mgsd_expanded, text_col='text', output_filename='MGSD_Expanded_wordcloud.png')

# Save to CSV files
target_dist_mgsd_expanded.to_csv('Target_Distribution.csv')
group_dist_mgsd_expanded.to_csv('Group_Distribution.csv')
text_length_mgsd_expanded.to_csv('Text_Length_Analysis.csv')