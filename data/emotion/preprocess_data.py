import numpy as np
import pandas as pd
import pdb
import json
from utils import sort_strings_substrings_last
import krippendorff


# load datasets
df1 = pd.read_csv('data/emotion/data_raw/data/full_dataset/goemotions_1.csv')
df2 = pd.read_csv('data/emotion/data_raw/data/full_dataset/goemotions_2.csv')
df3 = pd.read_csv('data/emotion/data_raw/data/full_dataset/goemotions_3.csv')
df = pd.concat([df1, df2, df3])

nr_of_ground_truth_annotators = len(df['rater_id'].unique())
print('nr_of_ground_truth_annotators:', nr_of_ground_truth_annotators)

nr_of_ground_truth_annotators_per_datapoint = len(df) / len(df['id'].unique())
print(f'Average number of annotators per datapoint: {nr_of_ground_truth_annotators_per_datapoint}')

ground_truth_columns = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

def get_annotations(row):
    return row[ground_truth_columns][row[ground_truth_columns] == 1].index.tolist()

# get list of annotations for each id
df['ground_truth'] = df.apply(get_annotations, axis=1)

df_temp_for_agreement = df.copy()

df = df[~df['example_very_unclear']]

df.drop(columns=ground_truth_columns+['example_very_unclear'], inplace=True)

df = df.groupby([
    'text',
    'id',
    'author',
    'subreddit',
    'link_id',
    'parent_id',
    'created_utc'
]).agg({
    'ground_truth': list,
    'rater_id': list,
}).reset_index()

# drop datapoints with less than 3 raters
df['nr_of_raters'] = df['rater_id'].apply(lambda x: len(set(x)))
df = df[df['nr_of_raters'] >= 3]

# only include datapoints where all annotators agree on a single label
# first flatten gt labels
df['all_gt_annotations'] = df['ground_truth'].apply(lambda x: [l for a in x for l in a])
df['nr_of_gt_annotations'] = df['all_gt_annotations'].apply(lambda x: len(x))
df['nr_of_unique_gt_annotations'] = df['all_gt_annotations'].apply(lambda x: len(set(x)))
df = df[
    (df['nr_of_gt_annotations'] == df['nr_of_raters']) &
    (df['nr_of_unique_gt_annotations'] == 1)
    ]

df['ground_truth'] = df['all_gt_annotations'].apply(lambda x: x[0])
df.drop(columns=['nr_of_gt_annotations'], inplace=True)
df.drop(columns=['nr_of_unique_gt_annotations'], inplace=True)
df.drop(columns=['all_gt_annotations'], inplace=True)
df.drop(columns=['rater_id'], inplace=True)
df.drop(columns=['nr_of_raters'], inplace=True)


with open('data/emotion/sentiment_dict.json', 'r') as f:
    sentiment_dict = json.load(f)

# Create a reverse mapping from emotion to sentiment
emotion_to_sentiment = {}
for sentiment, emotions in sentiment_dict.items():
    for emotion in emotions:
        emotion_to_sentiment[emotion] = sentiment

# Map ground truth values to sentiment
df['sentiment'] = df['ground_truth'].map(emotion_to_sentiment)


# filter for max 26 classes
# We consider only the most frequent 26 annotation categories and excluding the two classes 'pride' and 'grief', which are not present in this subset of the data.
all_gt_classes = list(df['ground_truth'].value_counts(
    dropna=False).head(26).keys())
all_gt_classes = sort_strings_substrings_last(all_gt_classes)
print('all_gt_classes:', all_gt_classes)

# print dict used for mapping LLM output to class
print('Use the following mapping for the prompt:')
print('dict_with_mapping_options = {')
alphabet_letters_capital = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
assert len(all_gt_classes) <= len(list(alphabet_letters_capital)
                                  ), "too many values in all_gt_classes"
for idx, c in enumerate(all_gt_classes):
    print(f"    '{alphabet_letters_capital[idx]}': '{c}',")
print('}')

df = df[df['ground_truth'].isin(all_gt_classes)]

print(f"\nFinal df dataset shape: {df.shape}")
print(f"Final columns: {list(df.columns)}")
print(
    f"Ground truth distribution: {df['ground_truth'].value_counts().sort_index()}")

df.reset_index(drop=True, inplace=True)

# sort df columns to start with 'ground_truth', then 'text', and then all other columns
priority_columns = ['ground_truth', 'text']
other_columns = [
    col for col in df.columns if col not in priority_columns]
df = df[priority_columns + other_columns]

df.to_csv('data/all_data_processed_full/emotion.csv', index=False)



# Get unique IDs and raters
unique_ids = df_temp_for_agreement['id'].unique()
unique_raters = df_temp_for_agreement['rater_id'].unique()

print(f"Number of unique texts: {len(unique_ids)}")
print(f"Number of unique raters: {len(unique_raters)}")

# Calculate alpha for each emotion separately
emotion_alphas = {}

for emotion in all_gt_classes:
    # Create a matrix where rows are items (texts) and columns are raters
    # Initialize with NaN (missing values)
    annotation_matrix = np.full((len(unique_ids), len(unique_raters)), np.nan)
    # Create mappings for indices
    id_to_idx = {id_val: idx for idx, id_val in enumerate(unique_ids)}
    rater_to_idx = {rater: idx for idx, rater in enumerate(unique_raters)}
    # Fill the matrix
    for _, row in df_temp_for_agreement.iterrows():
        text_idx = id_to_idx[row['id']]
        rater_idx = rater_to_idx[row['rater_id']]
        # Use binary encoding: 1 if emotion is present, 0 if not
        annotation_matrix[text_idx, rater_idx] = row[emotion]
    # Calculate Krippendorff's alpha for this emotion
    # Transpose because krippendorff expects raters as rows
    alpha = krippendorff.alpha(reliability_data=annotation_matrix.T, 
                               level_of_measurement='nominal')
    emotion_alphas[emotion] = alpha
    print(f"Krippendorff's alpha for '{emotion}': {alpha:.3f}")

# Calculate overall statistics
krippendorff_alpha = np.mean(list(emotion_alphas.values()))
print(f"Average alpha across all emotions: {krippendorff_alpha:.4f}")

