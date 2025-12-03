import pdb
import numpy as np
import pandas as pd
import krippendorff
from scipy import stats
from convokit import Corpus, download
from data.data_utils import process_full_dataset

df_stack = pd.read_csv(
    "data/politeness/data_raw/Stanford_politeness_corpus/stack-exchange.annotated.csv")
df_stack_meta = pd.read_csv(
    "data/politeness/data_raw/Stanford_politeness_corpus/stack-exchange.requests.csv")
df_stack_meta.rename(columns={'Request': 'Request_y'}, inplace=True)
# merge df_stack_meta onto df_stack (left) with 'Community' 'Id'
df_stack = df_stack.merge(df_stack_meta, on=['Community', 'Id'], how='left')
df_stack.rename(columns={'Request': 'text'}, inplace=True)

# df_stack_roles = pd.read_csv("data/politeness/data_raw/Stanford_politeness_corpus/stack-exchange.roles.csv")

politeness_corpus = Corpus(filename=download(
    "stack-exchange-politeness-corpus"))
politeness_corpus = politeness_corpus.get_utterances_dataframe(
)[['text', 'meta.Normalized Score', 'meta.Binary', 'meta.Annotations']]

# merge politeness_corpus onto df_stack for politeness_corpus.index == df_stack['Id'] and politeness_corpus['text'] df_stack['text']
# first check if both dataframes contain exactly one row for each id-text-combination
print("Checking id-text combinations in both dataframes...")

# Check df_stack for duplicates
df_stack_groups = df_stack.groupby(['Id', 'text']).size()
df_stack_duplicates = df_stack_groups[df_stack_groups > 1]
print(
    f"Duplicate id-text combinations in df_stack: {len(df_stack_duplicates)}")
if len(df_stack_duplicates) > 0:
    print("Examples of duplicates in df_stack:")
    print(df_stack_duplicates.head())

# Reset index to make it a column for merging
politeness_corpus_with_id = politeness_corpus.reset_index()
politeness_corpus_with_id.rename(columns={'id': 'index'}, inplace=True)

df_stack = df_stack.reset_index()

# Convert index to same type for proper comparison
df_stack['index'] = df_stack['index'].astype(str)
politeness_corpus_with_id['index'] = politeness_corpus_with_id['index'].astype(
    str)

# Check politeness_corpus for duplicates
politeness_groups = politeness_corpus_with_id.groupby(['index', 'text']).size()
politeness_duplicates = politeness_groups[politeness_groups > 1]
print(
    f"Duplicate id-text combinations in politeness_corpus: {len(politeness_duplicates)}")
if len(politeness_duplicates) > 0:
    print("Examples of duplicates in politeness_corpus:")
    print(politeness_duplicates.head())

# Create sets of id-text combinations for comparison
df_stack_pairs = set(zip(df_stack['index'], df_stack['text']))
politeness_pairs = set(
    zip(politeness_corpus_with_id['index'], politeness_corpus_with_id['text']))

print(f"\nTotal unique index-text pairs in df_stack: {len(df_stack_pairs)}")
print(
    f"Total unique index-text pairs in politeness_corpus: {len(politeness_pairs)}")

# Check if they contain exactly the same combinations
only_in_stack = df_stack_pairs - politeness_pairs
only_in_politeness = politeness_pairs - df_stack_pairs
common_pairs = df_stack_pairs & politeness_pairs

print(f"Pairs only in df_stack: {len(only_in_stack)}")
print(f"Pairs only in politeness_corpus: {len(only_in_politeness)}")
print(f"Common pairs: {len(common_pairs)}")

if len(only_in_stack) > 0:
    print("Examples of pairs only in df_stack:")
    print(list(only_in_stack)[:5])

if len(only_in_politeness) > 0:
    print("Examples of pairs only in politeness_corpus:")
    print(list(only_in_politeness)[:5])

# Verify exact match
exact_match = len(only_in_stack) == 0 and len(only_in_politeness) == 0
print(
    f"\nDataframes contain exactly the same id-text combinations: {exact_match}")

# Merge on both Id and text
df_stack = df_stack.merge(
    politeness_corpus_with_id[[
        'index', 'text', 'meta.Normalized Score', 'meta.Binary', 'meta.Annotations']],
    on=['index', 'text'],
    how='left'
)

df_stack['ground_truth'] = df_stack['meta.Binary'].map({
    -1: 'impolite',
    0: 'neutral',
    1: 'polite',
})


# calculate annotator agreement

# Convert meta.Normalized Score to float
df_stack['meta.Normalized Score'] = pd.to_numeric(df_stack['meta.Normalized Score'])

# first, obtain the thresholds for binarization, looking at cutoffs in meta.Normalized Score for all meta.Binary column values
print("\nAnalyzing thresholds for binarization...")
thresholds_df = df_stack[['meta.Normalized Score', 'meta.Binary']].drop_duplicates()
thresholds_df = thresholds_df.sort_values('meta.Normalized Score')

# Find thresholds between classes
impolite_neutral_threshold = thresholds_df[
   (thresholds_df['meta.Binary'] == -1) & 
   (thresholds_df['meta.Binary'].shift(-1) == 0)
]['meta.Normalized Score'].max()

neutral_polite_threshold = thresholds_df[
   (thresholds_df['meta.Binary'] == 0) & 
   (thresholds_df['meta.Binary'].shift(-1) == 1)
]['meta.Normalized Score'].max()

print(f"Impolite-Neutral threshold: {impolite_neutral_threshold}")
print(f"Neutral-Polite threshold: {neutral_polite_threshold}")

# now, normalize 'Score1', 'Score2', 'Score3', 'Score4', 'Score5' to be within -1 and 1
# Apply z-score normalization to each worker's scores independently
score_cols = ['Score1', 'Score2', 'Score3', 'Score4', 'Score5']

for col in score_cols:
   df_stack[f'{col}_normalized'] = stats.zscore(df_stack[col])

# Check if average of normalized scores corresponds to meta.Normalized Score
avg_normalized = df_stack[[f'{col}_normalized' for col in score_cols]].mean(axis=1)
print(f"Correlation between avg_normalized and meta.Normalized Score: {avg_normalized.corr(df_stack['meta.Normalized Score']):.4f}")

# next, binarize each of 'Score1', 'Score2', 'Score3', 'Score4', 'Score5'
for col in score_cols:
   df_stack[f'{col}_binarized'] = df_stack[f'{col}_normalized'].apply(
       lambda x: -1 if x < impolite_neutral_threshold 
       else 1 if x > neutral_polite_threshold 
       else 0
   )

# # then, calculate krippendorff alpha for all 5 binarized annotation scores
# # Prepare data for Krippendorff's alpha
# annotation_matrix = df_stack[[f'{col}_binarized' for col in score_cols]].T.values

# # Calculate Krippendorff's alpha
# krippendorff_alpha_full_dataset = krippendorff.alpha(reliability_data=annotation_matrix, level_of_measurement='nominal')
# print(f'krippendorff alpha full dataset: {krippendorff_alpha_full_dataset:.4f}')


# load data preprocessed by Gligoric et al. (2025), which includes all of the features
df_with_features = pd.read_csv(
    'data/politeness/data_raw/politeness_dataset.csv')
df_with_features = df_with_features[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9', 'Feature_10',
                                     'Feature_11', 'Feature_12', 'Feature_13', 'Feature_14', 'Feature_15', 'Feature_16', 'Feature_17', 'Feature_18', 'Feature_19', 'Feature_20', 'Feature_21', 'Politeness', 'Text']]
df_with_features['Politeness'] = df_with_features['Politeness'].map({
    0: 'impolite',
    1: 'polite',
})
df_with_features.rename(columns={'Text': 'text'}, inplace=True)
df_with_features.drop_duplicates(subset=['text'], keep='first', inplace=True)
df_stack = df_stack.merge(df_with_features, on=['text'], how='left')

aa = df_stack.drop_duplicates(subset=['text'], keep='first')
aaa = aa[aa['Politeness'].notna()]
aaa = df_stack[(df_stack['Politeness'].notna()) & (df_stack['Politeness'] == df_stack['ground_truth'])].drop_duplicates(subset=['text'], keep='first')
print(
    f'Notes: This reduces the number of datapoints from {len(aa)} to {len(aaa)}')

df_stack = df_stack[(df_stack['Politeness'].notna()) & (df_stack['Politeness'] == df_stack['ground_truth'])]

# test normalized score distribution (i.e., columns 'Normalized Score' and then also 'meta.Normalized Score) for each ground truth score
print("\nNormalized Score distribution by ground truth:")
for gt in ['impolite', 'neutral', 'polite']:
    subset = df_stack[df_stack['ground_truth'] == gt]['Normalized Score']
    if not subset.empty:
        print(
            f"  Ground truth {gt}: mean={subset.mean():.3f}, std={subset.std():.3f}, count={len(subset)}")

print("\n'meta.Normalized Score' column:")
for gt in ['impolite', 'neutral', 'polite']:
    subset = df_stack[df_stack['ground_truth'] == gt]['meta.Normalized Score']
    if not subset.empty:
        print(
            f"  Ground truth {gt}: mean={subset.mean():.3f}, std={subset.std():.3f}, count={len(subset)}")

# finally, remove unnecessary columns
df_stack.rename(columns={'Community_x': 'Community'}, inplace=True)


df_final = process_full_dataset('politeness_stack', dataset=df_stack, save_to_disk=False)


# Create a dictionary to map annotator-item pairs to their annotations
annotation_dict = {}

# Iterate through each row to collect annotations
for idx, row in df_final.iterrows():
    # For each of the 5 annotators per item
    for i in range(1, 6):
        turk_id = row[f'TurkId{i}']
        score = row[f'Score{i}_binarized']
        # Store the annotation with (annotator_id, item_index) as key
        annotation_dict[(turk_id, idx)] = score

# Get all unique annotators and items
all_annotators = sorted(set(
    df_final[['TurkId1', 'TurkId2', 'TurkId3', 'TurkId4', 'TurkId5']].values.flatten()
))
n_items = len(df_final)

print(f'Number of unique annotators: {len(all_annotators)}')

# Create annotation matrix (annotators x items)
# Use None for missing values (annotator didn't rate that item)
annotation_matrix_final = []
for annotator in all_annotators:
    row = []
    for item_idx in range(n_items):
        if (annotator, item_idx) in annotation_dict:
            row.append(annotation_dict[(annotator, item_idx)])
        else:
            row.append(np.nan)
    annotation_matrix_final.append(row)

# Convert to numpy array for krippendorff
annotation_matrix_final = np.array(annotation_matrix_final, dtype=float)

# Calculate Krippendorff's alpha on final dataset
krippendorff_alpha = krippendorff.alpha(reliability_data=annotation_matrix_final, level_of_measurement='nominal')
print(f'Krippendorff alpha: {krippendorff_alpha:.4f}')


columns_to_drop = ['Request_y', 'meta.Binary', 'Community_y', 'Normalized Score', 'Politeness', 'meta.Annotations', 'TEXT_FORMATTED_TO_CHECK_FOR_DUPLICATES']
# Only drop columns that actually exist
columns_to_drop = [col for col in columns_to_drop if col in df_stack.columns]
df_stack.drop(columns=columns_to_drop, inplace=True)

print(f"\nFinal dataset shape: {df_stack.shape}")
print(f"Final columns: {list(df_stack.columns)}")
print(
    f"Ground truth distribution: {df_stack['ground_truth'].value_counts().sort_index()}")

df_stack.reset_index(drop=True, inplace=True)

# sort df_stack columns to start with 'ground_truth', then 'text', and then all other columns
priority_columns = ['ground_truth', 'text']
priority2_columns = [
    col for col in df_stack.columns if col not in priority_columns and 'feature_' in col]
priority_columns = priority_columns + priority2_columns
other_columns = [
    col for col in df_stack.columns if col not in priority_columns]
df_stack = df_stack[priority_columns + other_columns]

df_stack.to_csv('data/all_data_processed_full/politeness_stack.csv', index=False)
