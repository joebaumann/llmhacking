import pdb
import pandas as pd
import numpy as np
import krippendorff
from scipy import stats
from convokit import Corpus, download
from data.data_utils import process_full_dataset

df_wiki = pd.read_csv(
    "data/politeness/data_raw/Stanford_politeness_corpus/wikipedia.annotated.csv")
df_wiki_meta = pd.read_csv(
    "data/politeness/data_raw/Stanford_politeness_corpus/wikipedia.requests.csv")
df_wiki_meta.rename(columns={'Request': 'Request_y'}, inplace=True)
# merge df_wiki_meta onto df_wiki (left) with 'Id'
df_wiki = df_wiki.merge(df_wiki_meta, on='Id', how='left')
df_wiki.rename(columns={'Request': 'text'}, inplace=True)

df_wiki_admins = pd.read_csv(
    "data/politeness/data_raw/Stanford_politeness_corpus/wikipedia.admins.csv")
# Create a dictionary mapping users to their adminship dates
df_wiki_admins_list = dict(
    # Fixed typo: 'Adiminship' -> 'Adminship'
    zip(df_wiki_admins['User'], df_wiki_admins['Adiminship date']))
df_wiki['is_admin'] = df_wiki['User'].apply(lambda x: x in df_wiki_admins_list)
df_wiki['Adminship date'] = df_wiki['User'].map(
    df_wiki_admins_list)  # Fixed typo
# The date when this status was gained through a Request for Adminship election process (http://en.wikipedia.org/wiki/Wikipedia:Requests_for_adminship) is indicated in yyyy-mm-dd  format (missing dates are indicated with NA)
# make datetime to be saved as csv later
df_wiki['Adminship date'] = pd.to_datetime(
    df_wiki['Adminship date'], errors='coerce')  # Fixed the incomplete line

politeness_corpus = Corpus(filename=download("wikipedia-politeness-corpus"))
politeness_corpus = politeness_corpus.get_utterances_dataframe(
)[['text', 'meta.Normalized Score', 'meta.Binary', 'meta.Annotations']]

# merge politeness_corpus onto df_wiki for politeness_corpus.index == df_wiki['Id'] and politeness_corpus['text'] df_wiki['text']
# first check if both dataframes contain exactly one row for each id-text-combination
print("Checking id-text combinations in both dataframes...")

# Check df_wiki for duplicates
df_wiki_groups = df_wiki.groupby(['Id', 'text']).size()
df_wiki_duplicates = df_wiki_groups[df_wiki_groups > 1]
print(f"Duplicate id-text combinations in df_wiki: {len(df_wiki_duplicates)}")
if len(df_wiki_duplicates) > 0:
    print("Examples of duplicates in df_wiki:")
    print(df_wiki_duplicates.head())

# Reset index to make it a column for merging
politeness_corpus_with_id = politeness_corpus.reset_index()
politeness_corpus_with_id.rename(columns={'id': 'Id'}, inplace=True)

# Convert Id to same type for proper comparison
df_wiki['Id'] = df_wiki['Id'].astype(str)
politeness_corpus_with_id['Id'] = politeness_corpus_with_id['Id'].astype(str)

# Check politeness_corpus for duplicates
politeness_groups = politeness_corpus_with_id.groupby(['Id', 'text']).size()
politeness_duplicates = politeness_groups[politeness_groups > 1]
print(
    f"Duplicate id-text combinations in politeness_corpus: {len(politeness_duplicates)}")
if len(politeness_duplicates) > 0:
    print("Examples of duplicates in politeness_corpus:")
    print(politeness_duplicates.head())

# Create sets of id-text combinations for comparison
df_wiki_pairs = set(zip(df_wiki['Id'], df_wiki['text']))
politeness_pairs = set(
    zip(politeness_corpus_with_id['Id'], politeness_corpus_with_id['text']))

print(f"\nTotal unique id-text pairs in df_wiki: {len(df_wiki_pairs)}")
print(
    f"Total unique id-text pairs in politeness_corpus: {len(politeness_pairs)}")

# Check if they contain exactly the same combinations
only_in_wiki = df_wiki_pairs - politeness_pairs
only_in_politeness = politeness_pairs - df_wiki_pairs
common_pairs = df_wiki_pairs & politeness_pairs

print(f"Pairs only in df_wiki: {len(only_in_wiki)}")
print(f"Pairs only in politeness_corpus: {len(only_in_politeness)}")
print(f"Common pairs: {len(common_pairs)}")

if len(only_in_wiki) > 0:
    print("Examples of pairs only in df_wiki:")
    print(list(only_in_wiki)[:5])

if len(only_in_politeness) > 0:
    print("Examples of pairs only in politeness_corpus:")
    print(list(only_in_politeness)[:5])

# Verify exact match
exact_match = len(only_in_wiki) == 0 and len(only_in_politeness) == 0
print(
    f"\nDataframes contain exactly the same id-text combinations: {exact_match}")

# Merge on both Id and text
df_wiki = df_wiki.merge(
    politeness_corpus_with_id[[
        'Id', 'text', 'meta.Normalized Score', 'meta.Binary', 'meta.Annotations']],
    on=['Id', 'text'],
    how='left'
)

df_wiki['ground_truth'] = df_wiki['meta.Binary'].map({
    -1: 'impolite',
    0: 'neutral',
    1: 'polite',
})


# calculate annotator agreement

# Convert meta.Normalized Score to float
df_wiki['meta.Normalized Score'] = pd.to_numeric(df_wiki['meta.Normalized Score'])

# first, obtain the thresholds for binarization, looking at cutoffs in meta.Normalized Score for all meta.Binary column values
print("\nAnalyzing thresholds for binarization...")
thresholds_df = df_wiki[['meta.Normalized Score', 'meta.Binary']].drop_duplicates()
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
   df_wiki[f'{col}_normalized'] = stats.zscore(df_wiki[col])

# Check if average of normalized scores corresponds to meta.Normalized Score
avg_normalized = df_wiki[[f'{col}_normalized' for col in score_cols]].mean(axis=1)
print(f"Correlation between avg_normalized and meta.Normalized Score: {avg_normalized.corr(df_wiki['meta.Normalized Score']):.4f}")

# next, binarize each of 'Score1', 'Score2', 'Score3', 'Score4', 'Score5'
for col in score_cols:
   df_wiki[f'{col}_binarized'] = df_wiki[f'{col}_normalized'].apply(
       lambda x: -1 if x < impolite_neutral_threshold 
       else 1 if x > neutral_polite_threshold 
       else 0
   )

# then, calculate krippendorff alpha for all 5 binarized annotation scores
# Prepare data for Krippendorff's alpha
annotation_matrix_full_dataset = df_wiki[[f'{col}_binarized' for col in score_cols]].T.values

# Calculate Krippendorff's alpha
krippendorff_alpha_full_dataset = krippendorff.alpha(reliability_data=annotation_matrix_full_dataset, level_of_measurement='nominal')
print(f'krippendorff alpha full dataset: {krippendorff_alpha_full_dataset:.4f}')





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
df_wiki = df_wiki.merge(df_with_features, on=['text'], how='left')

aa = df_wiki.drop_duplicates(subset=['text'], keep='first')
aaa = aa[aa['Politeness'].notna()]
aaa = df_wiki[(df_wiki['Politeness'].notna()) & (df_wiki['Politeness'] ==
                                                 df_wiki['ground_truth'])].drop_duplicates(subset=['text'], keep='first')
print(
    f'Notes: This reduces the number of datapoints from {len(aa)} to {len(aaa)}')

df_wiki = df_wiki[(df_wiki['Politeness'].notna()) & (
    df_wiki['Politeness'] == df_wiki['ground_truth'])]


# test normalized score distribution (i.e., columns 'Normalized Score' and then also 'meta.Normalized Score) for each ground truth score
print("\nNormalized Score distribution by ground truth:")
for gt in ['impolite', 'neutral', 'polite']:
    subset = df_wiki[df_wiki['ground_truth'] == gt]['Normalized Score']
    if not subset.empty:
        print(
            f"  Ground truth {gt}: mean={subset.mean():.3f}, std={subset.std():.3f}, count={len(subset)}")

print("\n'meta.Normalized Score' column:")
for gt in ['impolite', 'neutral', 'polite']:
    subset = df_wiki[df_wiki['ground_truth'] == gt]['meta.Normalized Score']
    if not subset.empty:
        print(
            f"  Ground truth {gt}: mean={subset.mean():.3f}, std={subset.std():.3f}, count={len(subset)}")


df_final = process_full_dataset('politeness_wiki', dataset=df_wiki, save_to_disk=False)


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

# finally, remove unnecessary columns
df_wiki.rename(columns={'Community_x': 'Community'}, inplace=True)

columns_to_drop = ['Request_y', 'meta.Binary', 'Community_y', 'Normalized Score', 'meta.Annotations']
# Only drop columns that actually exist
columns_to_drop = [col for col in columns_to_drop if col in df_wiki.columns]
df_wiki.drop(columns=columns_to_drop, inplace=True)

print(f"\nFinal dataset shape: {df_wiki.shape}")
print(f"Final columns: {list(df_wiki.columns)}")
print(
    f"Ground truth distribution: {df_wiki['ground_truth'].value_counts().sort_index()}")

df_wiki.reset_index(drop=True, inplace=True)

# sort df_wiki columns to start with 'ground_truth', then 'text', and then all other columns
priority_columns = ['ground_truth', 'text']
other_columns = [col for col in df_wiki.columns if col not in priority_columns]
df_wiki = df_wiki[priority_columns + other_columns]

df_wiki.to_csv('data/all_data_processed_full/politeness_wiki.csv', index=False)
