from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pdb
import numpy as np
import krippendorff
from tqdm import tqdm
from utils import sort_strings_substrings_last
import nltk
nltk.download('wordnet')
# from nltk.tokenize import word_tokenize


tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


df_explicit = pd.read_csv(
    "data/hatespeech/data_raw/implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv", delimiter='\t')
df_explicit.rename(columns={'post': 'text'}, inplace=True)
df_explicit.rename(columns={'class': 'ground_truth'}, inplace=True)
df_explicit['ground_truth'] = df_explicit['ground_truth'].replace({
    'explicit_hate': 'explicit hate',
    'implicit_hate': 'implicit hate',
    'not_hate': 'not hate',
})


### estimate annotator agreement ###

# Given information
n_tweets = len(df_explicit)
# reported agreement
perfect_agreement_rate = 0.456
majority_agreement_rate = 0.953
n_annotators_per_tweet = 3

# Label distribution on full data
value_counts = df_explicit['ground_truth'].value_counts(dropna=False)
label_counts = value_counts.values
labels = list(range(len(value_counts)))
p = np.array(label_counts) / n_tweets

alphas = []

for seed in tqdm(range(10000), desc='Simulating krippendorff alpha.', total=10000):
    rng = np.random.RandomState(seed)
    
    # Generate annotations for all tweets
    annotations = []
    
    for _ in range(n_tweets):
        # Decide agreement pattern
        rand = rng.random()
        
        if rand < perfect_agreement_rate:
            # Perfect agreement: all 3 annotators choose same label
            label = rng.choice(labels, p=p)
            tweet_annotations = [label, label, label]
        
        elif rand < majority_agreement_rate:
            # Majority agreement: 2 annotators agree, 1 disagrees
            majority_label = rng.choice(labels, p=p)
            # Choose a different label for the dissenting annotator
            other_labels = [l for l in labels if l != majority_label]
            # Weight disagreement by label frequency
            other_probs = [p[l] for l in other_labels]
            other_probs = np.array(other_probs) / np.sum(other_probs)
            minority_label = rng.choice(other_labels, p=other_probs)
            
            # Randomly arrange the annotations
            tweet_annotations = [majority_label, majority_label, minority_label]
            rng.shuffle(tweet_annotations)
        
        else:
            # No majority: all 3 annotators choose different labels
            tweet_annotations = list(labels)
            rng.shuffle(tweet_annotations)
        
        annotations.append(tweet_annotations)
    
    # Convert to format for Krippendorff's alpha
    # Each row is an annotator, each column is a tweet
    reliability_data = np.array(annotations).T
    
    alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
    alphas.append(alpha)

print(f"Average Krippendorff's alpha: {np.mean(alphas):.4f} (±{np.std(alphas):.4f})")
print(f"Min: {np.min(alphas):.4f}, Max: {np.max(alphas):.4f}")


df_implicit = pd.read_csv(
    "data/hatespeech/data_raw/implicit-hate-corpus/implicit_hate_v1_stg2_posts.tsv", delimiter='\t')
df_implicit.rename(columns={'post': 'text'}, inplace=True)
df_implicit.rename(columns={'implicit_class': 'ground_truth'}, inplace=True)
df_implicit = df_implicit[df_implicit['ground_truth'] != 'other']
df_implicit = df_implicit[df_implicit['extra_implicit_class'].isna()]
df_implicit.drop(columns=['extra_implicit_class'], inplace=True)

df_implicit['ground_truth'] = df_implicit['ground_truth'].replace({
    'white_grievance': 'white grievance',
    'incitement': 'incitement',
    'inferiority': 'inferiority',
    'irony': 'irony',
    'stereotypical': 'stereotypical',
    'threatening': 'threatening',
})

df_target = pd.read_csv(
    "data/hatespeech/data_raw/implicit-hate-corpus/implicit_hate_v1_stg3_posts.tsv", delimiter='\t')
df_target.rename(columns={'post': 'text'}, inplace=True)
df_target = df_target[df_target['target'].notna()]
df_target['target'].value_counts(dropna=False)

all_targets = list(df_target['target'].unique())


def clean_target(target):
    target = target.strip().lower().replace(
        '-', ' ').replace('.', '').replace(',', '')
    target = target.replace('person', 'people')
    target = target.replace('folks', 'people')
    target = target.replace('white people, black people',
                            'black and white people')
    target = target.replace(
        'white people and black people', 'black and white people')
    target = target.replace('whites and blacks', 'black and white people')
    target = target.replace('blacks', 'black people')
    target = target.replace('black lives matter',
                            'black lives matter supporters')
    target = target.replace('people people', 'people')
    target = target.replace('liberals and muslims', 'muslims and liberals')
    target = target.replace('muslins', 'muslims')
    target = target.replace('minorites', 'minorities')
    target = target.replace('minority groups', 'minorities')
    target = target.replace('whites', 'white people')
    target = target.replace('mexicans', 'mexican people')
    target = target.replace('asians', 'asian people')
    if target == 'black':
        target = 'black people'
    elif target == 'muslim':
        target = 'muslims'
    elif target == 'gays':
        target = 'gay people'
    elif target == 'gay':
        target = 'gay people'
    elif target == 'arabians':
        target = 'arabs'
    elif target in ['jew', 'jews']:
        target = 'jewish people'
    elif target in ['not specified', '"they" (group is not specified)', 'not specfied']:
        target = 'no specific group'
    elif target in ['illegal aliens', 'illegals', 'immigrants']:
        target = 'illegal immigrants'
    return target


df_target['target'] = df_target['target'].apply(clean_target)

all_targets = set(all_targets + list(df_target['target'].unique()))

# sort by substrings last
all_targets = sort_strings_substrings_last(all_targets)


def remove_group_info(target, all_targets):
    target = target.lower().strip()
    for i in all_targets:
        target = target.replace(i.lower().strip(), '')
    # tokenize
    # target = word_tokenize(target)
    target = tokenizer.tokenize(target)
    # remove stopwords
    target_no_stopwords = [token for token in target if token.isalpha(
    ) and token not in stop_words and len(token) > 1]
    if len(target_no_stopwords) > 0:
        target = target_no_stopwords
    # lemmatize
    target = set(sorted([lemmatizer.lemmatize(t) for t in target]))
    return target


df_target['implied_statement_without_target_info'] = df_target['implied_statement'].apply(
    lambda x: remove_group_info(x, all_targets))

df_target.rename(columns={'target': 'ground_truth'}, inplace=True)

# filter for max 26 classes
all_gt_classes = list(df_target['ground_truth'].value_counts(
    dropna=False).head(26).keys())
all_gt_classes = sort_strings_substrings_last(all_gt_classes)
# print dict used for mapping LLM output to class
print('Use the following mapping for the prompt:')
print('dict_with_mapping_options = {')
alphabet_letters_capital = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
assert len(all_gt_classes) <= len(list(alphabet_letters_capital)
                                  ), "too many values in all_gt_classes"
for idx, c in enumerate(all_gt_classes):
    print(f"    '{alphabet_letters_capital[idx]}': '{c}',")
print('}')

df_target = df_target[df_target['ground_truth'].isin(all_gt_classes)]


print(f"\nFinal df_explicit dataset shape: {df_explicit.shape}")
print(f"Final columns: {list(df_explicit.columns)}")
print(
    f"Ground truth distribution: {df_explicit['ground_truth'].value_counts().sort_index()}")

df_explicit.reset_index(drop=True, inplace=True)

# sort df_explicit columns to start with 'ground_truth', then 'text', and then all other columns
priority_columns = ['ground_truth', 'text']
other_columns = [
    col for col in df_explicit.columns if col not in priority_columns]
df_explicit = df_explicit[priority_columns + other_columns]

df_explicit.to_csv(
    'data/all_data_processed_full/hatespeech_explicit.csv', index=False)


print(f"\nFinal df_implicit dataset shape: {df_implicit.shape}")
print(f"Final columns: {list(df_implicit.columns)}")
print(
    f"Ground truth distribution: {df_implicit['ground_truth'].value_counts().sort_index()}")

df_implicit.reset_index(drop=True, inplace=True)

# sort df_implicit columns to start with 'ground_truth', then 'text', and then all other columns
priority_columns = ['ground_truth', 'text']
other_columns = [
    col for col in df_implicit.columns if col not in priority_columns]
df_implicit = df_implicit[priority_columns + other_columns]

df_implicit.to_csv(
    'data/all_data_processed_full/hatespeech_implicit.csv', index=False)


print(f"\nFinal df_target dataset shape: {df_target.shape}")
print(f"Final columns: {list(df_target.columns)}")
print(
    f"Ground truth distribution: {df_target['ground_truth'].value_counts().sort_index()}")

df_target.reset_index(drop=True, inplace=True)

# sort df_target columns to start with 'ground_truth', then 'text', and then all other columns
priority_columns = ['ground_truth', 'text']
other_columns = [
    col for col in df_target.columns if col not in priority_columns]
df_target = df_target[priority_columns + other_columns]

df_target.to_csv('data/all_data_processed_full/hatespeech_target.csv', index=False)
