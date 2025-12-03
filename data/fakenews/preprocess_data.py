import pandas as pd
import numpy as np
import json
import glob
from tqdm import tqdm
import os
import pdb
import pickle
from tqdm import tqdm
import tldextract


df_gos_fake = pd.read_csv(
    'data/fakenews/data_raw/FakeNewsNet/dataset/gossipcop_fake.csv')
df_gos_fake['ground_truth'] = 1  # 'fake'
df_gos_real = pd.read_csv(
    'data/fakenews/data_raw/FakeNewsNet/dataset/gossipcop_real.csv')
df_gos_real['ground_truth'] = 0  # 'real'
df_gos = pd.concat([df_gos_fake, df_gos_real])
df_pol_fake = pd.read_csv(
    'data/fakenews/data_raw/FakeNewsNet/dataset/politifact_fake.csv')
df_pol_fake['ground_truth'] = 1  # 'fake'
df_pol_real = pd.read_csv(
    'data/fakenews/data_raw/FakeNewsNet/dataset/politifact_real.csv')
df_pol_real['ground_truth'] = 0  # 'real'
df_pol = pd.concat([df_pol_fake, df_pol_real])


df_news_content_gossipcop_test = pickle.load(open(
    'data/fakenews/data_raw/SheepDog/data/news_articles/gossipcop_test.pkl', 'rb'))
df_news_content_gossipcop_test = pd.DataFrame({
    'text': df_news_content_gossipcop_test['news'],
    'ground_truth': df_news_content_gossipcop_test['labels']
})
df_news_content_gossipcop_test['source'] = 'gossipcop'

df_news_content_gossipcop_train = pickle.load(open(
    'data/fakenews/data_raw/SheepDog/data/news_articles/gossipcop_train.pkl', 'rb'))
df_news_content_gossipcop_train = pd.DataFrame({
    'text': df_news_content_gossipcop_train['news'],
    'ground_truth': df_news_content_gossipcop_train['labels']
})
df_news_content_gossipcop_train['source'] = 'gossipcop'

df_news_content_politifact_test = pickle.load(open(
    'data/fakenews/data_raw/SheepDog/data/news_articles/politifact_test.pkl', 'rb'))
df_news_content_politifact_test = pd.DataFrame({
    'text': df_news_content_politifact_test['news'],
    'ground_truth': df_news_content_politifact_test['labels']
})
df_news_content_politifact_test['source'] = 'politifact'

df_news_content_politifact_train = pickle.load(open(
    'data/fakenews/data_raw/SheepDog/data/news_articles/politifact_train.pkl', 'rb'))
df_news_content_politifact_train = pd.DataFrame({
    'text': df_news_content_politifact_train['news'],
    'ground_truth': df_news_content_politifact_train['labels']
})
df_news_content_politifact_train['source'] = 'politifact'

# data/fakenews/data_raw/SheepDog/data/news_articles/lun_test.pkl
# data/fakenews/data_raw/SheepDog/data/news_articles/lun_train.pkl


def match_with_metadata(df, df_meta):
    """
    Match each row in df with metadata by checking if any of the titles (lowercase) 
    in df_meta is a substring of the text (lowercase).
    Only consider df_meta rows where the ground_truth value matches.
    Merge df_meta on df in case of match.
    Ask for user input in case of multiple matches.
    """
    # Create a copy of df to avoid modifying the original
    df_matched = df.copy()

    # Initialize columns for metadata
    for col in df_meta.columns:
        if col not in df_matched.columns:
            df_matched[col] = np.nan
        else:
            print(f"  col '{col}' already in df")

    # Track matching statistics
    match_stats = []
    matches_found = 0
    multiple_matches = 0
    no_matches = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Matching df with metadata'):
        text_lower = str(row['text']).lower()
        ground_truth = row['ground_truth']

        # Find potential matches: same ground_truth and title is substring of text
        potential_matches = df_meta[
            (df_meta['ground_truth'] == ground_truth) &
            (df_meta['title'].str.lower().apply(lambda x: str(
                x) in text_lower if pd.notna(x) else False))
        ]

        match_stats.append(len(potential_matches))

        if len(potential_matches) == 0:
            no_matches += 1
            continue
        elif len(potential_matches) == 1:
            # Single match - merge the metadata
            matches_found += 1
            for col in df_meta.columns:
                if col not in ['ground_truth']:
                    df_matched.loc[idx, col] = potential_matches.iloc[0][col]
        else:
            # Multiple matches - ask user for input
            multiple_matches += 1
            # print(f"\nRow {idx}: Multiple matches found for text beginning with:")
            # print(f"'{text_lower[:100]}...'")
            # print(f"Ground truth: {ground_truth}")
            # print("\nPotential matches:")
            # for i, (_, match_row) in enumerate(potential_matches.iterrows()):
            #     print(f"{i}: {match_row['title']}")

            # while True:
            #     try:
            #         choice = input(f"Enter choice (0-{len(potential_matches)-1}, or 's' to skip): ")
            #         if choice.lower() == 's':
            #             print("Skipping this match.")
            #             break
            #         choice_idx = int(choice)
            #         if 0 <= choice_idx < len(potential_matches):
            #             selected_match = potential_matches.iloc[choice_idx]
            #             for col in df_meta.columns:
            #                 df_matched.loc[idx, col] = selected_match[col]
            #             print(f"Selected: {selected_match['title']}")
            #             break
            #         else:
            #             print(f"Please enter a number between 0 and {len(potential_matches)-1}")
            #     except ValueError:
            #         print("Please enter a valid number or 's' to skip")

    print(f"\nMatching completed:")
    print(f"- Single matches found: {matches_found}")
    print(f"- Multiple matches requiring user input: {multiple_matches}")
    print(f"- No matches found: {no_matches}")
    print(f"- Total rows processed: {len(df)}")

    return df_matched, match_stats


df_news_content_gossipcop_test_matched, match_stats1 = match_with_metadata(
    df_news_content_gossipcop_test, df_gos)
df_news_content_gossipcop_train_matched, match_stats2 = match_with_metadata(
    df_news_content_gossipcop_train, df_gos)
df_news_content_politifact_test_matched, match_stats3 = match_with_metadata(
    df_news_content_politifact_test, df_pol)
df_news_content_politifact_train_matched, match_stats4 = match_with_metadata(
    df_news_content_politifact_train, df_pol)


df = pd.concat([df_news_content_gossipcop_test_matched, df_news_content_gossipcop_train_matched,
               df_news_content_politifact_test_matched, df_news_content_politifact_train_matched])


df['ground_truth'] = df['ground_truth'].replace({
    0: 'real',
    1: 'fake',
})


def get_nr_of_tweets_sharing_the_news(tweet_ids):
    if pd.isna(tweet_ids):
        return None
    else:
        return len(tweet_ids.split('\t'))


df['nr_of_tweets_sharing_the_news'] = df['tweet_ids'].apply(
    get_nr_of_tweets_sharing_the_news)


def get_domain_from_url(url):
    if url is None or pd.isna(url):
        return None
    domain = None
    dom = tldextract.extract(url)
    domain_pre = dom.subdomain.lower().replace(
        'www', '').strip() if dom.subdomain else ''
    domain_main = dom.domain.lower().strip()
    domain = f"{domain_pre}.{domain_main}" if len(
        domain_pre) > 0 else domain_main
    return domain  # , full_domain


df['domain'] = df['news_url'].apply(get_domain_from_url)


columns_to_drop = ['tweet_ids']
# Only drop columns that actually exist
columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df.drop(columns=columns_to_drop, inplace=True)

print(f"\nFinal dataset shape: {df.shape}")
print(f"Final columns: {list(df.columns)}")
print(
    f"Ground truth distribution: {df['ground_truth'].value_counts().sort_index()}")

df.reset_index(drop=True, inplace=True)

# sort df columns to start with 'ground_truth', then 'text', and then all other columns
priority_columns = ['ground_truth', 'text']
other_columns = [
    col for col in df.columns if col not in priority_columns]
df = df[priority_columns + other_columns]

df.to_csv('data/all_data_processed_full/fakenews.csv', index=False)
