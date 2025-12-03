import pandas as pd


df = pd.read_csv("data/stance_climate/data_raw/GWStance/GWSD.tsv", sep='\t')
df.rename(columns={'sentence': 'text'}, inplace=True)
df['av_rating'] = df[['worker_0', 'worker_1', 'worker_2', 'worker_3', 'worker_4', 'worker_5', 'worker_6', 'worker_7']].replace({
    'disagrees': -1,
    'neutral': 0,
    'agrees': 1,
}).sum(axis=1)/8


def get_ground_truth(row):
    if pd.isna(row['disagree']) or pd.isna(row['agree']) or pd.isna(row['neutral']):
        if row['text'] == 'Global warming is happening and it will be dangerous to human health and welfare.':
            return 'agree'
        elif row['text'] == 'Global warming is a hoax.':
            return 'disagree'
        elif row['text'] == 'Some icebergs are cute.':
            return 'neutral'
        elif row['text'] == 'Over the past several years, the United States has seen an increase in business growth that has counteracted the lingering effects of the recession.':
            return 'neutral'
        else:
            # The sentence 'Alarming levels of sea level rise are predicted to threaten Florida over the next decades.' is ambiguously annotated. It's unclear if ground truth should be agrees or neutral. Thus, we omit it.
            return None
    else:
        if row[['disagree', 'agree', 'neutral']].max(skipna=True) >= 0.625:
            return row[['disagree', 'agree', 'neutral']].idxmax(skipna=True)
        else:
            return None


df['ground_truth'] = df.apply(lambda row: get_ground_truth(row), axis=1)
df = df.dropna(subset=['ground_truth'])


# df_meta1 = pd.read_pickle('data/stance_climate/data_raw/GWStance/1_data_scraping/output/dedup_combined_df_2000_1_1_to_2020_4_12.pkl')
# df_meta2 = pd.read_pickle('data/stance_climate/data_raw/GWStance/1_data_scraping/output/filtered_dedup_combined_df_2000_1_1_to_2020_4_12.pkl')
# df_meta3 = pd.read_pickle('data/stance_climate/data_raw/GWStance/1_data_scraping/output/temp_combined_df_2000_1_1_to_2020_4_12.pkl')
# df_meta = pd.concat([df_meta1, df_meta2, df_meta3])
# df_meta.drop_duplicates(subset=['guid'], keep='first', inplace=True)
# # this merge does not work, since guid do not match
# # df = df.merge(df_meta, on=['guid'], how='left')


columns_to_drop = ['in_held_out_test', 'Unnamed: 0']
# Only drop columns that actually exist
columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df.drop(columns=columns_to_drop, inplace=True)

print(f"\nFinal dataset shape: {df.shape}")
print(f"Final columns: {list(df.columns)}")
print(
    f"Ground truth distribution: {df['ground_truth'].value_counts(dropna=False)}")

df.reset_index(drop=True, inplace=True)

# sort df columns to start with 'ground_truth', then 'text', and then all other columns
priority_columns = ['ground_truth', 'text']
other_columns = [
    col for col in df.columns if col not in priority_columns]
df = df[priority_columns + other_columns]

df.to_csv('data/all_data_processed_full/stance_climate.csv', index=False)
