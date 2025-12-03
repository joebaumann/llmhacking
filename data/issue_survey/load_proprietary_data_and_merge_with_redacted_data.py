from data.issue_survey.preprocess_data import *

df_REDACTED = pd.read_csv(f'data/all_data_processed/essay_issue_survey_REDACTED.csv')
df = load_data()
cols_to_keep = [c for c in df.columns if c != 'ground_truth']
df = df[cols_to_keep]
df_merged = df_REDACTED.merge(df, on='original_id_and_wave', how='left')
df_merged.to_csv(f'data/all_data_processed/essay_issue_survey.csv', index=False)
