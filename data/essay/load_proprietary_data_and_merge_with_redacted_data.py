from data.essay.preprocess_data import *

df, cols_to_keep = load_data()

for col_name, task_name, _, _ in all_tasks_and_descriptions:
    task_name = task_name.lower()
    df_REDACTED = pd.read_csv(f'data/all_data_processed/essay_{task_name}_REDACTED.csv')
    df_REDACTED = df_REDACTED[[c for c in df_REDACTED.columns if not c.startswith('Unnamed')]]

    # Create a copy for this task and rename the ground truth column
    df_task = df.copy()
    df_task.rename(columns={col_name: 'ground_truth'}, inplace=True)

    # Keep only needed columns (exclude ground_truth since it comes from REDACTED)
    cols_from_full = [c for c in cols_to_keep if c != 'ground_truth']
    df_task = df_task[cols_from_full]
    
    df_merged = df_REDACTED.merge(df_task, on='ncdsid', how='left')

    # Reorder columns: id, ground_truth, text, [other metadata], ncdsid
    priority_columns = ['id', 'ground_truth', 'text']
    other_columns = [col for col in df_merged.columns if col not in priority_columns and col != 'ncdsid']
    df_merged = df_merged[priority_columns + other_columns + ['ncdsid']]
    
    df_merged.to_csv(f'data/all_data_processed/essay_{task_name}.csv', index=False)
    print(f"Saved essay_{task_name}.csv")
