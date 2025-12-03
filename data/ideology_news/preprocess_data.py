import pandas as pd
import json
import glob
import os
import pdb
from tqdm import tqdm

directory_path = "data/ideology_news/data_raw/Article-Bias-Prediction/data/jsons"

# Get all JSON files in the directory
json_pattern = os.path.join(directory_path, "*.json")
json_files = glob.glob(json_pattern)

if not json_files:
    print(f"No JSON files found in {directory_path}")
    pdb.set_trace()

# List to store all data
all_data = []

# Iterate through each JSON file
for file_path in tqdm(json_files, total=len(json_files), desc='Loading json files.'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue

df = pd.DataFrame(all_data)

df.rename(columns={'content': 'text'}, inplace=True)
df.rename(columns={'bias_text': 'ground_truth'}, inplace=True)

print(
    f"Successfully loaded {len(df)} records from {len(json_files)} JSON files")


def format_date_column(date):
    if pd.isna(date):
        return date  # Keep NaN values as-is
    elif date == '0001-11-30':
        return None
    # If date contains '/', convert from M/D/YY or MM/DD/YY format
    if '/' in str(date):
        try:
            # Parse the date assuming M/D/YY format
            parsed_date = pd.to_datetime(date, format='%m/%d/%y')
            # Return in ISO format
            return parsed_date.strftime('%Y-%m-%d')
        except:
            # If parsing fails, return original value
            print(f'date column transformation failed for {date}')
            return date
    # If date already contains '-', assume it's already in correct format
    return date


# ensure correct format of date column
df['date'] = df['date'].apply(format_date_column)

columns_to_drop = ['bias', 'content_original']
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

df.to_csv('data/all_data_processed_full/ideology_news.csv', index=False)
