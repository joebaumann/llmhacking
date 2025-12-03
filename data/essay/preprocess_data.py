import numpy as np
import pandas as pd
import re
import yaml
import munch
import copy
import pdb
import json
import pyreadstat
from pathlib import Path
from striprtf.striprtf import rtf_to_text

def load_data():
    # load metadata and annotations
    data, meta = pyreadstat.pyreadstat.read_sav(
        "data/essay/data_raw/UKDA-5790-spss/spss/spss28/ncds_essays_subset.sav", encoding='latin1')
    dataset = pd.DataFrame(data)
    column_names_to_labels = meta.column_names_to_labels
    variable_value_labels = meta.variable_value_labels

    dataset = dataset[dataset['Status'] == 1]

    all_ids = [i.upper() for i in list(dataset['ncdsid'].unique())]


    def parse_rtf_files(directory='', files_to_skip=[]):
        """
        Parse all RTF files in a directory and extract structured data.

        Args:
            directory (str): Directory path containing RTF files

        Returns:
            pd.DataFrame: DataFrame with parsed data from all RTF files
        """

        # List to store all parsed data
        data_list = []
        skipped_data = []

        # Find all .rtf files in the directory
        rtf_files = list(Path(directory).glob('**/*.rtf'))

        print(f"Found {len(rtf_files)} RTF files")

        for file_path in rtf_files:
            try:
                # Extract information from filename
                filename = file_path.name

                # Parse filename pattern: ncdsessay-{id}-{ncdsid}.rtf
                filename_match = re.match(
                    r'ncdsessay-(\d+)-([^.]+)\.rtf', filename)

                if not filename_match:
                    print(f"Warning: Unexpected filename format: {filename}")
                    pdb.set_trace()
                    continue

                id_from_filename = filename_match.group(1)
                ncdsid_from_filename = filename_match.group(2)

                # Read file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        text = rtf_to_text(content)
                except UnicodeDecodeError:
                    # Try with different encoding if UTF-8 fails
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                        text = rtf_to_text(content)
                    pdb.set_trace()

                if ncdsid_from_filename.upper() not in all_ids or any(f in str(file_path) for f in files_to_skip):
                    skipped_data.append({
                        'filename': filename,
                        'id': id_from_filename,
                        'ncdsid': ncdsid_from_filename,
                        'text_all': text,
                        'file_path': str(file_path),
                        'file_content': content,
                    })
                    continue

                # Parse text structure
                lines = text.strip().split('\n')

                if len(lines) < 3:
                    print(f"Warning: Unexpected text structure in {filename}")
                    pdb.set_trace()
                    continue

                # Extract ncdsid_new (first line)
                ncdsid_new = lines[0].strip()
                last_line = lines[-1]
                if 'word' in last_line.lower().strip():
                    last_line_removed = True
                    word_count = copy.deepcopy(last_line)
                    word_count = word_count.lower().strip().replace(
                        'word count', '').replace('words', '').replace(':', '').strip()
                    lines_with_essay = lines[2:-1]
                else:
                    last_line_removed = False
                    word_count = None
                    lines_with_essay = lines[2:]

                # Check for separator line
                if not lines[1].strip().startswith('---'):
                    print(f"Warning: Missing separator line in {filename}")
                    pdb.set_trace()

                # Extract main text text (everything between separator and "Words:" line)
                text_lines = []

                for i, line in enumerate(lines_with_essay, start=2):
                    if line.strip().startswith('Words:'):
                        break
                    else:
                        text_lines.append(line)

                # Join text
                essay = ' '.join(text_lines).strip()

                # Clean up extra whitespace
                essay = re.sub(r'\s+', ' ', essay)

                # Check if ncdsid matches ncdsid_new
                ncdsid_match = ncdsid_from_filename.upper() == ncdsid_new.upper()

                if not ncdsid_match:
                    print(f"Warning: NCDSID mismatch in {filename}")
                    print(f"  From filename: {ncdsid_from_filename}")
                    print(f"  From text: {ncdsid_new}")
                    pdb.set_trace()

                # Store parsed data
                data_entry = {
                    'filename': filename,
                    'id': id_from_filename,
                    'ncdsid': ncdsid_from_filename,
                    'text': essay,
                    'last_line_removed': last_line_removed,
                    'last_line': last_line,
                    'nr_of_words': word_count,
                    'file_path': str(file_path),
                    'file_content': content,
                }

                data_list.append(data_entry)

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                pdb.set_trace()

        # Create DataFrame
        df = pd.DataFrame(data_list)
        df_skipped = pd.DataFrame(skipped_data)

        # Print summary
        print(f"\nProcessed {len(df)} files successfully")

        return df


    files_to_skip = [
        # empty files
        'ncdsessay-164-n15455l.rtf',
        'ncdsessay-167-n15827u.rtf',
    ]

    directory_path = 'data/essay/data_raw/UKDA-5790-rtf/rtf'
    df = parse_rtf_files(directory_path, files_to_skip)

    variable_value_labels_updated = {}
    for c in dataset.columns:
        if c in variable_value_labels:
            value_mapping = {k: (None if k < 0 else v.strip())
                            for k, v in variable_value_labels[c].items()}
            variable_value_labels_updated[c] = value_mapping
            dataset[c] = dataset[c].map(value_mapping)


    dataset['ncdsid'] = dataset['ncdsid'].apply(lambda x: x.upper())
    df['ncdsid'] = df['ncdsid'].apply(lambda x: x.upper())

    df = df.merge(dataset, on='ncdsid', how='left')


    grouping_cols = [
        'n622c', 'n802', 'n810', 'n811', 'n812', 'n813', 'n910', 'n911', 'n914', 'n917', 'n920', 'n923', 'n926', 'n929', 'n1122', 'n1123', 'n1127', 'n1128', 'n1171', 'n1685', 'n958', 'abil11', 'fambak',
    ]
    cols_to_keep = []

    for c in grouping_cols:
        if c in df:
            if c in column_names_to_labels:
                new_col_name = ' '.join(column_names_to_labels[c].split(
                    ' ')[1:]).lower().replace("'", '').replace(",", '')
                new_col_name = new_col_name.strip()
                df.rename(columns={c: new_col_name}, inplace=True)
                df[new_col_name] = df[new_col_name].astype(str)
            else:
                new_col_name = c
                new_col_name = new_col_name.strip()
            cols_to_keep.append(new_col_name)
        else:
            pdb.set_trace()

    cols_to_keep += ['ncdsid', 'text', 'ground_truth']

    return df, cols_to_keep


def save_df(df, task_name, gt_col, cols_to_keep):

    df = copy.deepcopy(df)

    df.rename(columns={gt_col: 'ground_truth'}, inplace=True)

    df = df[df['ground_truth'].notna()]

    df = df[cols_to_keep]

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Final columns: {list(df.columns)}")
    print(
        f"Ground truth distribution: {df['ground_truth'].value_counts().sort_index()}")

    df.reset_index(drop=True, inplace=True)

    # sort df columns to start with 'ground_truth', then 'text', and then all other columns
    priority_columns = ['ground_truth', 'text']
    other_columns = [col for col in df.columns if col not in priority_columns]
    df = df[priority_columns + other_columns]

    df.to_csv(f'data/all_data_processed_full/essay_{task_name}.csv', index=False)


all_tasks_and_descriptions = [
    ('livsit', 'Living', 'Description of the anticipated living situation at age 25.',
    'Identify the writer\'s anticipated living situation at age 25. If the living situation changes during the essay (e.g., starts single then gets married), classify as "Changes during essay".'),
    ('Housewife', 'Housewife', 'Whether being a housewife is mentioned as an occupation or role.',
    'Determine if the writer expects to be a housewife or care for home and family (instead of or alongside having an occupation).'),
    ('occsoc', 'Worksocial', 'Whether social aspects of work are mentioned.',
    'Determine if social relations at work are mentioned (e.g., working with friends, interactions with boss, talking to colleagues).'),
    ('location', 'Location', 'Geographic location mentioned for living at age 25.',
    'Identify where the writer expects to live at age 25. "UK" includes any specific UK location except London (e.g., Denton). "Abroad" includes any non-UK location (e.g., Spain).'),
    ('domlab', 'Domestic', 'Type of domestic labor discussed.',
    'Identify mentions of domestic labor, whether done by self or others (e.g., "my wife will look after the children", "we will have a maid"). Multiple different tasks should be classified as "Multiple tasks".'),
    ('narrate', 'Narrative', 'Level of narrative structure in the essay.',
    'Assess narrative elements: pure descriptions with no time progression are "No"; essays with sequential events (this happened then that) are "Some narrative clauses"; extended stories about life progression are "Extended narrative".')
]


if __name__ == "__main__":

    df, cols_to_keep = load_data()
    for col_name, task_name, _, _ in all_tasks_and_descriptions:
        task_name = task_name.lower()
        # save processed file
        save_df(df, task_name, col_name, cols_to_keep)
