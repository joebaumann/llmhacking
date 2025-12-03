import pdb
import pandas as pd
import pyreadstat


def load_data():
    data, meta = pyreadstat.pyreadstat.read_sav(
        "data/issue_survey/data_raw/BES2024_W29_Panel_v29.1.sav")
    dataset = pd.DataFrame(data)  # Convert to pandas DataFrame
    # Extract metadata (specific to pyreadstat)
    column_names_to_labels = meta.column_names_to_labels  # Maps var name to label
    # Maps var name to {code: label} dict
    variable_value_labels = meta.variable_value_labels

    # 2026 version with open ended response data
    data_text, meta_text = pyreadstat.pyreadstat.read_sav(
        "data/issue_survey/data_raw/BES2024_W26Strings_v26.0.sav", encoding='latin1')
    # data_text, meta_text = pyreadstat.pyreadstat.read_dta("data/issue_survey/data_raw/BES2024_W26Strings_v26.0.dta")
    dataset_text = pd.DataFrame(data_text)  # Convert to pandas DataFrame
    # Extract metadata (specific to pyreadstat)
    column_names_to_labels_text = meta_text.column_names_to_labels  # Maps var name to label
    # Maps var name to {code: label} dict
    variable_value_labels_text = meta_text.variable_value_labels

    # merge both datasets
    df = dataset_text.merge(dataset, on='id', how='left')

    # Get MII_text columns
    # format: (wave, mii issue stated columns, mii open text column, mii ground truth category column)
    mii_cols = [(wave+1, f'miiW{wave+1}', f'MII_textW{wave+1}',
                f'mii_catW{wave+1}') for wave in range(25)]
    # mii_cols = [(wave+1, f'MII_textW{wave+1}', f'miiW{wave+1}', f'mii_catW{wave+1}') for wave in range(25)]
    # if mii issue stated column == 1.0 means that issue was stated and so mii open text column and mii ground truth category column are available


    # Define demographic and political columns to keep for groupings
    demographic_cols = [
        # Core Demographics
        'gender', 'ageGroupW1', 'p_educationW1', 'p_edlevelW1',
        'p_gross_personalW1', 'p_gross_householdW1', 'p_socgradeW1',
        'p_work_statW1', 'p_job_sectorW1', 'p_housingW1', 'p_maritalW1',

        # Geographic
        'countryW1', 'gorW1', 'new_pconW1',

        # Identity & Background
        'p_religionW1', 'p_country_birthW1', 'p_ethnicityW1',
        'p_disabilityW1', 'p_sexualityW1',

        # Political Behavior & Attitudes
        'partyIdW1', 'partyIdStrengthW1', 'generalElectionVoteW1',
        'p_past_vote_2019', 'p_past_vote_2017', 'p_eurefvote', 'p_eurefturnout',
        'polAttentionW1', 'leftRightW1',

        # Economic Attitudes
        'redistSelfW1', 'econPersonalRetroW1', 'econGenRetroW1',

        # Identity Scales
        'britishnessW1', 'scottishnessW1', 'welshnessW1', 'englishnessW1',

        # EU Attitudes
        'EUIntegrationSelfW1', 'euRefVoteW1',

        # Media Consumption
        'p_paper_readW1',
    ]

    # Filter demographic columns to only those that exist in the dataset
    available_demographic_cols = [
        col for col in demographic_cols if col in df.columns]
    print(
        f"Keeping {len(available_demographic_cols)} demographic/political columns")
    # print(f"{len(demographic_cols) - len(available_demographic_cols)} demographic/political columns not available")

    # Prepare list to collect long format data
    long_data_list = []

    for wave_num, mii_binary_col, mii_text_col, mii_cat_col in mii_cols:
        # Extract relevant columns for this wave (MII columns + demographics + id)
        cols_to_extract = ['id', mii_binary_col, mii_text_col,
                        mii_cat_col] + available_demographic_cols
        wave_data = df[cols_to_extract].copy()
        # Add wave number
        wave_data['wave'] = wave_num
        # Rename MII columns to standard names
        wave_data = wave_data.rename(columns={
            mii_binary_col: 'mii_issue_stated_binary',
            mii_text_col: 'text',
            mii_cat_col: 'ground_truth'
        })
        # Reorder columns (MII columns first, then demographics)
        mii_columns = ['id', 'wave',
                    'mii_issue_stated_binary', 'text', 'ground_truth']
        final_columns = mii_columns + available_demographic_cols
        wave_data = wave_data[final_columns]
        long_data_list.append(wave_data)

    # Concatenate all waves
    df_long = pd.concat(long_data_list, ignore_index=True)
    del long_data_list

    # drop all rows where mii_issue_stated_binary != 1.0 or text or ground truth empty or None or '__NA__'
    df_long = df_long[
        (df_long['mii_issue_stated_binary'] == 1.0) &
        (df_long['text'].notna()) &
        (df_long['text'] != '__NA__') &
        (df_long['text'] != '') &
        (df_long['ground_truth'].notna())
    ]

    # Reset index after filtering
    df_long = df_long.reset_index(drop=True)

    print(f"df_long shape: {df_long.shape}")
    print(f"Number of valid MII responses: {len(df_long)}")

    cols_to_keep = []


    print("""    def get_dataset_specific_groups(self, df):
            return \\""")

    for c in demographic_cols:
        if c in df_long:
            value_mapping = variable_value_labels[c]
            df_long[c] = df_long[c].map(value_mapping)
            if c in column_names_to_labels:
                new_col_name = column_names_to_labels[c].lower().replace("'", '').replace(",", '').replace(":", '').replace(
                    ")", '').replace("(", '').replace("-", ' ').replace("?", '').replace("!", '').replace("/", '').replace("\\", '')
                new_col_name = new_col_name.strip()
                df_long.rename(columns={c: new_col_name}, inplace=True)
                df_long[new_col_name] = df_long[new_col_name].apply(
                    lambda x: str(x) if pd.notna(x) else x)
            else:
                new_col_name = c
                new_col_name = new_col_name.strip()
            cols_to_keep.append(new_col_name)
            print(
                f"""            self.split_by_column_with_n_most_frequent_values(df, "{new_col_name}") + \\""")
            # print(f'   {list(df_long[new_col_name].unique())}')
        else:
            pdb.set_trace()


    print("\nFirst few rows:")
    print(df_long.head(10))
    print("\nSample of data types:")
    print(df_long.dtypes)

    # exlude ground truth code 46 (uncoded)
    df_long = df_long[df_long['ground_truth'] != 46.0]


    ground_truth_mapping = {
        15: 'Europe',
        12: 'Immigration',
        26: 'Economy-general',
        27: 'Economy-personal',
        28: 'Unemployment',
        29: 'Taxation',
        30: 'Debt/deficit',
        31: 'Inflation',
        32: 'Living costs',
        1: 'Health',
        48: 'Coronavirus',
        49: 'COVID-economy',
        11: 'Terrorism',
        33: 'Poverty',
        35: 'Inequality',
        36: 'Housing',
        40: 'Environment',
        2: 'Education',
        10: 'Welfare',
        34: 'Austerity',
        37: 'Social care',
        39: 'Transport/infrastructure',
        38: 'Pensions/ageing',
        4: 'Pol-neg',
        5: 'Partisan-neg',
        6: 'Societal divides',
        7: 'Morals',
        8: 'National identity, goals-loss',
        9: 'Racism/discrimination',
        13: 'Asylum',
        14: 'Crime',
        21: 'Foreign affairs',
        22: 'War',
        23: 'Defence',
        41: 'Pol values-authoritarian',
        42: 'Pol values-liberal',
        50: 'Gender/sexuality/family',
        43: 'Pol values-right',
        44: 'Pol values-left',
        3: 'Election outcome',
        16: 'Constitutional',
        17: 'International trade',
        18: 'Devolution',
        19: 'Scot-independence',
        24: 'Foreign emergency',
        25: 'Domestic emergency',
        45: 'Other',
        47: 'Referendum unspecified',
    }


    missing_keys = set(df_long['ground_truth'].unique()) - \
        set(ground_truth_mapping.keys())
    if missing_keys:
        print(f"Warning: Found unmapped ground truth values: {missing_keys}")
        print("Available ground truth values in data:",
            sorted(df_long['ground_truth'].unique()))
        print("Mapped ground truth values:", sorted(ground_truth_mapping.keys()))
        # # Either add these to your mapping or filter them out
        # df_long = df_long[df_long['ground_truth'].isin(ground_truth_mapping.keys())]
        pdb.set_trace()


    df_long['ground_truth'] = df_long['ground_truth'].map(ground_truth_mapping)


    # Filter out rows where text and ground_truth are the same
    df_long = df_long[
        df_long['text'].astype(str).str.lower().str.strip() !=
        df_long['ground_truth'].astype(str).str.lower().str.strip()
    ]


    print(f"\nNumber of valid MII responses: {len(df_long)}")
    print(len(df_long['ground_truth'].unique()),
        'unique ground truth values:', df_long['ground_truth'].unique())

    cols_to_keep += ['id', 'text', 'ground_truth', 'wave']

    df_final = df_long[cols_to_keep]


    print(f"\nFinal df_final dataset shape: {df_final.shape}")
    print(f"Final columns: {list(df_final.columns)}")
    print(
        f"Ground truth distribution: {df_final['ground_truth'].value_counts().sort_index()}")

    for c in df_final.columns:
        if c != 'text':
            print(f'\n--{c}--\n{df_final[c].value_counts().sort_index()}')


    df_final.reset_index(drop=True, inplace=True)

    # sort df_final columns to start with 'ground_truth', then 'text', and then all other columns
    priority_columns = ['ground_truth', 'text']
    other_columns = [
        col for col in df_final.columns if col not in priority_columns]
    df_final = df_final[priority_columns + other_columns]

    df_final.rename(columns={'id': 'id_original'}, inplace=True)
    df_final['original_id_and_wave'] = 'id_' + df_final['id_original'].astype(str) + '_wave_' + df_final['wave'].astype(str)

    return df_final


if __name__ == "__main__":
    df_final = load_data()
    df_final.to_csv('data/all_data_processed_full/issue_survey.csv', index=False)

