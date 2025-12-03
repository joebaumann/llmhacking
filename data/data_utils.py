import pdb
import copy
import re
import pandas as pd
import numpy as np
import yaml
import munch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from difflib import SequenceMatcher
from utils import sort_strings_substrings_last


all_tasks = [
    'politeness_wiki',
    'politeness_stack',
    'stance_climate',
    'ideology_news',
    'fakenews',
    'factuality',
    'hatespeech_explicit',
    'hatespeech_implicit',
    'hatespeech_target',
    'emotion',
    'essay_living',
    'essay_housewife',
    'essay_worksocial',
    'essay_location',
    'essay_domestic',
    'essay_narrative',
    'issue_survey',
    'humor',
    'topic',
    'ideology_tweets',
    'misinfo',
    'tone',
    'manifestos_issue',
    'manifestos_econ_ideology',
    'manifestos_social_ideology',
    'manifestos_issues_detailed',
    'relevance_tweets',
    'framesI_tweets',
    'framesII_tweets',
    'stance_tweets',
    'topic_tweets',
    'relevance_tweets23',
    'framesI_tweets23',
    'relevance_tweets17',
    'framesII_tweets17',
    'relevance_news',
    'framesI_news',
]

def map_dataset_name_to_class(dataset_name):
    if dataset_name == 'politeness_wiki':
        from data.politeness.load_data_and_prompts import PolitenessWikiDataset
        return PolitenessWikiDataset()
    if dataset_name == 'politeness_stack':
        from data.politeness.load_data_and_prompts import PolitenessStackDataset
        return PolitenessStackDataset()
    if dataset_name == 'stance_climate':
        from data.stance_climate.load_data_and_prompts import StanceClimateDataset
        return StanceClimateDataset()
    if dataset_name == 'ideology_news':
        from data.ideology_news.load_data_and_prompts import IdeologyNewsDataset
        return IdeologyNewsDataset()
    if dataset_name == 'fakenews':
        from data.fakenews.load_data_and_prompts import FakeNewsDataset
        return FakeNewsDataset()
    if dataset_name == 'factuality':
        from data.factuality.load_data_and_prompts import FactualityDataset
        return FactualityDataset()
    if dataset_name == 'hatespeech_explicit':
        from data.hatespeech.load_data_and_prompts import HateSpeechExplicitDataset
        return HateSpeechExplicitDataset()
    if dataset_name == 'hatespeech_implicit':
        from data.hatespeech.load_data_and_prompts import HateSpeechImplicitDataset
        return HateSpeechImplicitDataset()
    if dataset_name == 'hatespeech_target':
        from data.hatespeech.load_data_and_prompts import HateSpeechTargetDataset
        return HateSpeechTargetDataset()
    if dataset_name == 'emotion':
        from data.emotion.load_data_and_prompts import EmotionDataset
        return EmotionDataset()
    if dataset_name == 'essay_living':
        from data.essay.load_data_and_prompts import EssayLivingDataset
        return EssayLivingDataset()
    if dataset_name == 'essay_housewife':
        from data.essay.load_data_and_prompts import EssayHousewifeDataset
        return EssayHousewifeDataset()
    if dataset_name == 'essay_worksocial':
        from data.essay.load_data_and_prompts import EssayWorksocialDataset
        return EssayWorksocialDataset()
    if dataset_name == 'essay_location':
        from data.essay.load_data_and_prompts import EssayLocationDataset
        return EssayLocationDataset()
    if dataset_name == 'essay_domestic':
        from data.essay.load_data_and_prompts import EssayDomesticDataset
        return EssayDomesticDataset()
    if dataset_name == 'essay_narrative':
        from data.essay.load_data_and_prompts import EssayNarrativeDataset
        return EssayNarrativeDataset()
    if dataset_name == 'issue_survey':
        from data.issue_survey.load_data_and_prompts import IssueSurveyDataset
        return IssueSurveyDataset()
    if dataset_name == 'humor':
        from data.humor.load_data_and_prompts import HumorDataset
        return HumorDataset()
    if dataset_name == 'topic':
        from data.topic.load_data_and_prompts import TopicDataset
        return TopicDataset()
    if dataset_name == 'ideology_tweets':
        from data.ideology_tweets.load_data_and_prompts import IdeologyTweetsDataset
        return IdeologyTweetsDataset()
    if dataset_name == 'misinfo':
        from data.misinfo.load_data_and_prompts import MisinfoDataset
        return MisinfoDataset()
    if dataset_name == 'tone':
        from data.tone.load_data_and_prompts import ToneDataset
        return ToneDataset()
    if dataset_name == 'manifestos_issue':
        from data.manifestos_uk.load_data_and_prompts import ManifestosUKIssueDataset
        return ManifestosUKIssueDataset()
    if dataset_name == 'manifestos_econ_ideology':
        from data.manifestos_uk.load_data_and_prompts import ManifestosUKEconIdeologyDataset
        return ManifestosUKEconIdeologyDataset()
    if dataset_name == 'manifestos_social_ideology':
        from data.manifestos_uk.load_data_and_prompts import ManifestosUKSocialIdeologyDataset
        return ManifestosUKSocialIdeologyDataset()
    if dataset_name == 'manifestos_issues_detailed':
        from data.manifestos.load_data_and_prompts import ManifestosIssuesDetailedDataset
        return ManifestosIssuesDetailedDataset()
    if dataset_name == 'relevance_tweets':
        from data.gilardi_et_al_pnas.load_data_and_prompts import Gilardi2023_Data1Task1_Tweets_2020_2021
        return Gilardi2023_Data1Task1_Tweets_2020_2021()
    if dataset_name == 'framesI_tweets':
        from data.gilardi_et_al_pnas.load_data_and_prompts import Gilardi2023_Data1Task2_Tweets_2020_2021
        return Gilardi2023_Data1Task2_Tweets_2020_2021()
    if dataset_name == 'framesII_tweets':
        from data.gilardi_et_al_pnas.load_data_and_prompts import Gilardi2023_Data1Task3Tweets_2020_2021
        return Gilardi2023_Data1Task3Tweets_2020_2021()
    if dataset_name == 'stance_tweets':
        from data.gilardi_et_al_pnas.load_data_and_prompts import Gilardi2023_Data1Task4Tweets_2020_2021
        return Gilardi2023_Data1Task4Tweets_2020_2021()
    if dataset_name == 'topic_tweets':
        from data.gilardi_et_al_pnas.load_data_and_prompts import Gilardi2023_Data1Task5Tweets_2020_2021
        return Gilardi2023_Data1Task5Tweets_2020_2021()
    if dataset_name == 'relevance_tweets23':
        from data.gilardi_et_al_pnas.load_data_and_prompts import Gilardi2023_Data2Task1_Tweets2023Relevance
        return Gilardi2023_Data2Task1_Tweets2023Relevance()
    if dataset_name == 'framesI_tweets23':
        from data.gilardi_et_al_pnas.load_data_and_prompts import Gilardi2023_Data2Task2_Tweets2023Frame
        return Gilardi2023_Data2Task2_Tweets2023Frame()
    if dataset_name == 'relevance_tweets17':
        from data.gilardi_et_al_pnas.load_data_and_prompts import Gilardi2023_Data3Task1_CongressTweets_2017_2022_Relevance
        return Gilardi2023_Data3Task1_CongressTweets_2017_2022_Relevance()
    if dataset_name == 'framesII_tweets17':
        from data.gilardi_et_al_pnas.load_data_and_prompts import Gilardi2023_Data3Task2_CongressTweets_2017_2022PoliticalFrame
        return Gilardi2023_Data3Task2_CongressTweets_2017_2022PoliticalFrame()
    if dataset_name == 'relevance_news':
        from data.gilardi_et_al_pnas.load_data_and_prompts import Gilardi2023_Data4Task1_News_2020_2021
        return Gilardi2023_Data4Task1_News_2020_2021()
    if dataset_name == 'framesI_news':
        from data.gilardi_et_al_pnas.load_data_and_prompts import Gilardi2023_Data4Task2_News_2020_2021
        return Gilardi2023_Data4Task2_News_2020_2021()
    else:
        raise ValueError(
            f"Unknown data_name: {dataset_name}. Please add support for this dataset.")


class MyDataLoader:

    def __init__(self, data_directory, config_fn):
        self.data_directory = data_directory
        self.config = self.load_config(data_directory, config_fn)
        self.dataset_name = self.config.dataset_name
        self.data_and_task_name = self.config.data_and_task_name
        self.link_to_dataset = self.config.link_to_dataset
        self.data_type = self.config.data_type
        self.data_description = self.config.data_description
        self.task_description = self.config.task_description
        if hasattr(self.config, 'dataset_sampling_explanation'):
            self.dataset_sampling_explanation = self.config.dataset_sampling_explanation
        self.bibtex_identifier = self.config.bibtex_identifier

        self.output_mappings = self.get_all_output_mappings()
        self.reported_results = self.get_reported_results()

    @staticmethod
    def load_config(data_directory, config_fn):
        config_filepath = f'data/{data_directory}/{config_fn}'
        with open(config_filepath, 'r') as file:
            return munch.munchify(yaml.safe_load(file))

    def format_prompt(self, prompt_text, data):
        assert 'text' in data, f"'text' column is not in data. All columns: {data.columns}"
        # check if of type list
        formatted_prompt = copy.deepcopy(prompt_text)
        if isinstance(formatted_prompt, list):
            formatted_prompt[-1]['content'] = formatted_prompt[-1]['content'].format(
                **data)
        else:
            formatted_prompt = formatted_prompt.format(**data)
        return formatted_prompt

    def format_confidence_elicitation_prompt(self, prompt_text, data, previous_answer):
        assert 'text' in data, f"'text' column is not in data. All columns: {data.columns}"
        # check if of type list
        formatted_prompt = copy.deepcopy(prompt_text)
        formatted_prompt = formatted_prompt.format(
            previous_answer_placeholder=previous_answer,
            **data)
        return formatted_prompt

    def load_dataset(self, processed_data_dir='all_data_processed'):
        """Load the dataset for experiments."""
        dataset = pd.read_csv(
            f'data/{processed_data_dir}/{self.data_and_task_name}.csv',
            low_memory=False)
        dataset = dataset[dataset['ground_truth'].notna()]
        if 'ground_truth' in dataset.columns:
            dataset['ground_truth'] = dataset['ground_truth'].astype(str)
        if 'response_mapped' in dataset.columns:
            dataset['response_mapped'] = dataset['response_mapped'].astype(str)
        return dataset

    def save_final_dataset_to_disk(self, df_final, processed_data_dir='all_data_processed'):
        """Save the final dataset for experiments to disk."""
        fp = f'data/{processed_data_dir}/{self.data_and_task_name}.csv'
        df_final.to_csv(fp, index=False)
        print(f"Processed dataset saved to: {fp}")

    def load_prompts(self):
        """Load the dataset for experiments."""
        all_prompts = self.get_prompts()
        all_unique_prompts = set([p['description'] for p in all_prompts])
        assert len(all_unique_prompts) == len(all_prompts), f'Prompt descriptions must be unique! There are {len(all_prompts)} prompts in total but only {len(all_unique_prompts)} unique prompt descriptions.'
        return all_prompts

    def get_text_from_datapoint(self, datapoint):
        """Extract the text to be classified from a datapoint."""
        return datapoint['text']

    def get_ground_truth_from_datapoint(self, datapoint):
        """Extract the ground truth from a datapoint."""
        return datapoint['ground_truth']

    def load_human_annotations_for_alt_test(self, dataset):
        return None

    def get_all_output_mappings(self):
        raise NotImplementedError(
            "Subclasses must implement get_all_output_mappings method")

    def get_cols_to_stratify(self):
        raise NotImplementedError(
            "Subclasses must implement get_cols_to_stratify method")

    def get_reported_results(self):
        raise NotImplementedError(
            "Subclasses must implement get_reported_results method")

    def default_output_mapping(self, text, mapping_options=None):
        if text is None:
            raise ValueError("Text input cannot be None")
        if mapping_options is None:
            raise ValueError("mapping_options cannot be None")
        label = "na"
        all_found_labels = []
        mapping_options = {k.lower().strip(): v for k,
                        v in mapping_options.items()}
        # reorder mapping options by key to avoid wrong mappings where some keys are substrings of others
        ordered_keys = sort_strings_substrings_last(list(mapping_options.keys()))
        # order the mapping_options dict using ordered_keys
        mapping_options = {k: mapping_options[k] for k in ordered_keys}
        # Check for category keywords in the text
        text_lower = text.lower().strip()
        for letter, category in mapping_options.items():
            # Check if category appears as a standalone character or at word boundaries
            # if category in text_lower:
            if re.search(r'\b' + re.escape(category) + r'\b', text_lower):
                all_found_labels.append(category)
        # If only one category is found, use that
        if len(all_found_labels) == 1:
            label = all_found_labels[0]
        # Otherwise check if text starts with category name or letter
        else:
            text_stripped = text_lower.strip()
            # Check for category name at start
            for letter, category in mapping_options.items():
                if text_stripped.startswith(category):
                    label = category
                    break
            # Check for letter notation at start (A-K)
            if label == "na" and text_stripped and text_stripped[0] in mapping_options and not text_stripped[0].isdigit():
                label = mapping_options[text_stripped[0]]
        if len(all_found_labels) == 0 and label == "na":
            # check if the text contains any of the letters
            for letter, category in mapping_options.items():
                # Check if letter appears as a standalone character or at word boundaries
                # if letter in text_lower:
                if re.search(r'\b' + re.escape(letter) + r'\b', text_lower):
                    all_found_labels.append(category)
            if len(all_found_labels) == 1:
                label = all_found_labels[0]
        return label

    def split_data_by_keyword_based_grouping_function(self, df, keyword, case_sensitive=False):
        try:
            text_col = 'text'
            if 'random_seed_' in keyword:
                seed = int(keyword.split('_')[2])
                rng = np.random.default_rng(seed)
                if 'size' in keyword:
                    group_size = float(keyword.split('_')[4])
                    assert 0 < group_size < 1, f"Group size {group_size} must be between 0 and 1"
                else:
                    group_size = 0.5
                # now split the dataset into two randomly sampled groups
                group1 = df.sample(frac=group_size, random_state=rng)
                group2 = df.drop(group1.index)
                group1_name = f"{keyword}_group1"
                group2_name = f"{keyword}_group2"
            else:
                if case_sensitive:
                    group1 = df[df[text_col].str.contains(keyword)]
                    group2 = df[~df[text_col].str.contains(keyword)]
                else:
                    group1 = df[df[text_col].str.contains(keyword, case=False)]
                    group2 = df[~df[text_col].str.contains(
                        keyword, case=False)]

                group1_name = f"contains_{keyword}"
                group2_name = f"not_contains_{keyword}"
        except Exception as e:
            print(e)
            pdb.set_trace()

        return group1.copy(), group2.copy(), group1_name, group2_name

    def split_data_by_text_length(self, df):
        """Split by short vs long texts (based on character count)"""
        df_temp = df.copy()
        df_temp['text_length'] = df_temp['text'].str.len()
        median_length = df_temp['text_length'].median()

        group1 = df_temp[df_temp['text_length'] <= median_length]
        group2 = df_temp[df_temp['text_length'] > median_length]
        group1_name = 'short_texts'
        group2_name = 'long_texts'
        return group1.copy(), group2.copy(), group1_name, group2_name

    def get_most_frequent_words(self, df, n=100):
        """
        Get the x most frequent words from the DataFrame.

        Args:
            df: DataFrame containing text data
            x: Number of most frequent words to return

        Returns:
            List of x most frequent words
        """

        # Combine all text from the 'text' column
        all_text = df['text'].fillna("").astype(str).str.cat(sep=" ")

        # Tokenize text
        tokens = word_tokenize(all_text.lower())

        # Get NLTK stopwords
        stop_words = set(stopwords.words('english'))

        # Remove punctuation and stopwords, keep only alphabetic words
        filtered_words = [token for token in tokens
                          if token.isalpha() and token not in stop_words]

        # Count word frequencies and return top x words
        word_counts = Counter(filtered_words)
        most_frequent = word_counts.most_common(n)

        return [word for word, count in most_frequent]

    def get_default_groupings(self, df, nr_of_most_frequent_words_to_use_as_grouping_keywords=3):
        print('Get default groupings...')

        default_groupings = [
            ('random_seed_2', self.split_data_by_keyword_based_grouping_function, {
             'keyword': 'random_seed_2'}),
            ('random_seed_1_size_0.2', self.split_data_by_keyword_based_grouping_function, {
             'keyword': 'random_seed_1_size_0.2'}),
            ('random_seed_1_size_0.4', self.split_data_by_keyword_based_grouping_function, {
             'keyword': 'random_seed_1_size_0.4'}),
            ('request_length', self.split_data_by_text_length, {}),
        ]

        used_kw = []

        unique_classes = df['ground_truth'].unique()

        # max required value for n is freq. words plus twice the freq. words by class
        n = nr_of_most_frequent_words_to_use_as_grouping_keywords + 2 * \
            (nr_of_most_frequent_words_to_use_as_grouping_keywords*len(unique_classes))
        most_frequent_words = self.get_most_frequent_words(df, n=n)

        print(
            f'Retrieved {len(most_frequent_words)} unique words. Start keyword-based grouping.')

        kw_counter = 0
        print(f'Processing {len(most_frequent_words)} most frequent words...')
        for i, kw in enumerate(most_frequent_words):
            if i % 10 == 0:  # Progress every 10 words
                print(
                    f'  Progress: {i}/{len(most_frequent_words)} words processed')
            if kw_counter >= nr_of_most_frequent_words_to_use_as_grouping_keywords:
                break
            if kw not in used_kw:
                print(f'most frequent - {kw}')
                used_kw.append(kw)
                kw_counter += 1
                default_groupings.append(
                    (kw, self.split_data_by_keyword_based_grouping_function, {'keyword': kw}))

        print(
            f'Processing {len(unique_classes)} unique ground truth classes...')

        for class_idx, c in enumerate(unique_classes[:15]):
            # consider max 15 classes
            print(
                f'Processing class {class_idx + 1}/{len(unique_classes)}: "{c}"')
            proportion_difference = {}
            df_subset = df[df['ground_truth'] == c]
            if class_idx > 10:
                # only consider n=10 for first 10 classes
                nr_of_freq_words = 1
            else:
                nr_of_freq_words = n

            most_frequent_words_by_class = self.get_most_frequent_words(
                df_subset, n=nr_of_freq_words)

            print(
                f'  Found {len(most_frequent_words_by_class)} frequent words for class "{c}"')

            kw_counter = 0

            for word_idx, kw in enumerate(most_frequent_words_by_class):
                if word_idx % 10 == 0:  # Progress every 10 words for class-specific processing
                    print(
                        f'    Processing word {word_idx}/{len(most_frequent_words_by_class)} for class "{c}"')
                if kw in used_kw:
                    continue
                # try:
                group1, group2, _, _ = self.split_data_by_keyword_based_grouping_function(
                    df, keyword=kw)
                group1_c = len(group1[group1['ground_truth'] == c])
                group2_c = len(group2[group2['ground_truth'] == c])
                if group1_c > 0 and group2_c > 0:
                    group1_prop = group1_c/len(group1)
                    group2_prop = group2_c/len(group2)
                    proportion_difference[kw] = abs(group1_prop-group2_prop)
                if kw_counter < nr_of_most_frequent_words_to_use_as_grouping_keywords:
                    print(f'      - most in {c} - {kw}')
                    used_kw.append(kw)
                    kw_counter += 1
                    default_groupings.append(
                        (kw, self.split_data_by_keyword_based_grouping_function, {'keyword': kw}))
                # except Exception as e:
                #     print(f"Failed to generate grouping for class {c}")

            print(
                f'  Sorting {len(proportion_difference)} proportion differences for class "{c}"')
            sorted_prop_diff = dict(
                sorted(proportion_difference.items(), key=lambda item: item[1], reverse=True))

            kw_counter = 0
            for prop_idx, (kw, prop) in enumerate(sorted_prop_diff.items()):
                if prop_idx % 10 == 0 and prop_idx > 0:  # Progress every 10 items in proportion difference processing
                    print(
                        f'    Processing proportion difference {prop_idx}/{len(sorted_prop_diff)} for class "{c}"')
                if kw_counter >= nr_of_most_frequent_words_to_use_as_grouping_keywords:
                    break
                if kw not in used_kw:
                    print(f'      - most in {c} (abs prop diff={prop}) - {kw}')
                    used_kw.append(kw)
                    kw_counter += 1
                    default_groupings.append(
                        (kw, self.split_data_by_keyword_based_grouping_function, {'keyword': kw}))

        print(
            f'Done with default groupings... Generated {len(default_groupings)} total groupings.')
        return default_groupings

    def get_dataset_specific_groups(self):
        raise NotImplementedError(
            "Subclasses must implement get_dataset_specific_groups method")

    def get_groups(self, df):
        default_groupings = self.get_default_groupings(df)
        nr_of_default_groupings = len(default_groupings)
        dataset_specific_groups = self.get_dataset_specific_groups(df)
        nr_of_dataset_specific_groups = len(dataset_specific_groups)
        return default_groupings + dataset_specific_groups, nr_of_default_groupings, nr_of_dataset_specific_groups

    def split_by_temporal_quartiles(self, df, timestamp_col, split_type='q1_vs_rest', date_format=None, date_unit=None):
        """Split by temporal quartiles using specified split type"""
        df_temp = df.copy()
        datetime_args = {}
        if date_format is not None:
            datetime_args.update({'format': date_format})
        if date_unit is not None:
            datetime_args.update({'unit': date_unit})
        df_temp[timestamp_col] = pd.to_datetime(
            df_temp[timestamp_col], **datetime_args)

        # Remove rows with missing timestamps
        df_valid = df_temp.dropna(subset=[timestamp_col])

        if len(df_valid) == 0:
            raise ValueError(
                f"No valid timestamps found in column '{timestamp_col}'")

        # Calculate quartile cutoff points
        q25 = df_valid[timestamp_col].quantile(0.25)
        q50 = df_valid[timestamp_col].quantile(0.50)
        q75 = df_valid[timestamp_col].quantile(0.75)

        if split_type == 'q1_vs_rest':
            # Q1 vs Q2+Q3+Q4
            group1 = df_valid[df_valid[timestamp_col] <= q25]  # Q1
            group2 = df_valid[df_valid[timestamp_col] > q25]   # Q2+Q3+Q4
            group1_name = 'q1'
            group2_name = 'q2_q3_q4'
        elif split_type == 'q1q2_vs_rest':
            # Q1+Q2 vs Q3+Q4
            group1 = df_valid[df_valid[timestamp_col] <= q50]  # Q1+Q2
            group2 = df_valid[df_valid[timestamp_col] > q50]   # Q3+Q4
            group1_name = 'q1_q2'
            group2_name = 'q3_q4'
        elif split_type == 'q1q2q3_vs_q4':
            # Q1+Q2+Q3 vs Q4
            group1 = df_valid[df_valid[timestamp_col] <= q75]  # Q1+Q2+Q3
            group2 = df_valid[df_valid[timestamp_col] > q75]   # Q4
            group1_name = 'q1_q2_q3'
            group2_name = 'q4'
        else:
            raise ValueError(f"Unknown split_type: {split_type}")

        return group1.copy(), group2.copy(), group1_name, group2_name

    def add_temporal_split_groupings(self, timestamp_col='Timestamp', date_format=None, date_unit=None):
        return [
            ('temporal_q1_vs_rest', self.split_by_temporal_quartiles, {
             'split_type': 'q1_vs_rest', 'timestamp_col': timestamp_col, 'date_format': date_format, 'date_unit': date_unit}),
            ('temporal_q1q2_vs_rest', self.split_by_temporal_quartiles, {
             'split_type': 'q1q2_vs_rest', 'timestamp_col': timestamp_col, 'date_format': date_format, 'date_unit': date_unit}),
            ('temporal_q1q2q3_vs_q4', self.split_by_temporal_quartiles, {
             'split_type': 'q1q2q3_vs_q4', 'timestamp_col': timestamp_col, 'date_format': date_format, 'date_unit': date_unit}),
        ]

    def split_by_column_and_value(self, df, col_name, value, value_name):
        """Split by column and value"""
        group1 = df[df[col_name] == value]
        group2 = df[
            (df[col_name] != value) &
            (df[col_name].notna())
        ]
        if len(value_name) > 20:
            # shorten value_name for group description
            value_name = value_name[:20] + '...'
        group1_name = f'{col_name}_is_{value_name}'
        group2_name = f'{col_name}_isnot_{value_name}'
        return group1.copy(), group2.copy(), group1_name, group2_name

    def split_by_column_with_n_most_frequent_values(self, df, col_name, n=10):

        ten_most_frequent_values = df[col_name].value_counts().head(n)
        groupings = []
        for value, count in ten_most_frequent_values.items():
            value_name = str(value)
            if len(value_name) > 20:
                # shorten value_name for group description
                value_name = value_name[:20] + '...'
                # check if value_name already exists
                all_value_name = [v[2]['value_name'] for v in groupings if value_name in v[2]['value_name']]
                value_name = f'{value_name}{str(len(all_value_name))}'

            groupings.append((f'col_{col_name}_val_{value_name}', self.split_by_column_and_value, {
                             'col_name': col_name, 'value': value, 'value_name': value_name}))

        return groupings


    def clean_text(self, text):
        """Remove/normalize special characters for matching purposes"""
        import unicodedata

        if pd.isna(text):
            return text
        
        # Remove _x000D_ and similar XML/HTML encoded sequences
        text = re.sub(r'_x[0-9A-Fa-f]{4}_', '', text)
        
        # Normalize Unicode characters to closest ASCII equivalents FIRST
        # This converts 𝐋 to L, é to e, etc.
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove ALL punctuation and special characters, keep only alphanumeric
        text = re.sub(r'[^\w]', '', text)
        
        return text



def deduplicate_df(data_loader, dataset_name, dataset):
    print(f"Full dataset size:", len(dataset))
    prompts = data_loader.get_prompts()
    # groupings, nr_of_default_groupings, nr_of_dataset_specific_groups = data_loader.get_groups(dataset)

    if dataset_name not in ['issue_survey', 'manifestos_econ_ideology', 'manifestos_social_ideology', 'manifestos_issues_detailed']:
        # Calculate unique datapoints using first prompt
        # not for large datasets in list as these do not use any metadata for prompt formatting anyway
        first_prompt = prompts[0]['prompt_text']
        dataset['TEXT_FORMATTED_TO_CHECK_FOR_DUPLICATES'] = dataset.apply(
            lambda row: data_loader.format_prompt(first_prompt, row), axis=1)
    else:
        dataset['TEXT_FORMATTED_TO_CHECK_FOR_DUPLICATES'] = dataset['text']
    # shuffle data
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    # keep only first row for each unique value of TEXT_FORMATTED_TO_CHECK_FOR_DUPLICATES, dropping all other duplicates
    dataset = dataset.drop_duplicates(
        subset=['TEXT_FORMATTED_TO_CHECK_FOR_DUPLICATES'], keep='first')
    # drop column TEXT_FORMATTED_TO_CHECK_FOR_DUPLICATES again
    dataset = dataset.drop(columns=['TEXT_FORMATTED_TO_CHECK_FOR_DUPLICATES'])
    print(f"Deduplicated dataset size:", len(dataset))
    return dataset


def compare_stratified_distribution_with_original(df, df_final):
    # Compare distributions
    original_dist = df['ground_truth'].value_counts(
        normalize=True, dropna=False)
    sample_dist = df_final['ground_truth'].value_counts(
        normalize=True, dropna=False)

    print("Original size:", len(df))
    print("Sample size:", len(df_final))
    print("Sample percentage:", len(df_final) / len(df) * 100)

    # Check if distributions are similar
    print("\nDistribution comparison:")
    comparison = pd.DataFrame({
        'original': original_dist,  # .head(10),
        'sample': sample_dist,  # .head(10)
    })
    print(comparison)


def stratified_sample_for_large_datasets(df, cols_to_stratify):
    if len(df) > 10000:
        sample_fraction = 10000 / len(df)
        # prepare cols to stratify
        cols_strat = []
        cols_generated_for_strat = []
        for col_name in cols_to_stratify:
            if isinstance(col_name, tuple):
                name = col_name[0]
                nr_of_values_used_for_grouping = col_name[1]
                col_name_new = f'{name}_COLFORSTRAT'
                cols_strat.append(col_name_new)
                cols_generated_for_strat.append(col_name_new)
                most_frequent_values = df[name].value_counts().head(
                    nr_of_values_used_for_grouping)
                df[col_name_new] = df[name].apply(
                    lambda x: x if x in most_frequent_values else 'OTHERFORSTRAT')
            else:
                cols_strat.append(col_name)
        # Get sample with minimum 2 samples per subgroup
        df_final = df.groupby(cols_strat, group_keys=False).apply(
            lambda x: x.sample(
                n=max(2, int(len(x) * sample_fraction)), random_state=42) if len(x) >= 2 else x
        ).reset_index(drop=True)
        compare_stratified_distribution_with_original(df, df_final)
        # drop the column generated only for the stratification
        df_final = df_final.drop(columns=cols_generated_for_strat)
    else:
        df_final = df
    return df_final


def process_full_dataset(dataset_name, dataset=None, save_to_disk=True):
    data_loader = map_dataset_name_to_class(dataset_name)
    if dataset is None:
        dataset = data_loader.load_dataset(processed_data_dir='all_data_processed_full')
    df_deduplicated = deduplicate_df(data_loader, dataset_name, dataset)
    df_final = stratified_sample_for_large_datasets(
        df_deduplicated, data_loader.get_cols_to_stratify())

    # add index to final df
    df_final = df_final.reset_index(drop=True)
    df_final['id'] = df_final.index.map(lambda x: f"{dataset_name}_{x}")
    # Move id column to first position
    cols = ['id'] + [col for col in df_final.columns if col != 'id']
    df_final = df_final[cols]
    df_final = df_final[df_final['ground_truth'].notna()]
    print(f"Deduplicated dataset size:", len(df_final))

    if dataset_name in [
        'relevance_tweets',
        'framesI_tweets',
        'stance_tweets',
        'topic_tweets',
        'relevance_tweets23',
        'framesI_tweets23',
        'relevance_tweets17',
        'framesII_tweets17',
        'relevance_news',
        'framesI_news',
    ]:
        # load mturk annotations
        df_with_mturk_annotations = data_loader.load_mturk_annotations()
        df_with_mturk_annotations = df_with_mturk_annotations[['text_clean', 'mturk_annotation', 'WorkerId']]

        # Create cleaned text columns for matching
        df_final['text_clean'] = df_final['text'].apply(data_loader.clean_text)

        # Get df_final subset that was not found in df_with_mturk_annotations
        df_final_not_in_mturk = df_final[~df_final['text_clean'].isin(df_with_mturk_annotations['text_clean'])]
        # print(f'\n\nRows in df_final NOT in mturk: {len(df_final_not_in_mturk)}')
        
        # Get df_with_mturk_annotations subset that was not found in df_final
        df_mturk_not_in_old = df_with_mturk_annotations[~df_with_mturk_annotations['text_clean'].isin(df_final['text_clean'])]
        # print(f'\n\nRows in mturk NOT in df_final: {len(df_mturk_not_in_old)}')

        # Fix unmatched rows by finding best matches
        if len(df_final_not_in_mturk) > 0:
            print(f'\n{dataset_name}: Fixing {len(df_final_not_in_mturk)} unmatched rows...')
            # Track used MTurk rows
            used_mturk_indices = set()
            
            for idx, row in df_final_not_in_mturk.iterrows():
                # Find best match excluding already used rows
                similarities = df_with_mturk_annotations.apply(
                    lambda x: SequenceMatcher(None, row['text_clean'], x['text_clean']).ratio() 
                            if x.name not in used_mturk_indices else 0,
                    axis=1
                )
                best_idx = similarities.idxmax()
                
                # Calculate char diff BEFORE overwriting
                best_match_original = df_with_mturk_annotations.loc[best_idx, 'text_clean']
                char_diff = abs(len(row['text_clean']) - len(best_match_original))
                print(f"Similarity: {similarities.max():.3f} | Char diff: {char_diff}")
                
                # Overwrite text_clean in mturk df to match
                df_with_mturk_annotations.loc[best_idx, 'text_clean'] = row['text_clean']
                used_mturk_indices.add(best_idx) 

        # Merge on cleaned text, keeping the original text from df_final
        df_final = df_final.merge(df_with_mturk_annotations, on='text_clean', how='left')
        
        # Drop the text_clean column from final df (optional)
        df_final = df_final.drop(columns=['text_clean'])

        # Assert all rows were matched
        unmatched_count = df_final['mturk_annotation'].isna().sum()
        assert unmatched_count == 0, f"{dataset_name}: {unmatched_count} rows still unmatched after fuzzy matching!"
        # print(f"{dataset_name}: Successfully matched all {len(df_final)} rows")

    if save_to_disk:
        # save final df to disk
        data_loader.save_final_dataset_to_disk(df_final)

    return df_final
