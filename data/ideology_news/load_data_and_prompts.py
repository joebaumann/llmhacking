from data.data_utils import MyDataLoader
import pdb
import numpy as np
import pandas as pd


class IdeologyNewsDataset(MyDataLoader):
    def __init__(self, data_directory='ideology_news', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'A': 'Left',
            'B': 'Center',
            'C': 'Right',
        }
        dict_with_mapping_options_reversed = {
            'A': 'Left',
            'B': 'Right',
            'C': 'Center',
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
            'default output mapping reversed': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options_reversed}
            },
        }

    def get_reported_results(self):
        return []


    def get_dataset_specific_groups(self, df):

        topic_groupings = self.split_by_column_with_n_most_frequent_values(df, 'topic')
        source_groupings = self.split_by_column_with_n_most_frequent_values(df, 'source')
        authors_groupings = self.split_by_column_with_n_most_frequent_values(df, 'authors')

        temporal_split_groups = self.add_temporal_split_groupings(timestamp_col='date', date_format='%Y-%m-%d')

        # Combine parent and specific groupings
        return topic_groupings + source_groupings + authors_groupings + temporal_split_groups

    def get_cols_to_stratify(self):
        return [
            'ground_truth',
            ('topic', 10),
            ('source', 10),
            ('authors', 10),
        ]


    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following article has a {previous_answer_placeholder} political bias?

Output only a single number between 0 and 1, without any context or explanation.

Article: {text}

Probability:'''


    def get_prompts(self):
        return [
            {
                'description': '[original] [zero-shot-short] [gligoric]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # adapted from Table 2 in https://aclanthology.org/2025.naacl-long.179.pdf
                        "content": """What is the political bias of the following article? Output either A,B, or C. Output a letter only.
A) Left
B) Center
C) Right
Article: {text}
Answer:"""
                    }
                ],
            },
            {
                'description': '[original] [zero-shot-short] [ziems]',
                'compatible_output_mapping': [
                    'default output mapping reversed'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # taken from Ziems et al.
                        "content": """{text}

Which of the following leanings would a political scientist say that the above article has?
A: Left
B: Right
C: Center

Constraint: Answer with only the option above that is most accurate and nothing else."""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[orig*] [zero-shot-short] [gligoric]',
                'original_description': '[original] [zero-shot-short] [gligoric]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Determine the article’s political bias and respond with a single letter—A, B, or C—only.\nA: Left\nB: Center\nC: Right\n\nArticle:\n{text}\n\nResponse:'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[orig*] [zero-shot-short] [ziems]',
                'original_description': '[original] [zero-shot-short] [ziems]',
                'compatible_output_mapping': ['default output mapping reversed'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": '{text}\n\nFrom a political-science perspective, which ideological stance best characterizes the article above?\nA: Left\nB: Right\nC: Center\n\nReply constraint: Output only the single letter (A, B, or C) that corresponds to your selection, with nothing else.'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-2] orig:[orig*] [zero-shot-short] [gligoric]',
                'original_description': '[original] [zero-shot-short] [gligoric]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Identify the political bias of the article below and reply with ONLY one letter: A, B, or C.\nA: Left\nB: Center\nC: Right\n\nArticle:\n{text}\n\nResponse:'
                    }
                ],
            },
        ]
