from data.data_utils import MyDataLoader
import copy


class MisinfoDataset(MyDataLoader):
    def __init__(self, data_directory='misinfo', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'A': 'Misinformation',
            'B': 'Trustworthy',
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
        }

    def get_reported_results(self):
        return []


    def get_dataset_specific_groups(self, df):
        temporal_split_groups = self.add_temporal_split_groupings(timestamp_col='date', date_format='mixed')
        source_groupings = self.split_by_column_with_n_most_frequent_values(df, 'source')
        type_groupings = self.split_by_column_with_n_most_frequent_values(df, 'type')
        return temporal_split_groups + source_groupings + type_groupings

    def get_cols_to_stratify(self):
        return [
            'ground_truth',
            ('source', 10),
            ('type', 10),
        ]


    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following news headline is {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

News headline: {text}

Probability:'''

    def get_prompts(self):
        return [
            {
                'description': '[zero-shot-short] [ziems]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # from ziems et al.
                        "content": """News headline: {text}
                        Which of the following describes the above news headline?
A: Misinformation
B: Trustworthy

Constraint: Answer with only the option above that is most accurate and nothing else."""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[zero-shot-short] [ziems]',
                'original_description': '[zero-shot-short] [ziems]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'News headline: {text}\n\nSelect the label that best fits the headline:\nA: Misinformation\nB: Trustworthy\n\nReply with only the chosen letter (A or B) and nothing else.'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[zero-shot-short] [ziems]',
                'original_description': '[zero-shot-short] [ziems]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Headline to assess: {text}\nPlease select the label that best fits this headline:\nA: Misinformation\nB: Trustworthy\n\nInstruction: Respond with only the single letter (A or B) that corresponds to your selection.'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-2] orig:[zero-shot-short] [ziems]',
                'original_description': '[zero-shot-short] [ziems]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Headline: {text}\nSelect the label that best categorizes this headline:\nA: Misinformation\nB: Trustworthy\n\nRespond with only the single letter (A or B) that applies.'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-3] orig:[zero-shot-short] [ziems]',
                'original_description': '[zero-shot-short] [ziems]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'News headline: {text}\nPlease select the label that best fits the headline:\nA: Misinformation\nB: Trustworthy\n\nReply with ONLY the single letter (A or B) that corresponds to your choice.'
                    }
                ],
            },
        ]
