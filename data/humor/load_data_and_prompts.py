from data.data_utils import MyDataLoader
import copy


class HumorDataset(MyDataLoader):
    def __init__(self, data_directory='humor', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'True': 'True',
            'False': 'False',
        }
        dict_with_mapping_options_letters = {
            'A': 'True',
            'B': 'False',
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
            'default output mapping letters': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options_letters}
            },
        }

    def get_reported_results(self):
        return []


    def get_dataset_specific_groups(self, df):
        return self.add_temporal_split_groupings(timestamp_col='date')

    def get_cols_to_stratify(self):
        return [
            'ground_truth',
        ]

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the statement "the joke is humorous to most people" is {previous_answer_placeholder} for the following text?

Output only a single number between 0 and 1, without any context or explanation.

Text: {text}

Probability:'''

    def get_prompts(self):
        return [
            {
                'description': '[original] [zero-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # from ziems et al.
                        "content": """Joke: {text}

Constraint: Is the above joke humorous to most of the people? You must pick between "True" or "False" You cannot use any other words except for "True" or "False"."""
                    }
                ],
            },
            {
                'description': '[zero-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping letters'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Joke: {text}

Is the above joke humorous to most of the people? You must pick between one of these two options:
A: True
B: False

Constraint: Answer only 'A' or 'B'."""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[orig*] [zero-shot-short]',
                'original_description': '[original] [zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Joke: {text}\n\nTask: Decide if most people would find this joke funny.\nRespond exclusively with \"True\" for yes or \"False\" for no—no additional words.'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping letters'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": "Joke: {text}\n\nWould most people find this joke funny? Choose exactly one of the following:\nA: True\nB: False\n\nImportant: reply with only 'A' or 'B'."
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-2] orig:[orig*] [zero-shot-short]',
                'original_description': '[original] [zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Joke: {text}\n\nTask: Determine whether the majority of people would find the joke above humorous.\nRespond with exactly one word—either "True" or "False"—and provide no additional text.'
                    }
                ],
            },
        ]
