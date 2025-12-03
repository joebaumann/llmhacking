from data.data_utils import MyDataLoader
import copy


class TopicDataset(MyDataLoader):
    def __init__(self, data_directory='topic', config_fn='config.yaml'):
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
        senate_groupings = self.split_by_column_with_n_most_frequent_values(df, 'senate')
        democrat_groupings = self.split_by_column_with_n_most_frequent_values(df, 'democrat')
        ideology_groupings = self.split_by_column_with_n_most_frequent_values(df, 'ideology')
        return senate_groupings + democrat_groupings + ideology_groupings

    def get_cols_to_stratify(self):
        return [
            'ground_truth',
            ('senate', 10),
            ('democrat', 10),
            ('ideology', 10),
        ]

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the statement "the text relates to the economy" is {previous_answer_placeholder} for the following text?

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
                        "content": """Does the following text relate to the economy? (True/False)

text: {text}

label:"""
                    }
                ],
            },
            {
                'description': '[original] [few-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Does the following text relate to the economy? (True/False)

text: To provide that Federal expenditures shall not exceed Federal revenues, except in time of war or grave emergency declared by the Congress, and to provide for systematic reduction of the public debt

label:"""
                    },
                    {
                        "role": "assistant",
                        "content": """True"""
                    },
                    {
                        "role": "user",
                        "content": """text: To amend the Internal Revenue Act of 1954 to increase from $600 to $1,200 the personal income tax exemptions of a taxpayer (including the exemtion for a spouse, the exemption for a dependent, and the additional exemptions for old age and blindness)

label:"""
                    },
                    {
                        "role": "assistant",
                        "content": """True"""
                    },
                    {
                        "role": "user",
                        "content": """text: A bill to amend the Internal Revenue Code of 1954 to provide special loss carryover rules for insurance companies.

label:"""
                    },
                    {
                        "role": "assistant",
                        "content": """False"""
                    },
                    {
                        "role": "user",
                        "content": """text: To provide individuals with access to health information of which they are a subject, to ensure personal privacy, security, and confidentiality with respect to health related information in promoting the development of a nationwide interoperable health information infrastructure, to impose criminal and civil penalties for unauthorized use of personal health information, to provide for the strong enforcement of these rights, to protect States' rights, and for other purposes.

label:"""
                    },
                    {
                        "role": "assistant",
                        "content": """True"""
                    },
                    {
                        "role": "user",
                        "content": """text: A bill to amend title XVI of the Social Security Act to reduce from 21 to 18 the age at which a disabled child need no longer include his parents' income in determining his eligibility for supplemental security income benefits or the amount of such benefits.

label:"""
                    },
                    {
                        "role": "assistant",
                        "content": """False"""
                    },
                    {
                        "role": "user",
                        "content": """text: {text}

label:"""
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
                        "content": """Does the following text relate to the economy?
A: True
B: False

Answer only 'A' or 'B'.

text: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[few-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping letters'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Does the following text relate to the economy?
A: True
B: False

Answer only 'A' or 'B'.

text: To provide that Federal expenditures shall not exceed Federal revenues, except in time of war or grave emergency declared by the Congress, and to provide for systematic reduction of the public debt

Answer:"""
                    },
                    {
                        "role": "assistant",
                        "content": """A"""
                    },
                    {
                        "role": "user",
                        "content": """text: To amend the Internal Revenue Act of 1954 to increase from $600 to $1,200 the personal income tax exemptions of a taxpayer (including the exemtion for a spouse, the exemption for a dependent, and the additional exemptions for old age and blindness)

Answer:"""
                    },
                    {
                        "role": "assistant",
                        "content": """A"""
                    },
                    {
                        "role": "user",
                        "content": """text: A bill to amend the Internal Revenue Code of 1954 to provide special loss carryover rules for insurance companies.

Answer:"""
                    },
                    {
                        "role": "assistant",
                        "content": """B"""
                    },
                    {
                        "role": "user",
                        "content": """text: To provide individuals with access to health information of which they are a subject, to ensure personal privacy, security, and confidentiality with respect to health related information in promoting the development of a nationwide interoperable health information infrastructure, to impose criminal and civil penalties for unauthorized use of personal health information, to provide for the strong enforcement of these rights, to protect States' rights, and for other purposes.

Answer:"""
                    },
                    {
                        "role": "assistant",
                        "content": """A"""
                    },
                    {
                        "role": "user",
                        "content": """text: A bill to amend title XVI of the Social Security Act to reduce from 21 to 18 the age at which a disabled child need no longer include his parents' income in determining his eligibility for supplemental security income benefits or the amount of such benefits.

Answer:"""
                    },
                    {
                        "role": "assistant",
                        "content": """B"""
                    },
                    {
                        "role": "user",
                        "content": """text: {text}

Answer:"""
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
                        "content": 'Determine whether the following text concerns the economy. Reply only with "True" or "False".\n\ntext: {text}\n\nlabel:'
                    }
                ],
            },
        ]
