from data.data_utils import MyDataLoader
import pdb
import numpy as np
import pandas as pd


class StanceClimateDataset(MyDataLoader):
    def __init__(self, data_directory='stance_climate', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'A': 'Agree',
            'B': 'Neutral',
            'C': 'Disagree',
        }
        dict_with_mapping_options_reversed = {
            'A': 'Agree',
            'B': 'Disagree',
            'C': 'Neutral',
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
        return []

    def get_cols_to_stratify(self):
        return [
            'ground_truth',
        ]


    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following statement {previous_answer_placeholder} with the assertion "Climate change/global warming is a serious concern"?

Output only a single number between 0 and 1, without any context or explanation.

Statement: {text}

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
                        "content": """A statement can agree, be neutral, or disagree with the statement: "Climate change/global warming is a serious concern". Classify the following statement into one of the three categories. Output either A, B, or C. Output a letter only.
A) Agree
B) Neutral
C) Disagree
Statement: {text}
Answer:"""
                    }
                ],
            },
            {
                'description': '[original] [zero-shot-detailed] [without system prompt]',
                'compatible_output_mapping': [
                    'default output mapping reversed'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # adapted from annotation guideline in Appendix D of https://aclanthology.org/2020.findings-emnlp.296v3.pdf
                        "content": """Target opinion: Climate change/global warming is a serious concern.

SENTENCE: {text}

A. AGREES (the writer probably agrees with the target opinion)
This could be for any of reasons shown below:
- the writer mentions explicit support for the target
- the writer mentions support for something/someone aligned with the target
- the writer mentions opposition to something/someone opposing the target

B. DISAGREES (the writer probably disagrees with the target opinion)
This could be for any of the following:
- the writer mentions explicit opposition to the target
- the writer mentions opposition to someone/something aligned with the target
- the writer mentions support for someone/something opposing the target

C. NEUTRAL (it is unclear whether or not the writer supports the target opinion)
The writer could be either in support of or against the target; the sentence does not reveal the writer's stance.

Your task is to annotate the SENTENCE for agreeing (A), disagreeing (B), or being neutral (C) with respect to the target opinion. Output either A, B, or C. Output a letter only."""
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
                        "content": 'Decide whether the following statement agrees with, is neutral toward, or disagrees with the claim: "Climate change/global warming is a serious concern." Reply using only the corresponding letter code.\nA) Agree\nB) Neutral\nC) Disagree\nStatement: {text}\nAnswer:'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[orig*] [zero-shot-short] [gligoric]',
                'original_description': '[original] [zero-shot-short] [gligoric]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Assess the stance of the sentence below toward the claim: "Climate change/global warming is a serious concern." Reply with only one letter from the options provided.\nA) Agree\nB) Neutral\nC) Disagree\n\nSentence: {text}\nAnswer:'
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
                        "content": 'Consider the claim: "Climate change/global warming is a serious concern." Decide whether the statement below (A) agrees with, (B) is neutral toward, or (C) disagrees with this claim, and respond using only the corresponding letter.\nA) Agree\nB) Neutral\nC) Disagree\nStatement: {text}\nAnswer:'
                    }
                ],
            },
        ]
