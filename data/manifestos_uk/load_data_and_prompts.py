from data.data_utils import MyDataLoader
import copy
import json
import pdb


class ManifestosUK(MyDataLoader):
    
    def get_dataset_specific_groups(self, df):
        party_groupings = self.split_by_column_with_n_most_frequent_values(df, 'party')
        year_groupings = self.split_by_column_with_n_most_frequent_values(df, 'year')
        return party_groupings + year_groupings

    def get_cols_to_stratify(self):
        return [
            'ground_truth',
            ('party', 10),
            ('year', 10),
        ]


    def load_human_annotations_for_alt_test(self, dataset):
        """
        Load human annotations from the 'original_annotations' column.
        
        Parameters:
        -----------
        dataset : pd.DataFrame
            The dataset containing the 'original_annotations' column with JSON strings
        annotation_mapping : dict
            Mapping from numeric codes to textual labels
            
        Returns:
        --------
        dict : A dictionary of dictionaries where:
            - Outer keys represent annotators (annotator ids).
            - Inner dictionaries with keys representing instances (instance ids) 
              and values representing annotations.
        
        Example:
        {
            'annotator1': {'instance1': 'economy', 'instance2': 'social'},
            'annotator2': {'instance1': 'economy', 'instance2': 'economy', 'instance3': 'social'}
        }
        """

        annotation_mapping = self.get_annotation_mapping()
        
        humans_annotations = {}

        # Iterate through each row in the dataset
        for _, row in dataset.iterrows():
            instance_id = row['id']
            
            # Parse the JSON string from original_annotations
            annotations_json = json.loads(row['original_annotations'])
            
            # Process each annotation in the JSON
            for annotator_id, code in annotations_json.items():
                # Convert code to integer if it's a string
                code_int = int(float(code)) if isinstance(code, str) else int(code)
                
                # Skip if the code is not in the mapping (shouldn't happen with clean data)
                if code_int not in annotation_mapping:
                    print(f"Warning: Code {code_int} not in annotation_mapping for instance {instance_id}")
                    continue
                
                # Initialize annotator dictionary if not exists
                if annotator_id not in humans_annotations:
                    humans_annotations[annotator_id] = {}
                
                # Add the annotation with the mapped label
                humans_annotations[annotator_id][instance_id] = annotation_mapping[code_int]
        
        return humans_annotations



class ManifestosUKIssueDataset(ManifestosUK):
    def __init__(self, data_directory='manifestos_uk', config_fn='config_manifestos_issue.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'social': 'social',
            'economic': 'economy',
        }
        dict_with_mapping_options_with_letters = {
            'A': 'social',
            'B': 'economy',
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
            'default output mapping with letters': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options_with_letters}
            },
        }

    def get_annotation_mapping(self):
        """Return the annotation mapping for issue classification."""
        return {
            0: 'economy',
            1: 'social'
        }

    def get_reported_results(self):
        return []


    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following sentence mainly concerns {previous_answer_placeholder} policy?

Output only a single number between 0 and 1, without any context or explanation.

Sentence: {text}

Probability:'''

    def get_prompts(self):
        return [
            {
                'description': '[zero-shot-short] [original] barrie',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # from Barrie et al. (2025)
                        "content": """This task involves reading some sentences from political manifestos. You are asked whether the sentence refers to Social or Economic policy. For this sentence, enter your best judgment about whether it mainly refers to economic policy, to social policy. Answer only with one word 'Social' or 'Economic'.

Sentence: {text}

Answer: """
                    }
                ],
            },
            {
                'description': '[zero-shot-short] [original] ornstein',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # from Ornstein et al. (2025)
                        "content": """Decide whether this sentence from a British political manifesto is about Social Policy or Economic Policy.

Sentence: We will implement a comprehensive strategiy for ending low pay, notably by the introduction of a statutory national minimum wage.
Classification Economic Policy
--- 

Sentence: {text}
Classification: 
"""
                    }
                ],
            },
            {
                'description': '[few-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                # from Ornstein et al. (2025)
                'prompt_text': [
                    {
                        "role": "user",
                        "content": "Decide whether this sentence from a British political manifesto is about Social Policy or Economic Policy."
                    },
                    {
                        "role": "user",
                        "content": """Sentence: We will implement a comprehensive strategiy for ending low pay, notably by the introduction of a statutory national minimum wage.
Classification: """
                    },
                    {
                        "role": "assistant",
                        "content": "Economic"
                    },
                    {
                        "role": "user",
                        "content": """Sentence: {text}
Classification: """
                    },
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[zero-shot-short] [orig*] barrie',
                'original_description': '[zero-shot-short] [original] barrie',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Read the following sentence from a political manifesto and determine whether its main focus is social policy or economic policy. Respond with exactly one lowercase word—either 'social' or 'economic'—and provide no additional text.

Sentence: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[zero-shot-short] [orig*] ornstein',
                'original_description': '[zero-shot-short] [original] ornstein',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Determine whether the following sentence from a British political manifesto concerns Social Policy or Economic Policy.\n\nReply with exactly one of these two labels — social or economic — and nothing else.\n\nExample:\nSentence: We will implement a comprehensive strategy for ending low pay, notably by the introduction of a statutory national minimum wage.\nLabel: economic\n---\n\nSentence: {text}\nLabel: '
                    }
                ],
            },
        ]


class ManifestosUKSocialIdeologyDataset(ManifestosUK):
    def __init__(self, data_directory='manifestos_uk', config_fn='config_manifestos_social_ideology.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'liberal': 'liberal',
            'conservative': 'conservative',
        }
        dict_with_mapping_options_with_letters = {
            'A': 'liberal',
            'B': 'conservative',
        }
        dict_with_mapping_options_score = {
            '100': 'conservative',
            '99': 'conservative',
            '98': 'conservative',
            '97': 'conservative',
            '96': 'conservative',
            '95': 'conservative',
            '94': 'conservative',
            '93': 'conservative',
            '92': 'conservative',
            '91': 'conservative',
            '90': 'conservative',
            '89': 'conservative',
            '88': 'conservative',
            '87': 'conservative',
            '86': 'conservative',
            '85': 'conservative',
            '84': 'conservative',
            '83': 'conservative',
            '82': 'conservative',
            '81': 'conservative',
            '80': 'conservative',
            '79': 'conservative',
            '78': 'conservative',
            '77': 'conservative',
            '76': 'conservative',
            '75': 'conservative',
            '74': 'conservative',
            '73': 'conservative',
            '72': 'conservative',
            '71': 'conservative',
            '70': 'conservative',
            '69': 'conservative',
            '68': 'conservative',
            '67': 'conservative',
            '66': 'conservative',
            '65': 'conservative',
            '64': 'conservative',
            '63': 'conservative',
            '62': 'conservative',
            '61': 'conservative',
            '60': 'conservative',
            '59': 'conservative',
            '58': 'conservative',
            '57': 'conservative',
            '56': 'conservative',
            '55': 'conservative',
            '54': 'conservative',
            '53': 'conservative',
            '52': 'conservative',
            '51': 'conservative',
            '50': 'conservative',
            '49': 'liberal',
            '48': 'liberal',
            '47': 'liberal',
            '46': 'liberal',
            '45': 'liberal',
            '44': 'liberal',
            '43': 'liberal',
            '42': 'liberal',
            '41': 'liberal',
            '40': 'liberal',
            '39': 'liberal',
            '38': 'liberal',
            '37': 'liberal',
            '36': 'liberal',
            '35': 'liberal',
            '34': 'liberal',
            '33': 'liberal',
            '32': 'liberal',
            '31': 'liberal',
            '30': 'liberal',
            '29': 'liberal',
            '28': 'liberal',
            '27': 'liberal',
            '26': 'liberal',
            '25': 'liberal',
            '24': 'liberal',
            '23': 'liberal',
            '22': 'liberal',
            '21': 'liberal',
            '20': 'liberal',
            '19': 'liberal',
            '18': 'liberal',
            '17': 'liberal',
            '16': 'liberal',
            '15': 'liberal',
            '14': 'liberal',
            '13': 'liberal',
            '12': 'liberal',
            '11': 'liberal',
            '10': 'liberal',
            '9': 'liberal',
            '8': 'liberal',
            '7': 'liberal',
            '6': 'liberal',
            '5': 'liberal',
            '4': 'liberal',
            '3': 'liberal',
            '2': 'liberal',
            '1': 'liberal',
            '0': 'liberal',
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
            'default output mapping with letters': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options_with_letters}
            },
            'default output mapping score': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options_score}
            },
        }

    def get_reported_results(self):
        return []


    def get_annotation_mapping(self):
        """Return the annotation mapping for social ideology classification."""
        return {
            -1: 'liberal',
            1: 'conservative'
        }

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the social policy stance expressed in the following sentence is {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Sentence: {text}

Probability:'''

    def get_prompts(self):
        return [
            {
                'description': '[zero-shot-short] [original] barrie',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # from Barrie et al. (2025)
                        "content": """This task involves reading some sentences from political manifestos. You are given a sentence that refers to Social policy. For this sentence, enter your best judgment about whether the policy is liberal or conservative. Answer only with one word 'liberal' or 'conservative'.

Sentence: {text}

Answer: """
                    }
                ],
            },
            {
                'description': '[zero-shot-short] [original] ornstein',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # from Ornstein et al. (2025)
                        "content": """Decide whether this sentence from a British manifesto is Liberal or Conservative.

Sentence: We will implement a comprehensive strategiy for ending low pay, notably by the introduction of a statutory national minimum wage.
Classification: Liberal
---
Sentence: {text}
Classification: 
"""
                    }
                ],
            },
            {
                'description': '[few-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                # from Ornstein et al. (2025)
                'prompt_text': [
                    {
                        "role": "user",
                        "content": "Decide whether this sentence from a British manifesto is Liberal or Conservative."
                    },
                    {
                        "role": "user",
                        "content": """Sentence: We will implement a comprehensive strategiy for ending low pay, notably by the introduction of a statutory national minimum wage.
Classification: """
                    },
                    {
                        "role": "assistant",
                        "content": "Liberal"
                    },
                    {
                        "role": "user",
                        "content": """Sentence: {text}
Classification: """
                    },
                ],
            },
            {
                'description': '[zero-shot-short-number] [original]',
                'compatible_output_mapping': [
                    'default output mapping score'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # from Ornstein et al. (2025)
                        "content": """You will be provided with a text from a party manifesto. Where does this text stand on the 'liberal' to 'conservative' scale, in terms of social policy? Provide your response as a score between 0 and 100 where 0 means 'Extremely liberal' and 100 means 'Extremely conservative'. Respond *only* with your score.
Sentence: {text}
Score: 
"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[zero-shot-short] [orig*] barrie',
                'original_description': '[zero-shot-short] [original] barrie',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": "Read the following sentence from a political manifesto discussing social policy. Decide whether it reflects a liberal or conservative stance. Respond with exactly one word—either 'liberal' or 'conservative'—and nothing else.\n\nSentence: {text}\n\nAnswer:"
                    }
                ],
            },
        ]


class ManifestosUKEconIdeologyDataset(ManifestosUK):
    def __init__(self, data_directory='manifestos_uk', config_fn='config_manifestos_econ_ideology.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'left': 'left',
            'right': 'right',
        }
        dict_with_mapping_options_with_letters = {
            'A': 'left',
            'B': 'right',
        }
        dict_with_mapping_options_score = {
            '100': 'right',
            '99': 'right',
            '98': 'right',
            '97': 'right',
            '96': 'right',
            '95': 'right',
            '94': 'right',
            '93': 'right',
            '92': 'right',
            '91': 'right',
            '90': 'right',
            '89': 'right',
            '88': 'right',
            '87': 'right',
            '86': 'right',
            '85': 'right',
            '84': 'right',
            '83': 'right',
            '82': 'right',
            '81': 'right',
            '80': 'right',
            '79': 'right',
            '78': 'right',
            '77': 'right',
            '76': 'right',
            '75': 'right',
            '74': 'right',
            '73': 'right',
            '72': 'right',
            '71': 'right',
            '70': 'right',
            '69': 'right',
            '68': 'right',
            '67': 'right',
            '66': 'right',
            '65': 'right',
            '64': 'right',
            '63': 'right',
            '62': 'right',
            '61': 'right',
            '60': 'right',
            '59': 'right',
            '58': 'right',
            '57': 'right',
            '56': 'right',
            '55': 'right',
            '54': 'right',
            '53': 'right',
            '52': 'right',
            '51': 'right',
            '50': 'right',
            '49': 'left',
            '48': 'left',
            '47': 'left',
            '46': 'left',
            '45': 'left',
            '44': 'left',
            '43': 'left',
            '42': 'left',
            '41': 'left',
            '40': 'left',
            '39': 'left',
            '38': 'left',
            '37': 'left',
            '36': 'left',
            '35': 'left',
            '34': 'left',
            '33': 'left',
            '32': 'left',
            '31': 'left',
            '30': 'left',
            '29': 'left',
            '28': 'left',
            '27': 'left',
            '26': 'left',
            '25': 'left',
            '24': 'left',
            '23': 'left',
            '22': 'left',
            '21': 'left',
            '20': 'left',
            '19': 'left',
            '18': 'left',
            '17': 'left',
            '16': 'left',
            '15': 'left',
            '14': 'left',
            '13': 'left',
            '12': 'left',
            '11': 'left',
            '10': 'left',
            '9': 'left',
            '8': 'left',
            '7': 'left',
            '6': 'left',
            '5': 'left',
            '4': 'left',
            '3': 'left',
            '2': 'left',
            '1': 'left',
            '0': 'left',
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
            'default output mapping with letters': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options_with_letters}
            },
            'default output mapping score': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options_score}
            },
        }

    def get_reported_results(self):
        return []


    def get_annotation_mapping(self):
        """Return the annotation mapping for economic ideology classification."""
        return {
            -1: 'left',
            1: 'right'
        }

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the economic policy stance expressed in the following sentence is {previous_answer_placeholder}-wing?

Output only a single number between 0 and 1, without any context or explanation.

Sentence: {text}

Probability:'''

    def get_prompts(self):
        return [
            {
                'description': '[zero-shot-short] [original] barrie',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # from Barrie et al. (2025)
                        "content": """This task involves reading some sentences from political manifestos. You are given a sentence that refers to Economic policy. For this sentence, enter your best judgment about whether the policy is left or right wing. Answer only with one word 'left' or 'right'.

Sentence: {text}

Answer: """
                    }
                ],
            },
            {
                'description': '[zero-shot-short] [original] ornstein',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # modified from Ornstein et al. (2025)
                        "content": """Decide whether this sentence from a British manifesto is left or right, in terms of economic policy.
Sentence: {text}
Classification: 
"""
                    }
                ],
            },
            {
                'description': '[zero-shot-short-number] [original]',
                'compatible_output_mapping': [
                    'default output mapping score'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # from Ornstein et al. (2025)
                        "content": """You will be provided with a text from a party manifesto. Where does this text stand on the 'left' to 'right' wing scale, in terms of economic policy? Provide your response as a score between 0 and 100 where 0 means 'Extremely left' and 100 means 'Extremely right'. Respond *only* with your score.
Sentence: {text}
Score: 
"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[zero-shot-short] [orig*] barrie',
                'original_description': '[zero-shot-short] [original] barrie',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": "Examine the following sentence drawn from a political manifesto and judge whether its economic policy stance is left-wing or right-wing. Respond with exactly one word—either 'left' or 'right'—matching the keys in {{'left': 'left', 'right': 'right'}}. Provide no additional text.\n\nSentence: {text}\n\nAnswer: "
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[zero-shot-short] [orig*] ornstein',
                'original_description': '[zero-shot-short] [original] ornstein',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Identify whether the following sentence from a UK political manifesto reflects a left-wing or right-wing economic stance.\nSentence: {text}\nReply with exactly one word—either "left" or "right"—nothing else.\nClassification:'
                    }
                ],
            },
        ]
