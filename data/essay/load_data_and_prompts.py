from data.data_utils import MyDataLoader
import pdb


class EssayDataset(MyDataLoader):

    def get_dataset_specific_groups(self, df):
        return \
            self.split_by_column_with_n_most_frequent_values(df, "sex of child") + \
            self.split_by_column_with_n_most_frequent_values(df, "standard region") + \
            self.split_by_column_with_n_most_frequent_values(df, "date on schools questionnaire-month") + \
            self.split_by_column_with_n_most_frequent_values(df, "date on schools questionnaire-year") + \
            self.split_by_column_with_n_most_frequent_values(df, "sex of school head or principal") + \
            self.split_by_column_with_n_most_frequent_values(df, "sex of childs class teacher") + \
            self.split_by_column_with_n_most_frequent_values(df, "date on test booklet-month") + \
            self.split_by_column_with_n_most_frequent_values(df, "verbal score on general ability test") + \
            self.split_by_column_with_n_most_frequent_values(df, "non verbal score on gen ability test") + \
            self.split_by_column_with_n_most_frequent_values(df, "total score on general ability test") + \
            self.split_by_column_with_n_most_frequent_values(df, "reading comprehension test score") + \
            self.split_by_column_with_n_most_frequent_values(df, "mathematics test score") + \
            self.split_by_column_with_n_most_frequent_values(df, "copying designs test score") + \
            self.split_by_column_with_n_most_frequent_values(df, "childs mother figure") + \
            self.split_by_column_with_n_most_frequent_values(df, "why child away from own-adoptive mum") + \
            self.split_by_column_with_n_most_frequent_values(df, "childs father figure") + \
            self.split_by_column_with_n_most_frequent_values(df, "why child away from own-adoptive dad") + \
            self.split_by_column_with_n_most_frequent_values(df, "social class of father or male head") + \
            self.split_by_column_with_n_most_frequent_values(df, "social class fathermale head-groupd") + \
            self.split_by_column_with_n_most_frequent_values(df, "pupils occupation at 25") + \
            self.split_by_column_with_n_most_frequent_values(df, "ability age 11") + \
            self.split_by_column_with_n_most_frequent_values(
                df, "background variable used for essays analysis")

    def get_cols_to_stratify(self):
        return [
            "ground_truth",
            ("sex of child", 10),
            ("standard region", 10),
            ("date on schools questionnaire-month", 10),
            ("date on schools questionnaire-year", 10),
            ("sex of school head or principal", 10),
            ("sex of childs class teacher", 10),
            ("date on test booklet-month", 10),
            ("verbal score on general ability test", 10),
            ("non verbal score on gen ability test", 10),
            ("total score on general ability test", 10),
            ("reading comprehension test score", 10),
            ("mathematics test score", 10),
            ("copying designs test score", 10),
            ("childs mother figure", 10),
            ("why child away from own-adoptive mum", 10),
            ("childs father figure", 10),
            ("why child away from own-adoptive dad", 10),
            ("social class of father or male head", 10),
            ("social class fathermale head-groupd", 10),
            ("pupils occupation at 25", 10),
            ("ability age 11", 10),
            ("background variable used for essays analysis", 10),
        ]


class EssayLivingDataset(EssayDataset):
    def __init__(self, data_directory='essay', config_fn='config_essay_living.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            "A": "Not mentioned",
            "B": "Alone",
            "C": "Still living with parents",
            "D": "Married",
            "E": "Cohabiting",
            "F": "Friends",
            "G": "With other family members",
            "H": "Changes during essay",
            "I": "Other",
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
        }

    def get_reported_results(self):
        return []

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the writer's anticipated living situation at age 25 falls under the category {previous_answer_placeholder} in the following essay?

Output only a single number between 0 and 1, without any context or explanation.

Essay: {text}

Probability:'''

    def get_prompts(self):
        return [
            {
                'description': '[zero-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Identify the writer's anticipated living situation at age 25. If the living situation changes during the essay (e.g., starts single then gets married), classify as "Changes during essay".

A: Not mentioned
B: Alone
C: Still living with parents
D: Married
E: Cohabiting
F: Friends
G: With other family members
H: Changes during essay
I: Other

Output only one letter.

Essay: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Read the following essay and choose the single letter that best describes where the writer expects to be living at age 25. If the essay depicts more than one living arrangement over its course (e.g., starts alone and later married), select “H: Changes during essay”.

A: Not mentioned  
B: Alone  
C: Still living with parents  
D: Married  
E: Cohabiting  
F: Friends  
G: With other family members  
H: Changes during essay  
I: Other  

Return only the letter.

Essay: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Read the essay below and choose the single category that describes where the writer expects to live at age 25. If the essay presents more than one future living arrangement (e.g., begins alone but later plans to marry), select “H: Changes during essay”.\n\nA: Not mentioned\nB: Alone\nC: Still living with parents\nD: Married\nE: Cohabiting\nF: Friends\nG: With other family members\nH: Changes during essay\nI: Other\n\nOutput only the corresponding letter (A–I).\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-2] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'From the essay below, decide where the writer expects to live when they turn 25. If the essay portrays more than one future arrangement (e.g., starts off single and later marries), label it as "H: Changes during essay".\n\nChoose exactly one code:\nA: Not mentioned\nB: Alone\nC: Still living with parents\nD: Married\nE: Cohabiting\nF: Friends\nG: With other family members\nH: Changes during essay\nI: Other\n\nReturn only the single letter.\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-3] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Determine the living arrangement the author envisions for themselves at age 25. If this envisioned arrangement shifts within the essay (e.g., starts single, later indicates marriage), select “H: Changes during essay”.\n\nA: Not mentioned\nB: Alone\nC: Still living with parents\nD: Married\nE: Cohabiting\nF: Friends\nG: With other family members\nH: Changes during essay\nI: Other\n\nRespond with ONE letter (A–I) only.\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
        ]


class EssayHousewifeDataset(EssayDataset):
    def __init__(self, data_directory='essay', config_fn='config_essay_housewife.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            "A": "No",
            "B": "Yes",
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
        }

    def get_reported_results(self):
        return []

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the answer to whether the writer of the following essay expects to be a housewife or care for home and family is {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Text: {text}

Probability:'''

    def get_prompts(self):
        return [
            {
                'description': '[zero-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Determine if the writer expects to be a housewife or care for home and family (instead of or alongside having an occupation).

A: No
B: Yes

Output only one letter.

Essay: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Read the essay and decide whether the writer indicates an intention to be a housewife or to take primary responsibility for home and family (either exclusively or alongside other employment).

Respond with one single-letter code only:
A: No
B: Yes

Essay: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Decide whether the author expects to be a housewife or chiefly handle home and family duties (either replacing or accompanying paid work).\n\nA: No\nB: Yes\n\nRespond with exactly one letter.\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-2] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": "Examine the essay below and judge whether the author expects to be a housewife or to take primary responsibility for home and family, either in place of or alongside paid work.\n\nLabels:\nA: No\nB: Yes\n\nReply with ONE letter only (A or B).\n\nEssay: {text}\n\nAnswer:"
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-3] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'After reading the essay below, determine whether the author envisions becoming a housewife or primarily managing household and family responsibilities (either alone or alongside paid work).\n\nA: No\nB: Yes\n\nRespond with exactly one letter: A or B.\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
        ]


class EssayWorksocialDataset(EssayDataset):
    def __init__(self, data_directory='essay', config_fn='config_essay_worksocial.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            "A": "No",
            "B": "Yes",
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
        }

    def get_reported_results(self):
        return []

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the answer to the question of whether the following essay mentions social relations at work is {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Essay: {text}

Probability:'''

    def get_prompts(self):
        return [
            {
                'description': '[zero-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Determine if social relations at work are mentioned (e.g., working with friends, interactions with boss, talking to colleagues).

A: No
B: Yes

Output only one letter.

Essay: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Decide whether the essay includes references to social interactions in the workplace (e.g., working alongside friends, communicating with a boss, talking with colleagues).\n\nA: No\nB: Yes\n\nReturn exactly one letter.\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Read the essay below and determine whether it mentions any workplace social interactions (e.g., collaborating with friends, talking to a boss, or chatting with colleagues).\n\nA: No\nB: Yes\n\nReply with a single letter (A or B) only.\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-2] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Decide if the essay references social interactions at work (e.g., collaborating with friends, communicating with a boss, chatting with coworkers).

Possible outputs:
A: No
B: Yes

Return only one letter (A or B).

Essay: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-3] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Decide whether the essay mentions social interactions in the workplace (e.g., teaming up with friends, communicating with a boss, chatting with colleagues).\n\nA: No\nB: Yes\n\nReply with ONLY one letter.\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
        ]


class EssayLocationDataset(EssayDataset):
    def __init__(self, data_directory='essay', config_fn='config_essay_location.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            "A": "Not mentioned",
            "B": "London",
            "C": "UK",
            "D": "Abroad",
            "E": "More than one location",
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
        }

    def get_reported_results(self):
        return []

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that, in the following essay, the writer expects to live at age 25 in the category {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Essay: {text}

Probability:'''

    def get_prompts(self):
        return [
            {
                'description': '[zero-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Identify where the writer expects to live at age 25. "UK" includes any specific UK location except London (e.g., Denton). "Abroad" includes any non-UK location (e.g., Spain).

A: Not mentioned
B: London
C: UK
D: Abroad
E: More than one location

Output only one letter.

Essay: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Read the essay and indicate where the author expects to live at age 25.\n\nGuidelines:\n- Choose “C” (UK) for any specific UK location other than London (e.g., Denton).\n- Choose “D” (Abroad) for any location outside the UK (e.g., Spain).\n\nLabels:\nA: Not mentioned\nB: London\nC: UK\nD: Abroad\nE: More than one location\n\nReturn only the single letter that matches.\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'From the essay, determine where the writer expects to live at the age of 25. Use these definitions:\n- "UK": any location within the United Kingdom except London (e.g., Denton)\n- "Abroad": any location outside the UK (e.g., Spain)\n\nChoose only one of the labels below:\nA: Not mentioned\nB: London\nC: UK\nD: Abroad\nE: More than one location\n\nRespond with a single letter from A–E.\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-2] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Review the essay and decide where the writer expects to live at age 25. Treat any UK location other than London (e.g., Denton) as "UK" and any location outside the UK (e.g., Spain) as "Abroad".\n\nChoose one code from the list and respond with that single letter only:\nA: Not mentioned\nB: London\nC: UK\nD: Abroad\nE: More than one location\n\nEssay: {text}\n\nResponse:'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-3] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Determine the place the writer expects to live at age 25 and assign the correct code.\n\nA: Not mentioned\nB: London\nC: UK (any UK location except London, e.g., Denton)\nD: Abroad (any non-UK location, e.g., Spain)\nE: More than one location\n\nReturn ONLY one letter (A–E).\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
        ]


class EssayDomesticDataset(EssayDataset):
    def __init__(self, data_directory='essay', config_fn='config_essay_domestic.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            "A": "Not mentioned",
            "B": "Childcare",
            "C": "Cooking cleaning etc",
            "D": "Childcare & cooking cleaning",
            "E": "DIY or gardening",
            "F": "Multiple tasks",
            "G": "Sewing, knitting, needlework",
            "H": "Other",
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
        }

    def get_reported_results(self):
        return []

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the correct category for the domestic-labor mention in the following essay is {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Essay: {text}

Probability:'''

    def get_prompts(self):
        return [
            {
                'description': '[zero-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Identify mentions of domestic labor, whether done by self or others (e.g., "my wife will look after the children", "we will have a maid"). Multiple different tasks should be classified as "Multiple tasks".

A: Not mentioned
B: Childcare
C: Cooking cleaning etc
D: Childcare & cooking cleaning
E: DIY or gardening
F: Multiple tasks
G: Sewing, knitting, needlework
H: Other

Output only one letter.

Essay: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Identify any mention of domestic work performed by the writer or by someone else (e.g., “my wife will look after the children”, “we will have a maid”). If several different kinds of tasks appear, choose “Multiple tasks”.

Codes:
A: Not mentioned  
B: Childcare  
C: Cooking cleaning etc  
D: Childcare & cooking cleaning  
E: DIY or gardening  
F: Multiple tasks  
G: Sewing, knitting, needlework  
H: Other  

Return exactly one letter from the list above—nothing more.

Essay: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Examine the essay and determine whether it references any household work carried out by the writer or by someone else (e.g., "my wife will watch the kids", "we will hire a maid"). If the essay mentions more than one distinct type of work, select "Multiple tasks".\n\nReply with exactly one of these codes and nothing else:\nA: Not mentioned\nB: Childcare\nC: Cooking cleaning etc\nD: Childcare & cooking cleaning\nE: DIY or gardening\nF: Multiple tasks\nG: Sewing, knitting, needlework\nH: Other\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-2] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Read the essay below and decide whether it includes any reference to domestic work carried out by the writer or by someone else (e.g., "my wife will watch the kids", "we will hire a maid"). Classify the reference using exactly one of the following labels:\n\nA: Not mentioned\nB: Childcare\nC: Cooking cleaning etc\nD: Childcare & cooking cleaning\nE: DIY or gardening\nF: Multiple tasks\nG: Sewing, knitting, needlework\nH: Other\n\nChoose "F" if several distinct kinds of domestic tasks are mentioned. \n\nReturn ONLY the single corresponding letter.\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-3] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Read the following essay and decide whether it references any domestic work done by the writer or by someone else (e.g., "my wife will look after the children", "we will have a maid"). Choose the single best category below; if the essay mentions more than one distinct type of task, select "F: Multiple tasks".\n\nReturn only the corresponding letter:\nA: Not mentioned\nB: Childcare\nC: Cooking cleaning etc\nD: Childcare & cooking cleaning\nE: DIY or gardening\nF: Multiple tasks\nG: Sewing, knitting, needlework\nH: Other\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
        ]


class EssayNarrativeDataset(EssayDataset):
    def __init__(self, data_directory='essay', config_fn='config_essay_narrative.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            "A": "No",
            "B": "Some narrative clauses",
            "C": "Extended narrative",
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
        }

    def get_reported_results(self):
        return []

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following essay shows {previous_answer_placeholder} content (where narrative refers to sequential events or life progression)?

Output only a single number between 0 and 1, without any context or explanation.

Essay: {text}

Probability:'''

    def get_prompts(self):
        return [
            {
                'description': '[zero-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Assess narrative elements: pure descriptions with no time progression are "No"; essays with sequential events (this happened then that) are "Some narrative clauses"; extended stories about life progression are "Extended narrative".

A: No
B: Some narrative clauses
C: Extended narrative

Output only one letter.

Essay: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Determine the essay’s narrative level.\n\nA: No – purely descriptive; no temporal progression.\nB: Some narrative clauses – a brief chain of events (e.g., “this happened, then that happened”).\nC: Extended narrative – a fuller, longer-term story about life or events.\n\nRespond with exactly one letter (A, B, or C).\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Evaluate the essay’s narrativity using the mapping below.

Mapping:
{{'A': 'No', 'B': 'Some narrative clauses', 'C': 'Extended narrative'}}

Guidelines
A – No: purely descriptive writing with no progression through time.  
B – Some narrative clauses: a brief sequence of events (e.g., “this happened, then that”).  
C – Extended narrative: a fuller story that traces events or life developments over time.

Return only the single letter A, B, or C.

Essay: {text}

Answer:"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-2] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Evaluate the essay’s narrative depth:\n\nA (No): Descriptive only, no unfolding timeline.\nB (Some narrative clauses): Contains a basic series of events (e.g., event after event).\nC (Extended narrative): Offers a comprehensive, continuous storyline about life progression.\n\nRespond with JUST one letter — A, B, or C.\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-3] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Determine the narrative level of the essay below.\n\nA: No – solely descriptive, with no progression over time.\nB: Some narrative clauses – a sequence of events occurs (e.g., “this happened, then that”).\nC: Extended narrative – a longer, ongoing storyline or life progression.\n\nReply with only one letter: A, B, or C.\n\nEssay: {text}\n\nAnswer:'
                    }
                ],
            },
        ]
