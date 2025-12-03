from data.data_utils import MyDataLoader
import copy


class ToneDataset(MyDataLoader):
    def __init__(self, data_directory='tone', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'positive': 'positive',
            'neutral': 'neutral',
            'negative': 'negative',
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
        return []

    def get_cols_to_stratify(self):
        return [
            'ground_truth',
        ]

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the tone of the following political advertisement is {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Political advertisement: {text}

Probability:'''


    def get_prompts(self):
        return [
            {
                'description': '[original] [zero-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                # see Figure 3 in Ornstein
                'prompt_text': [
                    {
                        "role": "system",
                        "content": """Decide whether the tone of a political advertisement is positive, neutral, or negative.
Ad Text: [Announcer]: Mark Udall claims he voted for tax cuts 65 times.
[Reporter]: "Actually it's not true."
[Mark Udall]: "I'm just kidding."
Udall claims that now he supports our troops, drilling off shore, bi-partisanship.
[Mark Udall]: "I'm just kidding."
[Announcer]: But through his career he has opposed them all. Udall has a lot of ads, but a shortage of truth.
[Mark Udall]: "I'm just kidding."
[Announcer]: Mark Udall is not honest about his past and now it's catching up with him.
The National Republican Senatorial Committee is responsible for the content of this advertising.
[PFB]: NATIONAL REPUBLICAN SENATORIAL COMMITTEE (9)
Tone: Negative"""
                    },
                    {
                        "role": "user",
                        "content": """Ad Text: {text}
Tone:"""
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
                        "role": "system",
                        "content": """Decide whether the tone of a political advertisement is positive, neutral, or negative."""
                    },
                    {
                        "role": "user",
                        "content": """Ad Text: [Announcer]: Mark Udall claims he voted for tax cuts 65 times.
[Reporter]: "Actually it's not true."
[Mark Udall]: "I'm just kidding."
Udall claims that now he supports our troops, drilling off shore, bi-partisanship.
[Mark Udall]: "I'm just kidding."
[Announcer]: But through his career he has opposed them all. Udall has a lot of ads, but a shortage of truth.
[Mark Udall]: "I'm just kidding."
[Announcer]: Mark Udall is not honest about his past and now it's catching up with him.
The National Republican Senatorial Committee is responsible for the content of this advertising.
[PFB]: NATIONAL REPUBLICAN SENATORIAL COMMITTEE (9)
Tone:"""
                    },
                    {
                        "role": "assistant",
                        "content": """Negative"""
                    },
                    {
                        "role": "user",
                        "content": """Ad Text: {text}
Tone:"""
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[orig*] [zero-shot-short]',
                'original_description': '[original] [zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {'role': 'system', 'content': 'Your task is to identify the overall tone of a political advertisement as positive, neutral, or negative.\n\nExample\nAd Text:\n[Announcer]: Mark Udall claims he voted for tax cuts 65 times.\n[Reporter]: \"Actually it\'s not true.\"\n[Mark Udall]: \"I\'m just kidding.\"\nUdall claims that now he supports our troops, drilling off shore, bi-partisanship.\n[Mark Udall]: \"I\'m just kidding.\"\n[Announcer]: But throughout his career he has opposed them all. Udall has a lot of ads, but a shortage of truth.\n[Mark Udall]: \"I\'m just kidding.\"\n[Announcer]: Mark Udall is not honest about his past and now it\'s catching up with him.\nThe National Republican Senatorial Committee is responsible for the content of this advertising.\n[PFB]: NATIONAL REPUBLICAN SENATORIAL COMMITTEE (9)\nTone: negative\n\nWhen answering, output exactly one of these words and nothing else:\npositive\nneutral\nnegative'},
                    {'role': 'user', 'content': 'Ad Text: {text}\nTone:'}]
            },
            {
                'description': '[paraphrase-nr-1] orig:[orig*] [zero-shot-short]',
                'original_description': '[original] [zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {'role': 'system', 'content': 'Identify the overall tone of the following political advertisement. Reply with exactly one word—positive, neutral, or negative—and nothing else.\n\nExample\nAd Text: [Announcer]: Mark Udall claims he voted for tax cuts 65 times.\n[Reporter]: "Actually it\'s not true."\n[Mark Udall]: "I\'m just kidding."\nUdall claims that now he supports our troops, drilling off shore, bi-partisanship.\n[Mark Udall]: "I\'m just kidding."\n[Announcer]: But through his career he has opposed them all. Udall has a lot of ads, but a shortage of truth.\n[Mark Udall]: "I\'m just kidding."\n[Announcer]: Mark Udall is not honest about his past and now it\'s catching up with him.\nThe National Republican Senatorial Committee is responsible for the content of this advertising.\n[PFB]: NATIONAL REPUBLICAN SENATORIAL COMMITTEE (9)\nTone: negative'},
                    {'role': 'user', 'content': 'Ad Text: {text}\nTone:'}]
            },
            {
                'description': '[paraphrase-nr-2] orig:[orig*] [zero-shot-short]',
                'original_description': '[original] [zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {'role': 'system', 'content': 'Classify the tone of a political advertisement as positive, neutral, or negative, and reply with only one of these labels.\n\nExample\n-------\nAd Text:\n[Announcer]: Mark Udall claims he voted for tax cuts 65 times.\n[Reporter]: "Actually it\'s not true."\n[Mark Udall]: "I\'m just kidding."\nUdall claims that now he supports our troops, drilling off shore, bi-partisanship.\n[Mark Udall]: "I\'m just kidding."\n[Announcer]: But throughout his career he has opposed them all. Udall has a lot of ads, but a shortage of truth.\n[Mark Udall]: "I\'m just kidding."\n[Announcer]: Mark Udall is not honest about his past and now it\'s catching up with him.\nThe National Republican Senatorial Committee is responsible for the content of this advertising.\n[PFB]: NATIONAL REPUBLICAN SENATORIAL COMMITTEE (9)\nTone: negative'},
                    {'role': 'user', 'content': 'Ad Text: {text}\nTone:'}]
            },
        ]
