from data.data_utils import MyDataLoader
import copy


class FactualityDataset(MyDataLoader):
    def __init__(self, data_directory='factuality', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'True': 'Supported',
            'False': 'Not-supported',
            'Irrelevant': 'Irrelevant',
        }
        dict_with_mapping_options_letters = {
            'A': 'Supported',
            'B': 'Not-supported',
            'C': 'Irrelevant',
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

        topic_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'topic')
        generation_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'generation')
        model_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'model')

        # Combine parent and specific groupings
        return topic_groupings + generation_groupings + model_groupings

    def get_cols_to_stratify(self):
        return [
            "ground_truth",
            ("topic", 10),
            ("generation", 10),
            ("model", 10),
        ]

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following statement is {previous_answer_placeholder} with respect to the topic "{topic}"?

Output only a single number between 0 and 1, without any context or explanation.

Statement: {text}

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
                        # prompt template and formatting taken from https://github.com/shmsw25/FActScore/blob/main/factscore/factscorer.py#L223
                        # see also Sections 3.3 and 4.1 in https://aclanthology.org/2023.emnlp-main.741.pdf
                        "content": """Answer the question about {topic}.

Input: {text} True or False?

If the input is clearly not related to {topic}, output Irrelevant. Output a single word only.
Output:"""
                    }
                ],
            },
            {
                'description': '[detailed] [zero-shot-detailed]',
                'compatible_output_mapping': [
                    'default output mapping letters'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # see Sections 3.3 and 4.1 in https://aclanthology.org/2023.emnlp-main.741.pdf
                        "content": """### TOPIC:
{topic}

### ATOMIC FACT:
{text}

### TASK:
Evaluate the atomic fact in relation to the given topic. First determine if the fact is relevant to the topic. If relevant, assess its factual correctness.

### CLASSIFICATION CRITERIA:
- A: SUPPORTED - The fact is clearly related to the topic AND is factually correct
- B: NOT-SUPPORTED - The fact is related to the topic BUT is factually incorrect or unverifiable
- C: IRRELEVANT - The fact is not related to the topic (regardless of its truthfulness)

Provide ONLY the letter A, B, or C as your answer.

### ANSWER:"""
                    }
                ],
            },
            {
                'description': '[zero-shot-detailed] [with-full-output]',
                'compatible_output_mapping': [
                    'default output mapping letters'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """### PROMPT:
Tell me a bio of {topic}.

### GENERATED BIO:
{generation}

### ATOMIC FACT:
{text}

### TASK:
The GENERATED BIO is an LLM-generated response to the prompt. The ATOMIC FACT is a short sentence conveying one piece of information extracted from the GENERATED BIO.
Evaluate the ATOMIC FACT in relation to the prompt. First, determine if the fact is relevant to the prompt. If relevant, assess its factual correctness.

### CLASSIFICATION CRITERIA:
- A: SUPPORTED - The ATOMIC FACT is definitely correct.
- B: NOT-SUPPORTED - The ATOMIC FACT is factually incorrect or unverifiable.
- C: IRRELEVANT - The ATOMIC FACT is not related to the prompt.

Provide ONLY the letter A, B, or C as your answer.

### ANSWER:"""
                    }
                ],
            },
            {
                'description': '[few-shot-detailed]',
                'compatible_output_mapping': [
                    'default output mapping letters'
                ],
                # see Table 7 and Sections 3.3 and 4.1 in https://aclanthology.org/2023.emnlp-main.741.pdf
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """### TASK:
Evaluate the atomic fact in relation to the given topic. First determine if the fact is relevant to the topic. If relevant, assess its factual correctness.

### CLASSIFICATION CRITERIA:
- A: SUPPORTED - The fact is clearly related to the topic AND is factually correct
- B: NOT-SUPPORTED - The fact is related to the topic BUT is factually incorrect or unverifiable
- C: IRRELEVANT - The fact is not related to the topic (regardless of its truthfulness)

Provide ONLY the letter A, B, or C as your answer.

### TOPIC:
Ylona Garcia

### ATOMIC FACT:
Ylona Garcia has appeared in various TV shows.

### ANSWER:"""
                    },
                    {
                        "role": "assistant",
                        "content": "A"
                    },
                    {
                        "role": "user",
                        "content": """### TOPIC:
Ylona Garcia

### ATOMIC FACT:
She has appeared in ASAP.

### ANSWER:"""
                    },
                    {
                        "role": "assistant",
                        "content": "A"
                    },
                    {
                        "role": "user",
                        "content": """### TOPIC:
Ylona Garcia

### ATOMIC FACT:
She has appeared in Wansapanataym Presents: Annika PINTAsera.

### ANSWER:"""
                    },
                    {
                        "role": "assistant",
                        "content": "B"
                    },
                    {
                        "role": "user",
                        "content": """### TOPIC:
John Estes

### ATOMIC FACT:
William Estes is an American.

### ANSWER:"""
                    },
                    {
                        "role": "assistant",
                        "content": "C"
                    },
                    {
                        "role": "user",
                        "content": """### TOPIC:
John Estes

### ATOMIC FACT:
William Estes' role on Blue Bloods is Jameson \\"Jamie\\" Reagan.

### ANSWER:"""
                    },
                    {
                        "role": "assistant",
                        "content": "C"
                    },
                    {
                        "role": "user",
                        "content": """### TOPIC:
{topic}

### ATOMIC FACT:
{text}

### ANSWER:"""
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
                        "content": 'Assess the claim about {topic}.\n\nInput: {text}\n\nIs this claim True or False?\n\nIf the input is not clearly related to {topic}, reply with Irrelevant.\n\nRespond with exactly one word: True, False, or Irrelevant.\nOutput:'
                    }
                ],
            },
        ]
