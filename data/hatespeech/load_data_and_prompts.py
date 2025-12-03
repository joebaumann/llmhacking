from data.data_utils import MyDataLoader
import copy


class HateSpeechDataset(MyDataLoader):
    def get_hatespeech_explanation_for_detailed_prompt(self):
        return """Help us determine the content of the texts we provide.
Imagine you are a content moderator tagging hate speech. Look at the questions, and mark the appropriate answers.

Definitions:

The Social Media Community defines discriminatory (hate) speech as content that promotes, attacks, or threatens other people based on their actual or perceived race, ethnicity, national origin, age, religious affiliation, sex, gender or gender identity, sexual orientation, disability or disease. Examples of posts tagged as hate on Twitter includes, but is not limited to behavior that harasses individuals or groups of people with:
- violent threats;
- wishes for the physical harm, death, or disease of individuals or groups;
- references to mass murder, violent events, or specific means of violence in which/with which such groups have been the primary targets or victims;
- behavior that incites fear about a protected group;
- repeated and/or non-consensual slurs, epithets, racist and sexist tropes, or other content that degrades someone.

Explicit hate speech description:
- Explicit hate speech often relies on the usage of specific keywords or expresses explicit hatred to attack a person or a group of people based on their protected and prominent properties such as ethnicity, race, ethnicity, national origin, religion, sex, gender, and sexual orientation.
- An example tweet of overt discriminatory language is: "@us 1 i'll tear your limbs apart and feed them to the f*cking sharks you n*gger".
- Another example is: "they all brown people even mexicans hate them all " " fact / hater, hate, hate hate !"

Implicit hate speech description:
- Constitutes more complex, abstract and coded expressions and attitudes with prejudicial views towards other individuals based on their prominent characteristics without the reliance on explicit hate language.
- Atwater explained in his quote how can discourse appeal to racists without sounding racist: "You start out in 1954 by saying, "N*gger, n*gger, n*gger." By 1968 you can't say "n*gger" — that hurts you, backfires. So you say stuff like, uh, forced busing, states' rights, and all that stuff, and you're getting so abstract. Now, you're talking about cutting taxes, and all these things you're talking about are totally economic things and a byproduct of them is, blacks get hurt worse than whites... "We want to cut this," is much more abstract than even the busing thing, uh, and a hell of a lot more abstract than "N*gger, n*gger.""
- An example tweet is "He Told The World: The Immortal Words of Adolf Hitler"
- Note that implicit hate speech can come in many shapes and forms such as sarcasm, stereotyping, frustration, degrading, and misinformation.

Not hate speech content description:
- Irrelevant content that does not attack a person or a group of people based on their protected characteristics.
- The content might still use offensive terms but does not attack people based on their protected characteristics.
Example tweets:
- "India has proven it can manufacture: general electric ceo - #indiatomorrow" (This post represents benign content)
- "We love the black race they are brothers you c*cks*ckers we are all one race" (This post contains an offensive term but does not attack people based on their protected characteristics.)

"""


class HateSpeechExplicitDataset(HateSpeechDataset):
    def __init__(self, data_directory='hatespeech/hatespeech_explicit', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'A': 'explicit hate',
            'B': 'implicit hate',
            'C': 'not hate',
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
        return '''How likely is it that the following text is {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Text: {text}

Probability:'''

    def get_prompts(self):
        short_prompt = """Does this text attack a person or a group of people, explicitly or implicitly, based on their protected characteristics?
- A: Yes explicit hate speech
- B: Yes implicit hate speech
- C: Not hate speech

Tip: Explicit hate speech uses explicit hatred expressions or offensive keywords such as n*gger, c*nt, etc. to attack a person or a group of people based on their protected characteristics.
Implicit hate speech is a more complex attack on a person or a group of people without relying on explicit keywords. Beware that implicit discriminatory speech may be subtle.
Protected characteristics include ethnicity, race, national origin, religion, sex, gender, and sexual orientation. Note that pointing out racism should not be considered as hate speech.

Answer A, B, or C only.

Answer:"""
        return [
            {
                'description': '[zero-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # see Appendix A in https://aclanthology.org/2021.emnlp-main.29.pdf
                        "content": "Text: {text}\n\n" + short_prompt
                    }
                ],
            },
            {
                'description': '[zero-shot-detailed]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # see Appendix A in https://aclanthology.org/2021.emnlp-main.29.pdf
                        "content": self.get_hatespeech_explanation_for_detailed_prompt() + "Text: {text}\n\n" + short_prompt
                    }
                ],
            },
            {
                'description': '[few-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                # see Appendix A in https://aclanthology.org/2021.emnlp-main.29.pdf
                'prompt_text': [
                    {
                        "role": "user",
                        "content": short_prompt
                    },
                    {
                        "role": "user",
                        "content": """@us 1 i'll tear your limbs apart and feed them to the f*cking sharks you n*gger"""
                    },
                    {
                        "role": "assistant",
                        "content": "A"
                    },
                    {
                        "role": "user",
                        "content": """they all brown people even mexicans hate them all " " fact / hater, hate, hate hate !"""
                    },
                    {
                        "role": "assistant",
                        "content": "A"
                    },
                    {
                        "role": "user",
                        "content": """He Told The World: The Immortal Words of Adolf Hitler"""
                    },
                    {
                        "role": "assistant",
                        "content": "B"
                    },
                    {
                        "role": "user",
                        "content": """Text: India has proven it can manufacture: general electric ceo - #indiatomorrow"""
                    },
                    {
                        "role": "assistant",
                        "content": "C"
                    },
                    {
                        "role": "user",
                        "content": """Text: We love the black race they are brothers you c*cks*ckers we are all one race"""
                    },
                    {
                        "role": "assistant",
                        "content": "C"
                    },
                    {
                        "role": "user",
                        "content": """Text: {text}"""
                    },
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Text for review: {text}

Determine whether the text explicitly or implicitly targets a person or group because of a protected characteristic.

Reply with one single-letter label:

A – explicit hate  
   (openly hateful wording or slurs such as “n*gger,” “c*nt,” etc.)

B – implicit hate  
   (subtle or indirect hateful content without relying on overt slurs)

C – not hate  
   (no hateful attack; simply discussing racism is not hate speech)

Protected characteristics include race, ethnicity, national origin, religion, sex, gender, and sexual orientation.

Respond with A, B, or C only."""
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
                        "content": 'Text: {text}\n\nQuestion: Does this text attack an individual or group for their protected characteristic(s), either overtly or subtly?\n\nChoose one option:\n- A: Explicit hate – direct hateful language or slurs (e.g., n*gger, c*nt) aimed at the target.\n- B: Implicit hate – hateful content conveyed indirectly or through subtle means, without clear slurs.\n- C: Not hate – no hateful attack present (simply condemning racism is not hate speech).\n\nProtected characteristics include ethnicity, race, national origin, religion, sex, gender, and sexual orientation.\n\nReply with only the single letter A, B, or C.\n\nAnswer:'
                    }
                ],
            },
        ]


class HateSpeechImplicitDataset(HateSpeechDataset):
    def __init__(self, data_directory='hatespeech/hatespeech_implicit', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'A': 'white grievance',
            'B': 'incitement',
            'C': 'inferiority',
            'D': 'irony',
            'E': 'stereotypical',
            'F': 'threatening',
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
        return '''How likely is it that the following sentence best fits the hate-speech category {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Sentence: {text}

Probability:'''

    def get_prompts(self):
        short_prompt = """Which of the following categories of hate speech best describes the sentence above?
A: White Grievance (frustration over a minority group's perceived privilege and casting majority groups as the real victims of racism)
B: Incitement to Violence (flaunting in-group unity and power or elevating known hate groups and ideologies)
C: Inferiority Language (implies one group or individual is inferior to another, including dehumanization and toxification)
D: Irony (the use of sarcasm, humor, and satire to attack or demean a protected class or individual)
E: Stereotypes and Misinformation (associating a protected class with negative attributes)
F: Threatening and Intimidation (conveys a speaker commitment to a target's pain, injury, damage, loss, or violation of rights)

Constraint: Answer with one of the options above that is most accurate and nothing else. Always choose at least one of the options."""
        return [
            {
                'description': '[original] [zero-shot-short] [ziems]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # taken from Ziems et al.
                        "content": "{text}\n\n" + short_prompt
                    }
                ],
            },
            {
                'description': '[zero-shot-detailed]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # see Appendix A in https://aclanthology.org/2021.emnlp-main.29.pdf
                        "content": self.get_hatespeech_explanation_for_detailed_prompt() + "{text}\n\n" + short_prompt
                    }
                ],
            },
            {
                'description': '[few-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                # see Appendix A in https://aclanthology.org/2021.emnlp-main.29.pdf
                'prompt_text': [
                    {
                        "role": "user",
                        "content": short_prompt
                    },
                    {
                        "role": "user",
                        "content": "white revolution is the only solution"
                    },
                    {
                        "role": "assistant",
                        "content": "B"
                    },
                    {
                        "role": "user",
                        "content": "yes you are fine in a white majority country. how is mexico doing these days? people come here because you couldn't build it."
                    },
                    {
                        "role": "assistant",
                        "content": "C"
                    },
                    {
                        "role": "user",
                        "content": "black people only seem to thrive in white countries. how strange"
                    },
                    {
                        "role": "assistant",
                        "content": "D"
                    },
                    {
                        "role": "user",
                        "content": "soon white people will not be a majority anywhere. you believe"
                    },
                    {
                        "role": "assistant",
                        "content": "A"
                    },
                    {
                        "role": "user",
                        "content": "muslims are always wanting to kill someone! its their duty to kill the infidel."
                    },
                    {
                        "role": "assistant",
                        "content": "E"
                    },
                    {
                        "role": "user",
                        "content": "oh how i wish a man with a gun was there! wake up white people! they hate us"
                    },
                    {
                        "role": "assistant",
                        "content": "F"
                    },
                    {
                        "role": "user",
                        "content": "{text}"
                    },
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[orig*] [zero-shot-short] [ziems]',
                'original_description': '[original] [zero-shot-short] [ziems]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": "{text}\n\nChoose the hate-speech label (A–F) that most accurately fits the sentence above:\nA: White Grievance – resentment of a minority’s perceived privilege and portrayal of the majority as the real victims of racism\nB: Incitement to Violence – celebrating in-group power, endorsing violent action, or elevating established hate movements/ideologies\nC: Inferiority Language – asserting that a person or group is lesser, non-human, or tainted\nD: Irony – using sarcasm, humor, or satire to attack or demean a protected class or individual\nE: Stereotypes & Misinformation – linking a protected class to negative attributes or false claims\nF: Threatening & Intimidation – expressing an intent to cause pain, injury, damage, loss, or rights violations\n\nRespond with exactly one letter (A, B, C, D, E, or F) and nothing else."
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[orig*] [zero-shot-short] [ziems]',
                'original_description': '[original] [zero-shot-short] [ziems]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": "{text}\n\nChoose the single category from the list below that most accurately captures the hate-speech nature of the sentence above:\nA: White Grievance – resentment over minorities’ presumed advantages and portrayal of the majority as the true victims of racism\nB: Incitement – calls for or glorification of violence, or celebration of extremist groups and their power\nC: Inferiority – language depicting a person or group as lesser, dehumanized, or toxic\nD: Irony – sarcastic, humorous, or satirical attacks on a protected group or individual\nE: Stereotypical – attributing negative traits or spreading misinformation about a protected class\nF: Threatening – statements expressing an intention to cause harm, injury, loss, or rights violations\n\nAnswer strictly with one letter (A, B, C, D, E, or F) and nothing else."
                    }
                ],
            },
        ]


class HateSpeechTargetDataset(HateSpeechDataset):
    def __init__(self, data_directory='hatespeech/hatespeech_target', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'A': 'illegal immigrants',
            'B': 'minorities',
            'C': 'black people',
            'D': 'muslims',
            'E': 'jewish people',
            'F': 'non white people',
            'G': 'liberals',
            'H': 'progressives',
            'I': 'no specific group',
            'J': 'people of color',
            'K': 'democrats',
            'L': 'gay people',
            'M': 'islamic people',
            'N': 'leftists',
            'O': 'arabs',
            'P': 'mexican people',
            'Q': 'christians',
            'R': 'white men',
            'S': 'white women',
            'T': 'white conservatives',
            'U': 'conservative males',
            'V': 'antifa',
            'W': 'asian people',
            'X': 'white people',
            'Y': 'conservatives',
            'Z': 'women',
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

        implied_statement_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'implied_statement')
        implied_statement_without_target_info_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'implied_statement_without_target_info')

        # Combine parent and specific groupings
        return implied_statement_groupings + implied_statement_without_target_info_groupings

    def get_cols_to_stratify(self):
        return [
            'ground_truth',
            ("implied_statement", 10),
            ("implied_statement_without_target_info", 10),
        ]

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the implicit hate speech in the following tweet is directed at {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Tweet with implicit hate speech: {text}

Probability:'''

    def get_prompts(self):
        short_prompt = """The above tweet has been categorized as "implicit hate speech" in a prior annotation stage; a more complex and subtle attack on a person or a group of people based on their protected characteristics without relying on explicit keywords.
The goal of the task is to infer the targeted group even if it is not explicitly mentioned.
Which group of people does this post refer to?
A: illegal immigrants
B: minorities
C: black people
D: muslims
E: jewish people
F: non white people
G: liberals
H: progressives
I: no specific group
J: people of color
K: democrats
L: gay people
M: islamic people
N: leftists
O: arabs
P: mexican people
Q: christians
R: white men
S: white women
T: white conservatives
U: conservative males
V: antifa
W: asian people
X: white people
Y: conservatives
Z: women

Output a letter only.

Answer:"""
        return [
            {
                'description': '[zero-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # see Appendix A in https://aclanthology.org/2021.emnlp-main.29.pdf
                        "content": "Tweet: {text}\n\n" + short_prompt
                    }
                ],
            },
            {
                'description': '[zero-shot-detailed]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # see Appendix A in https://aclanthology.org/2021.emnlp-main.29.pdf
                        "content": self.get_hatespeech_explanation_for_detailed_prompt() + "Tweet: {text}\n\n" + short_prompt
                    }
                ],
            },
            {
                'description': '[few-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                # see Appendix A in https://aclanthology.org/2021.emnlp-main.29.pdf
                'prompt_text': [
                    {
                        "role": "user",
                        "content": short_prompt
                    },
                    {
                        "role": "user",
                        "content": "white revolution is the only solution"
                    },
                    {
                        "role": "assistant",
                        "content": "X: white people"
                    },
                    {
                        "role": "user",
                        "content": "yes you are fine in a white majority country. how is mexico doing these days? people come here because you couldn't build it."
                    },
                    {
                        "role": "assistant",
                        "content": "P: mexican people"
                    },
                    {
                        "role": "user",
                        "content": "black people only seem to thrive in white countries. how strange"
                    },
                    {
                        "role": "assistant",
                        "content": "C: black people"
                    },
                    {
                        "role": "user",
                        "content": "soon white people will not be a majority anywhere. you believe"
                    },
                    {
                        "role": "assistant",
                        "content": "B: minorities"
                    },
                    {
                        "role": "user",
                        "content": "muslims are always wanting to kill someone! its their duty to kill the infidel."
                    },
                    {
                        "role": "assistant",
                        "content": "D: muslims"
                    },
                    {
                        "role": "user",
                        "content": "oh how i wish a man with a gun was there! wake up white people! they hate us"
                    },
                    {
                        "role": "assistant",
                        "content": "F: non white people"
                    },
                    {
                        "role": "user",
                        "content": "{text}"
                    },
                ],
            },
            {
                'description': '[paraphrase-nr-0] orig:[zero-shot-short]',
                'original_description': '[zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": """Tweet: {text}

The tweet above has already been identified as containing implicit hate speech—an indirect or subtle attack on individuals or groups with protected characteristics that avoids explicit slurs.

Task: Infer which group is being targeted, even if the tweet never names them outright.

Choose exactly one option from the list and reply with ONLY its corresponding letter (A–Z).

A: illegal immigrants  
B: minorities  
C: black people  
D: muslims  
E: jewish people  
F: non white people  
G: liberals  
H: progressives  
I: no specific group  
J: people of color  
K: democrats  
L: gay people  
M: islamic people  
N: leftists  
O: arabs  
P: mexican people  
Q: christians  
R: white men  
S: white women  
T: white conservatives  
U: conservative males  
V: antifa  
W: asian people  
X: white people  
Y: conservatives  
Z: women  

Return only the single chosen letter.

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
                        "content": """Tweet: {text}

The tweet above has been previously tagged as “implicit hate speech,” meaning it contains a subtle or indirect attack against a protected group without overt slurs. Your task is to determine which group is being targeted, even if the group is not named explicitly.

Select the best-fitting option from the list below and reply with its single corresponding letter only—no extra text:

A: illegal immigrants  
B: minorities  
C: black people  
D: muslims  
E: jewish people  
F: non white people  
G: liberals  
H: progressives  
I: no specific group  
J: people of color  
K: democrats  
L: gay people  
M: islamic people  
N: leftists  
O: arabs  
P: mexican people  
Q: christians  
R: white men  
S: white women  
T: white conservatives  
U: conservative males  
V: antifa  
W: asian people  
X: white people  
Y: conservatives  
Z: women  

Answer:"""
                    }
                ],
            },
        ]
