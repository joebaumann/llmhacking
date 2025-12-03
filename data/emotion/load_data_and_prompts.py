from data.data_utils import MyDataLoader
import copy


class EmotionDataset(MyDataLoader):
    def __init__(self, data_directory='emotion', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):

        dict_with_mapping_options = {
            'A': 'neutral',
            'B': 'admiration',
            'C': 'gratitude',
            'D': 'amusement',
            'E': 'love',
            'F': 'curiosity',
            'G': 'anger',
            'H': 'disapproval',
            'I': 'confusion',
            'J': 'annoyance',
            'K': 'fear',
            'L': 'optimism',
            'M': 'joy',
            'N': 'caring',
            'O': 'sadness',
            'P': 'surprise',
            'Q': 'disappointment',
            'R': 'excitement',
            'S': 'disgust',
            'T': 'desire',
            'U': 'realization',
            'V': 'remorse',
            'W': 'embarrassment',
            'X': 'nervousness',
            'Y': 'relief',
            'Z': 'approval',
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

        subreddit_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'subreddit', n=30)
        author_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'author')
        sentiment_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'sentiment')

        temporal_split_groups = self.add_temporal_split_groupings(
            timestamp_col='created_utc', date_unit='s')

        # Combine parent and specific groupings
        return subreddit_groupings + author_groupings + sentiment_groupings + temporal_split_groups
    
    def get_cols_to_stratify(self):
        return [
            'ground_truth',
            ('subreddit', 30),
            ('author', 10),
            ('sentiment', 10),
        ]

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the primary emotion expressed in the following text is {previous_answer_placeholder}?

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
                        # see Appendix A in https://aclanthology.org/2020.acl-main.372.pdf
                        "content": """Text: {text}

Please identify the primary emotion expressed by the writer in the text above. Choose from the following 25 emotions or neutral:

A: neutral (Not certain about any emotion being expressed.)
B: admiration (Finding something impressive or worthy of respect.)
C: gratitude (A feeling of thankfulness and appreciation.)
D: amusement (Finding something funny or being entertained.)
E: love (A strong positive emotion of regard and affection.)
F: curiosity (A strong desire to know or learn something.)
G: anger (A strong feeling of displeasure or antagonism.)
H: disapproval (Having or expressing an unfavorable opinion.)
I: confusion (Lack of understanding, uncertainty.)
J: annoyance (Mild anger, irritation.)
K: fear (Being afraid or worried.)
L: optimism (Hopefulness and confidence about the future or the success of something.)
M: joy (A feeling of pleasure and happiness.)
N: caring (Displaying kindness and concern for others.)
O: sadness (Emotional pain, sorrow.)
P: surprise (Feeling astonished, startled by something unexpected.)
Q: disappointment (Sadness or displeasure caused by the nonfulfillment of one's hopes or expectations.)
R: excitement (Feeling of great enthusiasm and eagerness.)
S: disgust (Revulsion or strong disapproval aroused by something unpleasant or offensive.)
T: desire (A strong feeling of wanting something or wishing for something to happen.)
U: realization (Becoming aware of something.)
V: remorse (Regret or guilty feeling.)
W: embarrassment (Self-consciousness, shame, or awkwardness.)
X: nervousness (Apprehension, worry, anxiety.)
Y: relief (Reassurance and relaxation following release from anxiety or distress.)
Z: approval (Having or expressing a favorable opinion.)

Output only the corresponding letter.

Answer:"""
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
                        # see Appendix A in https://aclanthology.org/2020.acl-main.372.pdf
                        "content": """Text: {text}

Please identify the primary emotion expressed by the writer in the text above. Choose from the following 25 emotions or neutral:

A: neutral (Not certain about any emotion being expressed.)
B: admiration (Finding something impressive or worthy of respect.)
C: gratitude (A feeling of thankfulness and appreciation.)
D: amusement (Finding something funny or being entertained.)
E: love (A strong positive emotion of regard and affection.)
F: curiosity (A strong desire to know or learn something.)
G: anger (A strong feeling of displeasure or antagonism.)
H: disapproval (Having or expressing an unfavorable opinion.)
I: confusion (Lack of understanding, uncertainty.)
J: annoyance (Mild anger, irritation.)
K: fear (Being afraid or worried.)
L: optimism (Hopefulness and confidence about the future or the success of something.)
M: joy (A feeling of pleasure and happiness.)
N: caring (Displaying kindness and concern for others.)
O: sadness (Emotional pain, sorrow.)
P: surprise (Feeling astonished, startled by something unexpected.)
Q: disappointment (Sadness or displeasure caused by the nonfulfillment of one's hopes or expectations.)
R: excitement (Feeling of great enthusiasm and eagerness.)
S: disgust (Revulsion or strong disapproval aroused by something unpleasant or offensive.)
T: desire (A strong feeling of wanting something or wishing for something to happen.)
U: realization (Becoming aware of something.)
V: remorse (Regret or guilty feeling.)
W: embarrassment (Self-consciousness, shame, or awkwardness.)
X: nervousness (Apprehension, worry, anxiety.)
Y: relief (Reassurance and relaxation following release from anxiety or distress.)
Z: approval (Having or expressing a favorable opinion.)

Emotion categories by sentiment (these reflect general tendencies only, so ignore this categorization whenever it doesn't fit the specific context):

Positive: D (amusement), R (excitement), M (joy), E (love), T (desire), L (optimism), N (caring), B (admiration), C (gratitude), Y (relief), Z (approval)
Negative: G (anger), J (annoyance), H (disapproval), Q (disappointment), S (disgust), W (embarrassment), K (fear), X (nervousness), V (remorse), O (sadness)
Ambiguous: F (curiosity), I (confusion), U (realization), P (surprise)
Neutral: A (neutral)

Output only the corresponding letter.

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
                        "content": """Text: {text}

Determine the single, predominant emotion the author conveys. Select exactly one option from the list and reply with its one-letter code only.

A: neutral – no clear emotion  
B: admiration – feeling impressed or respectful  
C: gratitude – thankfulness and appreciation  
D: amusement – finding something funny or entertaining  
E: love – strong affection or fondness  
F: curiosity – eagerness to know or learn  
G: anger – intense displeasure or hostility  
H: disapproval – negative judgment or criticism  
I: confusion – uncertainty or lack of understanding  
J: annoyance – irritation or mild anger  
K: fear – feeling afraid or worried  
L: optimism – hopefulness about the future  
M: joy – happiness or delight  
N: caring – kindness and concern for others  
O: sadness – sorrow or unhappiness  
P: surprise – astonishment at something unexpected  
Q: disappointment – unhappiness over unmet expectations  
R: excitement – great enthusiasm or eagerness  
S: disgust – revulsion or strong distaste  
T: desire – strong wanting or wishing  
U: realization – sudden understanding or awareness  
V: remorse – regret or guilt  
W: embarrassment – self-conscious shame or awkwardness  
X: nervousness – anxiety or apprehension  
Y: relief – comfort following stress or worry  
Z: approval – positive endorsement or agreement

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
                        "content": "Passage: {text}\n\nDetermine the primary emotion conveyed by the author. Select exactly one choice from the list below—25 emotions plus neutral—and return ONLY its corresponding capital letter (A-Z).\n\nA: neutral (no clear emotion)\nB: admiration (respect, being impressed)\nC: gratitude (thankfulness, appreciation)\nD: amusement (finding something funny or entertaining)\nE: love (strong positive affection)\nF: curiosity (eagerness to know or learn)\nG: anger (strong displeasure or hostility)\nH: disapproval (expressing an unfavorable opinion)\nI: confusion (uncertainty, lack of understanding)\nJ: annoyance (irritation, mild anger)\nK: fear (feeling afraid or worried)\nL: optimism (hopefulness about the future)\nM: joy (pleasure, happiness)\nN: caring (kindness, concern for others)\nO: sadness (sorrow, emotional pain)\nP: surprise (astonishment, being startled)\nQ: disappointment (distress over unmet expectations)\nR: excitement (great enthusiasm, eagerness)\nS: disgust (revulsion, strong aversion)\nT: desire (strong wish or want)\nU: realization (sudden awareness)\nV: remorse (regret, guilt)\nW: embarrassment (self-consciousness, shame)\nX: nervousness (apprehension, anxiety)\nY: relief (ease after stress)\nZ: approval (favorable opinion)\n\nOutput a single uppercase letter only."
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
                        "content": """Text: {text}

Your task: Identify the predominant emotion the author expresses. Pick exactly one option from the list below and reply with its single-letter code only (A–Z).

A: neutral – no clear emotion detected  
B: admiration – respect or being impressed  
C: gratitude – thankfulness, appreciation  
D: amusement – finding something funny/entertaining  
E: love – strong affection  
F: curiosity – desire to know or learn  
G: anger – intense displeasure or hostility  
H: disapproval – unfavorable opinion  
I: confusion – uncertainty, lack of understanding  
J: annoyance – irritation, mild anger  
K: fear – feeling afraid or worried  
L: optimism – hopeful confidence about the future  
M: joy – pleasure, happiness  
N: caring – kindness, concern for others  
O: sadness – sorrow, emotional pain  
P: surprise – astonishment at the unexpected  
Q: disappointment – displeasure from unmet expectations  
R: excitement – great enthusiasm  
S: disgust – revulsion, strong dislike  
T: desire – strong wanting  
U: realization – sudden awareness  
V: remorse – regret or guilt  
W: embarrassment – self-conscious shame or awkwardness  
X: nervousness – anxiety, apprehension  
Y: relief – easing of distress  
Z: approval – favorable opinion

Output: the single corresponding letter only."""
                    }
                ],
            },            
        ]
