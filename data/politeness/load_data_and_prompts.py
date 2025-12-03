from data.data_utils import MyDataLoader
import pdb
import numpy as np
import pandas as pd
import nltk
# from data.politeness.helper_files_for_original_grouping.politeness_strategies_extractor import PolitenessStrategiesExtractor

# # Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')


class PolitenessDataset(MyDataLoader):

    def split_by_politeness_feature(self, df, feature_col_name):
        """Split by presence/absence of a specific politeness feature"""

        group1 = df[df[feature_col_name] == 1]  # Has feature
        group2 = df[df[feature_col_name] == 0]  # Doesn't have feature

        group1_name = f'has_{feature_col_name}'
        group2_name = f'no_{feature_col_name}'

        return group1.copy(), group2.copy(), group1_name, group2_name

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'A': 'Polite',
            'B': 'Impolite',
        }
        return {
            'default output mapping': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': dict_with_mapping_options}
            },
        }

    def load_human_annotations_for_alt_test(self, dataset):
        
        #   A dictionary of dictionaries where:
        #   - Outer keys represent annotators (annotator ids).
        #   - Inner dictionaries with keys representing instances (instance ids) and values representing annotations.
        #   **Example:**
        #   ```python
        #   {
        #       'annotator1': {'instance1': 'A', 'instance2': 'B'},
        #       'annotator2': {'instance1': 'A', 'instance2': 'C', 'instance3': 'A'}
        #   }
        #   ```

        humans_annotations = {}
        
        # Map binarized scores to labels (lowercase as per requirement)
        score_to_label = {
            1: 'polite',
            -1: 'impolite'
        }
        
        # Iterate through each row in the dataset
        for _, row in dataset.iterrows():
            instance_id = row['id']
            
            # For each of the 5 annotators per item
            for i in range(1, 6):
                turk_id = row[f'TurkId{i}']
                score = row[f'Score{i}_binarized']
                
                # Skip neutral annotations (score = 0)
                if score == 0:
                    continue
                
                # Initialize annotator dictionary if not exists
                if turk_id not in humans_annotations:
                    humans_annotations[turk_id] = {}
                
                # Add the annotation
                humans_annotations[turk_id][instance_id] = score_to_label[score]
        
        return humans_annotations

    def get_groups_super(self, df):

        temporal_split_groups = self.add_temporal_split_groupings()

        # Add the 20 politeness feature groupings
        feature_groups = []
        for col_name in df.columns:
            if col_name.startswith('Feature_'):
                feature_groups.append((col_name, self.split_by_politeness_feature, {
                                      'feature_col_name': col_name}))

        # Combine all groupings
        return temporal_split_groups + feature_groups

    def get_super_prompts(self):
        return [
            {
                'description': '[original] [zero-shot-short] [gligoric]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # see Table 2 in https://aclanthology.org/2025.naacl-long.179.pdf
                        "content": """Is the following text polite? Output either A or B. Output a letter only.
A) Polite
B) Impolite
Text: {text}
Answer:"""
                    }
                ],
            },
            {
                'description': '[original] [zero-shot-short] [ziems]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # taken from Ziems et al.
                        "content": """{text}

Based on formal workplace social norms, which of the following best describes the above conversation? 
A: Polite
B: Impolite

Constraint: Answer with only the option above that is most accurate and nothing else."""
                    }
                ],
            },
        ]


class PolitenessWikiDataset(PolitenessDataset):
    def __init__(self, data_directory='politeness/politeness_wiki', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)
        # self.strategies_extractor = PolitenessStrategiesExtractor()

    def get_dataset_specific_groups(self, df):

        def split_by_admin_status(df):
            """Split by whether the user is a Wikipedia administrator"""
            group1 = df[df['is_admin'] == True]
            group2 = df[df['is_admin'] == False]
            group1_name = 'admins'
            group2_name = 'non_admins'
            return group1.copy(), group2.copy(), group1_name, group2_name

        # Get the parent class groupings
        super_groups = self.get_groups_super(df)

        # Add Wikipedia-specific groupings
        wiki_specific_groups = [
            ('admin_status', split_by_admin_status, {}),
        ]

        # Combine parent and specific groupings
        return super_groups + wiki_specific_groups

    def get_cols_to_stratify(self):
        feature_cols = []
        for col_name in self.load_dataset(processed_data_dir='all_data_processed_full').columns:
            if col_name.startswith('Feature_'):
                feature_cols.append(col_name)

        return [
            'ground_truth',
            'is_admin',
        ] + feature_cols

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following text is {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Text: {text}

Probability:'''

    def get_prompts(self):
        return self.get_super_prompts() + [
            {
                'description': '[paraphrase-nr-0] orig:[orig*] [zero-shot-short] [gligoric]',
                'original_description': '[original] [zero-shot-short] [gligoric]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Determine if the text below is polite. Reply with a single letter only.\nA: Polite\nB: Impolite\n\nText: {text}\n\nRespond with A or B:'
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
                        "content": '{text}\n\nAccording to standard professional etiquette, select the option that best categorizes the conversation:\nA: Polite\nB: Impolite\n\nReply with only the corresponding letter (A or B) and no additional text.'
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
                        "content": 'Decide if the text below is polite or impolite. Respond with a single letter (A or B) only.\nA) Polite\nB) Impolite\nText: {text}\nAnswer:'
                    }
                ],
            },
        ]

    def get_reported_results(self):
        return []



class PolitenessStackDataset(PolitenessDataset):
    def __init__(self, data_directory='politeness/politeness_stack', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)
        # self.strategies_extractor = PolitenessStrategiesExtractor()

    def get_dataset_specific_groups(self, df):

        def split_by_stack_overflow_vs_others(df):
            """Split by Stack Overflow community vs all other communities"""
            group1 = df[df['Community'] == '092011 Stack Overflow']
            group2 = df[df['Community'] != '092011 Stack Overflow']

            group1_name = 'stack_overflow'
            group2_name = 'other_communities'

            return group1.copy(), group2.copy(), group1_name, group2_name

        def split_by_reputation_quartiles(df, split_type='q1_vs_rest'):
            """Split by reputation quartiles using specified split type"""
            df_with_rep = df.dropna(subset=['Reputation']).copy()

            if len(df_with_rep) == 0:
                return df_with_rep, df_with_rep, 'q1', 'rest'

            # Calculate quartile cutoff points
            q25 = df_with_rep['Reputation'].quantile(0.25)
            q50 = df_with_rep['Reputation'].quantile(0.50)
            q75 = df_with_rep['Reputation'].quantile(0.75)

            if split_type == 'q1_vs_rest':
                # Q1 vs Q2+Q3+Q4
                group1 = df_with_rep[df_with_rep['Reputation'] <= q25]  # Q1
                # Q2+Q3+Q4
                group2 = df_with_rep[df_with_rep['Reputation'] > q25]
                group1_name = 'rep_q1'
                group2_name = 'rep_q2_q3_q4'
            elif split_type == 'q1q2_vs_rest':
                # Q1+Q2 vs Q3+Q4
                group1 = df_with_rep[df_with_rep['Reputation'] <= q50]  # Q1+Q2
                group2 = df_with_rep[df_with_rep['Reputation'] > q50]   # Q3+Q4
                group1_name = 'rep_q1_q2'
                group2_name = 'rep_q3_q4'
            elif split_type == 'q1q2q3_vs_q4':
                # Q1+Q2+Q3 vs Q4
                # Q1+Q2+Q3
                group1 = df_with_rep[df_with_rep['Reputation'] <= q75]
                group2 = df_with_rep[df_with_rep['Reputation'] > q75]   # Q4
                group1_name = 'rep_q1_q2_q3'
                group2_name = 'rep_q4'
            else:
                raise ValueError(f"Unknown split_type: {split_type}")

            return group1.copy(), group2.copy(), group1_name, group2_name

        def split_by_user_activity_level(df):
            """Split by user activity level (based on upvotes + downvotes)"""
            df_with_votes = df.dropna(subset=['Upvotes', 'Downvotes']).copy()

            if len(df_with_votes) == 0:
                return df_with_votes, df_with_votes, 'low_activity', 'high_activity'

            df_with_votes['total_votes'] = df_with_votes['Upvotes'] + \
                df_with_votes['Downvotes']

            # Split by total votes median
            median_votes = df_with_votes['total_votes'].median()
            group1 = df_with_votes[df_with_votes['total_votes']
                                   <= median_votes]
            group2 = df_with_votes[df_with_votes['total_votes'] > median_votes]

            group1_name = 'low_activity'
            group2_name = 'high_activity'

            return group1.copy(), group2.copy(), group1_name, group2_name

        def split_by_upvotes_level(df):
            """Split by upvotes level (high vs low upvotes)"""
            df_with_upvotes = df.dropna(subset=['Upvotes']).copy()

            if len(df_with_upvotes) == 0:
                return df_with_upvotes, df_with_upvotes, 'low_upvotes', 'high_upvotes'

            # Split by upvotes median
            median_upvotes = df_with_upvotes['Upvotes'].median()
            group1 = df_with_upvotes[df_with_upvotes['Upvotes']
                                     <= median_upvotes]
            group2 = df_with_upvotes[df_with_upvotes['Upvotes']
                                     > median_upvotes]

            group1_name = 'low_upvotes'
            group2_name = 'high_upvotes'

            return group1.copy(), group2.copy(), group1_name, group2_name

        # Get the parent class groupings
        super_groups = self.get_groups_super(df)

        # Add StackExchange-specific groupings that work with available columns
        stack_specific_groups = [
            ('community_type', split_by_stack_overflow_vs_others, {}),
            ('user_reputation_q1_vs_rest', split_by_reputation_quartiles,
             {'split_type': 'q1_vs_rest'}),
            ('user_reputation_q1q2_vs_rest', split_by_reputation_quartiles,
             {'split_type': 'q1q2_vs_rest'}),
            ('user_reputation_q1q2q3_vs_q4', split_by_reputation_quartiles,
             {'split_type': 'q1q2q3_vs_q4'}),
            ('user_activity', split_by_user_activity_level, {}),
            ('upvotes_level', split_by_upvotes_level, {}),
        ]

        # Combine parent and specific groupings
        return super_groups + stack_specific_groups

    def get_cols_to_stratify(self):
        # Add the 20 politeness feature groupings
        feature_cols = []
        for col_name in self.load_dataset(processed_data_dir='all_data_processed_full').columns:
            if col_name.startswith('Feature_'):
                feature_cols.append(col_name)

        return [
            'ground_truth',
            'Community',
        ] + feature_cols


    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following text is {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Text: {text}

Probability:'''


    def get_prompts(self):
        return self.get_super_prompts() + [
            {
                'description': '[paraphrase-nr-0] orig:[orig*] [zero-shot-short] [gligoric]',
                'original_description': '[original] [zero-shot-short] [gligoric]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Assess whether the following text is courteous. Reply with a single letter—A or B—and nothing else.\nA) Polite\nB) Impolite\nText: {text}\nAnswer:'
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
                        "content": """{text}

According to formal workplace etiquette, select the option that best characterizes the conversation above:
A: Polite
B: Impolite

Respond with only the single letter (A or B) that most accurately applies."""
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
                        "content": """Please determine whether the text below is polite. Respond with a single letter only: A or B.  
A) Polite  
B) Impolite  

Text: {text}  
Answer:"""
                    }
                ],
            },
        ]

    def get_reported_results(self):
        return []
