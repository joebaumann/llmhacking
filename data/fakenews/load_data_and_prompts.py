from data.data_utils import MyDataLoader
import pdb
import numpy as np
import pandas as pd


class FakeNewsDataset(MyDataLoader):
    def __init__(self, data_directory='fakenews', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            'Real': 'Real',
            'Fake': 'Fake',
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

        def split_by_twitter_engagement_quartiles(df, split_type='q1_vs_rest'):
            """Split by nr_of_tweets_sharing_the_news quartiles using specified split type"""
            df_with_engagement = df.dropna(subset=['nr_of_tweets_sharing_the_news']).copy()

            if len(df_with_engagement) == 0:
                return df_with_engagement, df_with_engagement, 'q1', 'rest'

            # Calculate quartile cutoff points
            q25 = df_with_engagement['nr_of_tweets_sharing_the_news'].quantile(0.25)
            q50 = df_with_engagement['nr_of_tweets_sharing_the_news'].quantile(0.50)
            q75 = df_with_engagement['nr_of_tweets_sharing_the_news'].quantile(0.75)

            if split_type == 'q1_vs_rest':
                # Q1 vs Q2+Q3+Q4
                group1 = df_with_engagement[df_with_engagement['nr_of_tweets_sharing_the_news'] <= q25]  # Q1
                # Q2+Q3+Q4
                group2 = df_with_engagement[df_with_engagement['nr_of_tweets_sharing_the_news'] > q25]
                group1_name = 'rep_q1'
                group2_name = 'rep_q2_q3_q4'
            elif split_type == 'q1q2_vs_rest':
                # Q1+Q2 vs Q3+Q4
                group1 = df_with_engagement[df_with_engagement['nr_of_tweets_sharing_the_news'] <= q50]  # Q1+Q2
                group2 = df_with_engagement[df_with_engagement['nr_of_tweets_sharing_the_news'] > q50]   # Q3+Q4
                group1_name = 'rep_q1_q2'
                group2_name = 'rep_q3_q4'
            elif split_type == 'q1q2q3_vs_q4':
                # Q1+Q2+Q3 vs Q4
                # Q1+Q2+Q3
                group1 = df_with_engagement[df_with_engagement['nr_of_tweets_sharing_the_news'] <= q75]
                group2 = df_with_engagement[df_with_engagement['nr_of_tweets_sharing_the_news'] > q75]   # Q4
                group1_name = 'rep_q1_q2_q3'
                group2_name = 'rep_q4'
            else:
                raise ValueError(f"Unknown split_type: {split_type}")

            return group1.copy(), group2.copy(), group1_name, group2_name


        source_groupings = self.split_by_column_with_n_most_frequent_values(df, 'source')
        domain_groupings = self.split_by_column_with_n_most_frequent_values(df, 'domain')

        # Add groupings based on the number of nr of tweets sharing the news
        twitter_engagement_groupings = [
            ('twitter_engagement_q1_vs_rest', split_by_twitter_engagement_quartiles,{'split_type': 'q1_vs_rest'}),
            ('twitter_engagement_q1q2_vs_rest', split_by_twitter_engagement_quartiles,{'split_type': 'q1q2_vs_rest'}),
            ('twitter_engagement_q1q2q3_vs_q4', split_by_twitter_engagement_quartiles,{'split_type': 'q1q2q3_vs_q4'}),
        ]

        # Combine parent and specific groupings
        return source_groupings + domain_groupings + twitter_engagement_groupings

    def get_cols_to_stratify(self):
        return [
            'ground_truth',
            ("source", 10),
            ("domain", 10),
        ]


    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the information in the following text is {previous_answer_placeholder}?

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
                        # see Appendix B.1 in https://dl.acm.org/doi/pdf/10.1145/3637528.3671977
                        "content": """Question: Does the following contain real or fake news (or information)? Answer in one word with either Real or Fake. {text}
Answer: """
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
                        "content": 'Task: Determine whether the information in the text below is authentic or fabricated. Reply with exactly one word—Real or Fake. {text}\nAnswer: '
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[orig*] [zero-shot-short]',
                'original_description': '[original] [zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Task: Examine the text below and decide whether it is Real or Fake news. Respond with exactly one word—Real or Fake—without any additional text. {text}\nAnswer: '
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
                        "content": 'Evaluate whether the passage below conveys real or fake news. Respond with exactly one word—Real or Fake.\n{text}\nAnswer: '
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-3] orig:[orig*] [zero-shot-short]',
                'original_description': '[original] [zero-shot-short]',
                'compatible_output_mapping': ['default output mapping'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": 'Determine whether the following content is genuine news or fabricated information. Reply with a single word—Real or Fake. {text}\nResponse: '
                    }
                ],
            },
        ]
