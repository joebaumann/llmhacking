from utils import *
import pdb
import copy
import pandas as pd
import numpy as np
import krippendorff
from data.data_utils import MyDataLoader



class Gilardi2023(MyDataLoader):
    """Base class for Gilardi et al. (2023) tasks containing shared prompts."""

    def __init__(self, data_directory, config_fn):

        # Define common prompt details that are shared between different tasks

        # define prompt details
        # prompt copied from Gilardi code: data/gilardi_et_al_pnas/data_raw/dataverse_repo/src/03-01-chatgpt-Zeroshot-Task-template.py
        # available for download at: https://doi.org/10.7910/DVN/PQYF6M
        self.gilardi_content_moderation_prompt_code = "In this job, you will be shown a sample of Tweets collected from the social media platform Twitter. Your task will be to determine if the Tweets have to do with “content” moderation” or not. \n “Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines. \n Every time someone posts something on a platform like Facebook or Twitter, that piece of content goes through a review process (‘content moderation’) to ensure that it is not illegal, hateful or inappropriate and that it complies with the rules of the site. When that is not the case, that piece of content can be removed, flagged, labelled as or ‘disputed’. \n Deciding what should be allowed on social media is not always easy. For example, many sites ban child pornography and terrorist content as it is illegal. However, things are less clear when it comes to content about the safety of vaccines or politics, for example. Even when people agree that some content should be blocked, they do not always agree about the best way to do so, about how effective it is and who should do it (the government or private companies, human moderators or artificial intelligence)."

        # prompt copied from paper Appendix A: https://www.pnas.org/lookup/suppl/doi:10.1073/pnas.2305016120/-/DCSupplemental
        self.gilardi_content_moderation_prompt_paper = "For this task, you will be asked to annotate a sample of tweets about content moderation. Before describing the task, we explain what we mean by “content moderation”.\n“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines. Every time someone posts something on a platform like Facebook or Twitter, that piece of content goes through a review process (“content moderation”) to ensure that it is not illegal, hateful or inappropriate and that it complies with the rules of the site. When that is not the case, that piece of content can be removed, flagged, labeled as or ‘disputed.’\nDeciding what should be allowed on social media is not always easy. For example, many sites ban child pornography and terrorist content as it is illegal. However, things are less clear when it comes to content about the safety of vaccines or politics, for example. Even when people agree that some content should be blocked, they do not always agree about the best way to do so, how effective it is, and who should do it (the government or private companies, human moderators, or artificial intelligence)."

        # prompt copied from paper Appendix B: https://www.pnas.org/lookup/suppl/doi:10.1073/pnas.2305016120/-/DCSupplemental
        self.gilardi_political_content_prompt_paper = "For this task, you will be asked to annotate a sample of tweets to determine if they include political content or not. For the purposes of this task, tweets are “relevant” if they include political content, and “irrelevant” if they do not. Before describing the task, we explain what we mean by “political content”. “Political content” refers to any tweets that pertain to politics or government policies at the local, national, or international level. This can include tweets that discuss political figures, events, or issues, as well as tweets that use political language or hashtags. To determine if tweets include political content or not, consider several factors, such as the use of political keywords or hashtags, the mention of political figures or events, the inclusion of links to news articles or other political sources, and the overall tone and sentiment of the tweet, which may indicate whether it is conveying a political message or viewpoint."
    

        # define prompt details

        # prompt copied from Gilardi code: data/gilardi_et_al_pnas/data_raw/dataverse_repo/src/03-01-chatgpt-Zeroshot-Task-template.py
        # available for download at: https://doi.org/10.7910/DVN/PQYF6M
        self.content_moderation_relevance_specific_prompt_code = self.gilardi_content_moderation_prompt_code + \
            "\n For each tweet in the sample: Carefully read the text of the Tweet, paying close attention to details. Classify the Tweet as either irrelevant (0) or relevant (1). \n Tweets should be coded as relevant when they directly relate to content moderation. This includes Tweets that discuss social media platforms’ content moderation rules and practices, and Tweets that discuss governments’ regulation of online content moderation. This also includes Tweets that discuss mild forms of content moderation, like flagging Tweets and Tweets when they indirectly relate to content moderation.\n Tweets should be coded as irrelevant if they do not refer to content moderation or if they are themselves examples of moderated content. This would include, for example, a Tweet by Donald Trump that Twitter has labelled as ‘disputed’, a Tweet claiming that something is false, or a Tweet containing sensitive content. just label 'relevant' or 'irrelevant' without any more explanation "

        # 2nd part of prompt copied from paper Appendix C: https://www.pnas.org/lookup/suppl/doi:10.1073/pnas.2305016120/-/DCSupplemental
        self.content_moderation_relevance_specific_prompt_paper = self.gilardi_content_moderation_prompt_paper + \
            "\nFor each tweet in the sample, follow these instructions:\n1. Carefully read the text of the tweet, paying close attention to details.\n2. Classify the tweet as either relevant (1) or irrelevant (0)\nTweets should be coded as RELEVANT when they directly relate to content moderation, as defined above. This includes tweets that discuss: social media platforms’ content moderation rules and practices, governments’ regulation of online content moderation, and/or mild forms of content moderation like flagging. Tweets should be coded as IRRELEVANT if they do not refer to content moderation, as defined above, or if they are themselves examples of moderated content. This would include, for example, a Tweet by Donald Trump that Twitter has labeled as “disputed”, a tweet claiming that something is false, or a tweet containing sensitive content. Such tweets might be subject to content moderation, but are not discussing content moderation. Therefore, they should be coded as irrelevant for our purposes."


        # now define all output mappings

        self.relevance_output_mapping = {
            'relevant': 'relevant',
            'irrelevant': 'irrelevant',
        }
        self.relevance_output_mapping_extended_0_and_1 = {
            '1': 'relevant',
            '0': 'irrelevant',
        }
        self.relevance_output_mapping_extended_A_and_B = {
            'A': 'relevant',
            'B': 'irrelevant',
        }
        self.frame_output_mapping = {
            'problem': 'problem',
            'solution': 'solution',
            'neutral': 'neutral',
        }
        self.frame_output_mapping_extended_A_and_B_and_C = {
            'A': 'problem',
            'B': 'solution',
            'C': 'neutral',
        }
        self.frame_political_output_mapping = {
            'economy': 'economy',
            'morality': 'morality',
            'fairness': 'fairness',
            'policy': 'policy',
            'law': 'law',
            'security': 'security',
            'health': 'health',
            'quality': 'quality',
            'political': 'political',
            'external': 'external',
            'other': 'other',
        }
        self.frame_political_output_mapping_extended_A_to_K = {
            "a": "economy",
            "b": "morality",
            "c": "fairness",
            "d": "policy",
            "e": "law",
            "f": "security",
            "g": "health",
            "h": "quality",
            "i": "political",
            "j": "external",
            "k": "other"
        }
        self.stance_2023_output_mapping = {
            'positive': 'in favor of',
            'negative': 'against',
            'neutral': 'neutral',
        }
        self.stance_2024_classes_output_mapping = {
            'in favor of': 'in favor of',
            'against': 'against',
            'neutral': 'neutral',
        }
        self.stance_2024_classes_output_mapping_extended_A_and_B_and_C = {
            'A': 'in favor of',
            'B': 'against',
            'C': 'neutral',
        }
        self.topic_political_output_mapping = {
            "section 230": "section 230",
            "trump ban": "trump ban",
            "twitter support": "twitter support",
            "platform policies": "platform policies",
            "complaint": "complaint",
            "other": "other",
        }
        self.topic_political_output_mapping_extended_A_to_F = {
            "a": "section 230",
            "b": "trump ban",
            "c": "twitter support",
            "d": "platform policies",
            "e": "complaint",
            "f": "other",
        }

        super().__init__(data_directory, config_fn)


    def get_reported_results(self):
        return []


    def order_df_columns(self, df):
        # sort df columns to start with 'ground_truth', then 'text', and then all other columns
        priority_columns = ['ground_truth', 'text']
        other_columns = [col for col in df.columns if col not in priority_columns]
        df = df[priority_columns + other_columns]
        return df

    def get_dataset_specific_groups(self, df):
        return []

    def get_cols_to_stratify(self):
        return [
            'ground_truth',
        ]


    def get_frame_gt(self, row, annotator_name):
        """Determine the ground truth frame label based on annotator values."""
        if row[f'problem_frame_{annotator_name}'] == 1 and row[f'solution_frame_{annotator_name}'] == 0:
            return 'problem'
        elif row[f'problem_frame_{annotator_name}'] == 0 and row[f'solution_frame_{annotator_name}'] == 1:
            return 'solution'
        elif row[f'problem_frame_{annotator_name}'] == 0 and row[f'solution_frame_{annotator_name}'] == 0:
            return 'neutral'
        else:
            # throw an error if the row has a problem_frame and solution_frame both set to 1
            raise ValueError(
                f"Row has both problem_frame and solution_frame set to 1 for annotator {annotator_name}")


    def get_stance_gt(self, row, annotator_name):
        """Determine the ground truth frame label based on annotator values."""
        if row[f'{annotator_name}_stance_pro'] == 1 and row[f'{annotator_name}_stance_neutral'] == 0 and row[f'{annotator_name}_stance_contra'] == 0:
            return 'in favor of'
        if row[f'{annotator_name}_stance_pro'] == 0 and row[f'{annotator_name}_stance_neutral'] == 0 and row[f'{annotator_name}_stance_contra'] == 1:
            return 'against'
        if row[f'{annotator_name}_stance_pro'] == 0 and row[f'{annotator_name}_stance_neutral'] == 1 and row[f'{annotator_name}_stance_contra'] == 0:
            return 'neutral'
        else:
            # throw an error if the row has a problem_frame and solution_frame both set to 1
            raise ValueError(
                f"Row has both problem_frame and solution_frame set to 1 for annotator {annotator_name}")


    def get_max_occuring_class_in_output(self, text, classes, all_found_labels, default_label):
        # count the number of times each class occurs in the text
        class_counts = {c: text.count(c) for c in all_found_labels}
        # sort the classes by count
        sorted_classes = sorted(
            class_counts, key=lambda x: class_counts[x], reverse=True)
        # check if the two most common classes have the same count
        if len(sorted_classes) > 1 and class_counts[sorted_classes[0]] == class_counts[sorted_classes[1]]:
            # if so, return neutral
            label = default_label
        else:
            label = sorted_classes[0]
        return label


    def get_detailed_frames_ground_truth(self, datapoint, gt=None):
        """Extract the ground truth from a datapoint."""
        if gt is None:
            gt = datapoint['ground_truth'].lower().strip()
        else:
            gt = gt.lower().strip()
        if gt == 'external regulation and reputation':
            gt = 'external'
        elif gt == 'security and defense frames':
            gt = 'security'
        elif gt == 'policy prescription and evaluation':
            gt = 'policy'
        elif gt == 'constitutionality and jurisprudency':
            gt = 'constitutionality'
        elif gt == 'fairness and equality':
            gt = 'fairness'
        elif gt == 'law and order, crime and justice frames':
            gt = 'law'
        elif gt == 'health and safety':
            gt = 'health'
        elif gt == 'security and defense':
            gt = 'security'
        elif gt == 'law and order, crime and justice':
            gt = 'law'
        elif gt == 'constitutionality and jurisprudence':
            gt = 'constitutionality'
        elif gt == 'quality of life':
            gt = 'quality'
        elif gt == 'capacity and resources':
            gt = 'capacity'
        elif gt == 'cultural identity':
            gt = 'cultural'
        elif gt == 'public opinion':
            gt = 'public'
        return gt


    # Helper function for intercoder agreement calculation
    def intercoder_agreement_percentage(self, df, columns):
        """
        Calculates the percentage of instances where all annotators report the same class.
        """
        # Drop rows with NaN values in the selected columns
        filtered_df = df[columns].dropna()
        # Count instances where all annotators agree (i.e., all values in a row are the same)
        agreement_count = (filtered_df.nunique(axis=1) == 1).sum()
        # Calculate percentage agreement
        total_instances = len(filtered_df)
        agreement_percentage = (
            agreement_count / total_instances) if total_instances > 0 else 0
        return agreement_percentage

    def intercoder_agreement_krippendorff(self, df, columns):
        """
        Calculates the krippendorff alpha.
        """
        # Extract annotation data for specified columns
        data = df[columns].copy()
        
        # Convert categorical labels to numeric values
        # Get unique labels across all columns (excluding NaN)
        all_labels = pd.concat([data[col] for col in columns]).dropna().unique()
        label_to_num = {label: i for i, label in enumerate(all_labels)}
        
        # Apply mapping to each column
        for col in columns:
            data[col] = data[col].map(label_to_num)
        
        # Transpose so that each row represents an annotator's ratings
        reliability_data = data.T.values
        
        # Calculate Krippendorff's alpha
        krippendorff_alpha = krippendorff.alpha(
            reliability_data=reliability_data, 
            level_of_measurement='nominal'
        )
        
        return krippendorff_alpha


    def _normalize_mturk_annotation(self, x):
        x_lower = str(x).strip().lower()
        if self.data_and_task_name in ['framesII_tweets', 'framesII_tweets17']:
            x_lower = self.get_detailed_frames_ground_truth(self, gt=x_lower)
        if x_lower == 'economy':
            return 'economic'
        elif x_lower == 'irrelevnt':
            return 'irrelevant'
        elif x_lower == 'neither':
            return 'neutral'
        elif x_lower == 'complaints':
            return 'complaint'
        else:
            return x_lower

    def load_mturk_annotations(self):
        """Load mTurk annotations from batch results file and return as DataFrame."""
        
        mturk_file = self._get_mturk_file_path()
        delimiter = self._get_mturk_delimiter()
        
        try:
            df = pd.read_csv(mturk_file, delimiter=delimiter, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            df = pd.read_csv(mturk_file, delimiter=delimiter, encoding='latin-1', on_bad_lines='skip')
        
        # Get text column
        text_col = self._get_mturk_text_col()
        
        # Filter to rows with mTurk answers and text
        df = df[df['Answer.category.label'].notna() & df[text_col].notna()].copy()
        
        # Normalize mTurk annotations
        df['mturk_annotation'] = df['Answer.category.label'].apply(
            self._normalize_mturk_annotation
        )

        df['text_clean'] = df[text_col].apply(self.clean_text)

        # Group by text to get all annotations per text
        result = df.groupby('text_clean').agg({
            'mturk_annotation': lambda x: list(x),
            'WorkerId': lambda x: list(x)
        }).reset_index()
        
        
        return result

    # Default implementations - override in subclasses as needed
    def _get_mturk_file_path(self):
        raise NotImplementedError("Subclass must implement _get_mturk_file_path")

    def _get_mturk_delimiter(self):
        return ','

    def _get_mturk_encoding(self):
        return 'utf-8'

    def _get_mturk_text_col(self):
        return 'Input.text'





class Gilardi2023_Data1Task1_Tweets_2020_2021(Gilardi2023):
    """Implementation for Gilardi 2023 Relevance Task."""

    def __init__(self, data_directory='gilardi_et_al_pnas', config_fn='config_relevance_tweets.yaml'):
        super().__init__(data_directory, config_fn)


    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following tweet is {previous_answer_placeholder} to the topic of content moderation?

Output only a single number between 0 and 1, without any context or explanation.

Tweet: {text}

Probability:'''

    def get_prompts(self):
        
        # Define relevance-specific prompt details
        # This part of the prompt is specific to the relevance task

        # the following prompts are copied from Gilardi 2024 code: data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv
        # available for download at: https://osf.io/adkun/files/osfstorage
        # a=pd.read_csv("data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv")

        relevance_specific_prompt_code_2024_zero = "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be banned or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Tweet is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like banning, flagging, shadowbanning, or account suspension.\n\nB: Tweet is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nNow, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
        ban_prompt_rel1 = "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be banning or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Tweet is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like banning, flagging, shadowbanning, or account suspension.\n\nB: Tweet is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nNotice that whenever a tweet mentions 'ban' or 'banning', it is typically RELEVANT to content moderation.\n\nNow, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
        ban_prompt_rel2 = "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Tweet is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like flagging, shadowbanning, or account suspension.\n\nB: Tweet is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nNow, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
        ban_prompt_irrel1 = "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Tweet is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like flagging, shadowbanning, or account suspension.\n\nB: Tweet is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nNotice that whenever a tweet mentions 'ban' or 'banning', it is typically IRRELEVANT to content moderation.\n\nNow, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
        ban_prompt_irrel2 = "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Tweet is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like flagging, or account suspension.\n\nB: Tweet is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nNotice that whenever a tweet mentions 'ban' or 'banning', it is typically IRRELEVANT to content moderation.\n\nNow, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"

        relevance_specific_prompt_code_2024_few = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Tweet is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like flagging, shadowbanning, or account suspension.\n\nB: Tweet is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nThe tweets below are some examples of 'RELEVANT' and 'IRRELEVANT' texts to content moderation in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "Please stop lying. We all know you can’t stand Sanders and the platform he stands for and people that support progressive policies.cBut can you at least not LIE? It’s a really bad look. Thanks."
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Hey guys, my TL is hella cluttered from all the follow backs I did this season. I’m going to reinstate my “if you only rt content” I’m not going to follow your policy. No hard feelings and I hope you stick around, but if I unfollow you that’s likely why. ✌🏼"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "New from Pediatrics: social media influencers generate millions of impressions for unhealthy food and drink brands through product placement. The authors call on the FTC to strengthen regulations on YouTube product placement featuring young children. https://t.co/Yicupispun"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "This is a tweet from today of the sitting President of the United States accusing a respected journalist and outspoken critic of his of MURDER with NO EVIDENCE whatsoever. Zero. Twitter should immediately remove this tweet and suspend this account. This is INSANE. https://t.co/chlp7iKBXL"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Now, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
            }
        ]


        relevance_specific_prompt_code_2024_cot = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Tweet is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like flagging, shadowbanning, or account suspension.\n\nB: Tweet is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nThe tweets below are some examples of 'RELEVANT' or 'IRRELEVANT' texts to content moderation in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "Please stop lying. We all know you can’t stand Sanders and the platform he stands for and people that support progressive policies.cBut can you at least not LIE? It’s a really bad look. Thanks."
            },
            {
                "role": "assistant",
                "content": "Based on the given criteria, the text provided is IRRELEVANT to content moderation. It doesn't discuss social media platforms' content moderation rules and practices, censorship, governments' regulation of online content moderation, or any forms of content moderation like flagging, shadowbanning, or account suspension. Instead, it appears to be a message directed at two individuals and their opinions about Sanders and progressive policies."
            },
            {
                "role": "user",
                "content": "Hey guys, my TL is hella cluttered from all the follow backs I did this season. I’m going to reinstate my “if you only rt content” I’m not going to follow your policy. No hard feelings and I hope you stick around, but if I unfollow you that’s likely why. ✌🏼"
            },
            {
                "role": "assistant",
                "content": "Based on the given criteria, the text provided is IRRELEVANT to content moderation. It doesn't discuss social media platforms' content moderation rules and practices, censorship, governments' regulation of online content moderation, or any mild forms of content moderation like flagging, shadowbanning, or account suspension. Instead, it appears to be a personal statement about cluttered timelines and a follow-back policy on Twitter."
            },
            {
                "role": "user",
                "content": "New from Pediatrics: social media influencers generate millions of impressions for unhealthy food and drink brands through product placement. The authors call on the FTC to strengthen regulations on YouTube product placement featuring young children. https://t.co/Yicupispun"
            },
            {
                "role": "assistant",
                "content": "Based on the given criteria, the text provided is RELEVANT to content moderation. It discusses the impact of social media influencers and product placement on unhealthy food and drink brands. It also mentions calling on the FTC (Federal Trade Commission) to strengthen regulations on YouTube product placement featuring young children. This relates to content moderation as it involves discussing regulations and practices related to content promotion and advertising on social media platforms."
            },
            {
                "role": "user",
                "content": "This is a tweet from today of the sitting President of the United States accusing a respected journalist and outspoken critic of his of MURDER with NO EVIDENCE whatsoever. Zero. Twitter should immediately remove this tweet and suspend this account. This is INSANE. https://t.co/chlp7iKBXL"
            },
            {
                "role": "assistant",
                "content": "Based on the given criteria, the text provided is RELEVANT to content moderation. It discusses the actions and potential response of Twitter in relation to a tweet made by the sitting President of the United States. It addresses the need for Twitter to remove the tweet and suspend the account due to the serious accusation made without evidence. This involves discussing content moderation rules and practices, specifically regarding the removal of content and potential account suspension."
            },
            {
                "role": "user",
                "content": "Now, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
            }
        ]

        # relevance_specific_promot_paper_2024 = ""

        return [
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (code) tweets_2020_2021 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance',
                        'gilardi relevance (extended with 0 and 1)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": self.content_moderation_relevance_specific_prompt_code
                        },
                        {
                            "role": "user",
                            "content": " here's the tweet I picked, please label it as 'Relevant' or 'Irrelevant' by answering with one word:\n{text}"
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (code) tweets_2020_2021 relevance (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi relevance',
                        'gilardi relevance (extended with 0 and 1)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": self.content_moderation_relevance_specific_prompt_code + " here's the tweet I picked, please label it as 'Relevant' or 'Irrelevant' by answering with one word:\n{text}"
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets_2020_2021 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance',
                        'gilardi relevance (extended with 0 and 1)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": self.content_moderation_relevance_specific_prompt_paper
                        },
                        {
                            "role": "user",
                            "content": "Here's the tweet I picked, please label it as 'Relevant' or 'Irrelevant' by answering with one word:\n{text}"
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets_2020_2021 relevance (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with 0 and 1)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": self.content_moderation_relevance_specific_prompt_paper + " Here's the tweet I picked, please label it as 'Relevant' or 'Irrelevant' by answering with one word:\n{text}"
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets_2020_2021 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with A and B)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": relevance_specific_prompt_code_2024_zero[0]
                        },
                        {
                            "role": "user",
                            "content": relevance_specific_prompt_code_2024_zero[1]
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets_2020_2021 relevance (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi relevance',
                        'gilardi relevance (extended with A and B)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": relevance_specific_prompt_code_2024_zero[0] + "\n\n" + relevance_specific_prompt_code_2024_zero[1]
                        },
                    ],
                },
                {
                    'description': '[original] [few-shot-detailed] gilardi 2024 (code) tweets_2020_2021 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance',
                        'gilardi relevance (extended with A and B)'],
                    'prompt_text': relevance_specific_prompt_code_2024_few,
                },
            ]

    def get_all_output_mappings(self):

        return {
            'gilardi relevance': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.relevance_output_mapping},
            },
            'gilardi relevance (extended with 0 and 1)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.relevance_output_mapping_extended_0_and_1},
            },
            'gilardi relevance (extended with A and B)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.relevance_output_mapping_extended_A_and_B},
            },
        }


    def get_dataset_name(self):
        """Get the name of the dataset."""
        return "Gilardi2023_Data1Task1_Tweets_2020_2021"
    
    def load_full_dataset(self, return_statistics_for_plotting=False, return_dataset_used_by_gilardi=False):
        df_raw = pd.read_excel("data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/research_assistants_data/relevance_task/annotation_data_ra_completed.xlsx")


        # drop rows with NaN values in relevant columns
        df = df_raw.dropna(subset=['relevant_fabio', 'relevant_paula'], how='all')
        # Process for relevance task
        df_relevance = df.dropna(subset=['relevant_fabio', 'relevant_paula'])

        # Now drop duplicates
        df_relevance_without_duplicates = df_relevance.drop_duplicates(subset=['status_id'], keep='first')
        nr_of_duplicates_based_on_id = len(df_relevance) - len(df_relevance_without_duplicates)
        print(f"\nRows after dropping duplicates (based on id): {len(df_relevance_without_duplicates)}")
        if not return_dataset_used_by_gilardi:
            df_relevance_without_duplicates = df_relevance_without_duplicates.drop_duplicates(subset=['text'], keep='first')
        print(f"\nRows after dropping duplicates: {len(df_relevance_without_duplicates)}")

        # calculate trained annotators agreement
        trained_annotators_agreement = self.intercoder_agreement_percentage(df_relevance_without_duplicates, ['relevant_fabio', 'relevant_paula'])
        trained_annotators_agreement_krippendorff = self.intercoder_agreement_krippendorff(df_relevance_without_duplicates, ['relevant_fabio', 'relevant_paula'])
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement: {trained_annotators_agreement:.4f}')
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement_krippendorff: {trained_annotators_agreement_krippendorff:.4f}')

        df_relevance_with_gt = df_relevance_without_duplicates[
            df_relevance_without_duplicates['relevant_fabio'] == df_relevance_without_duplicates['relevant_paula']
        ]
        df_relevance_with_gt['ground_truth'] = copy.deepcopy(
            df_relevance_with_gt['relevant_fabio']
        )
        

        nr_of_rows_with_missing_annotations = len(df) - len(df_relevance)
        nr_of_duplicates = len(df_relevance) - len(df_relevance_without_duplicates)
        nr_of_rows_with_disagreement = len(df_relevance_without_duplicates) - len(df_relevance_with_gt)

        dataset_statistics = {
            'Dataset name': self.get_dataset_name(),
            'Total rows (raw)': len(df_raw),
            'Total rows': len(df),
            'Rows without GT NaN values': len(df_relevance),
            'Rows without duplicate values': len(df_relevance_without_duplicates),
            'Trained annotators agreement': trained_annotators_agreement,
            'Trained annotators agreement (krippendorff)': trained_annotators_agreement_krippendorff,
            'Rows with missing annotations': nr_of_rows_with_missing_annotations,
            'Rows with duplicates (based on id)': nr_of_duplicates_based_on_id,
            'Rows with duplicates': nr_of_duplicates,
            'Rows with disagreement': nr_of_rows_with_disagreement,
            'Final dataset size': len(df_relevance_with_gt),
            # 'Rows which all trained annotators agree [relevant=1]': sum(df_relevance_with_gt['ground_truth']==1),
            'Value counts': df_relevance_with_gt['ground_truth'].value_counts(),
        }
        for k, v in dataset_statistics.items():
            print(f"    {k}: {v}")

        df_relevance_with_gt = self.order_df_columns(df_relevance_with_gt)
        df_relevance_with_gt['ground_truth'] = df_relevance_with_gt['ground_truth'].map({0: 'irrelevant', 1: 'relevant'})

        if return_statistics_for_plotting:
            return df_relevance_with_gt, dataset_statistics
        else:
            return df_relevance_with_gt

    def _get_mturk_file_path(self):
        return 'data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/mTurk_data/batch_results_relevance_final.csv'

class Gilardi2023_Data1Task2_Tweets_2020_2021(Gilardi2023):
    """Implementation for Gilardi 2023 Frame Task."""

    def __init__(self, data_directory='gilardi_et_al_pnas', config_fn='config_framesI_tweets.yaml'):
        super().__init__(data_directory, config_fn)

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following tweet describes content moderation as a {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Tweet: {text}

Probability:'''

    def get_prompts(self):
        # prompt copied from paper Appendix E: https://www.pnas.org/lookup/suppl/doi:10.1073/pnas.2305016120/-/DCSupplemental
        frame_problem_solution_prompt = "\nContent moderation can be seen from two different perspectives:\n• Content moderation can be seen as a PROBLEM; for example, as a restriction of free speech\n• Content moderation can be seen as a SOLUTION; for example, as a protection from harmful speech\nFor each tweet in the sample, follow these instructions:\n1. Carefully read the text of the tweet, paying close attention to details.\n2. Classify the tweet as describing content moderation as a problem, as a solution, or neither.\nTweets should be classified as describing content moderation as a PROBLEM if they emphasize negative effects of content moderation, such as restrictions to free speech, or the biases that can emerge from decisions regarding what users are allowed to post.\nTweets should be classified as describing content moderation as a SOLUTION if they emphasize positive effects of content moderation, such as protecting users from various kinds of harmful content, including hate speech, misinformation, illegal adult content, or spam.\nTweets should be classified as describing content moderation as NEUTRAL if they do not emphasize possible negative or positive effects of content moderation, for example if they simply report on the content moderation activity of social media platforms without linking them to potential advantages or disadvantages for users or stakeholders."

        gilardi_frame_prompt_code = self.gilardi_content_moderation_prompt_code + \
            frame_problem_solution_prompt
        gilardi_frame_prompt_paper = self.gilardi_content_moderation_prompt_paper + \
            frame_problem_solution_prompt

        tweet_instruction_prompt = "Here's the tweet I picked, please label it as 'Problem', 'Solution', or 'Neutral' by answering with one word:\n{text}"


        # the following prompts are copied from Gilardi 2024 code: data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv
        # available for download at: https://osf.io/adkun/files/osfstorage
        # a=pd.read_csv("data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv")
        frame_specific_prompt_code_2024_zero = ["“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as describing content moderation as a problem, as a solution, or neither:\n\nA: Tweet describes content moderation as a PROBLEM if they emphasize negative effects of it, such as restrictions to free speech, censorship, or the biases that can emerge from decisions regarding what users are allowed to post.\n\nB: Tweet describes content moderation as a SOLUTION if they emphasize positive effects of it, such as protecting users from harmful content such as hate speech, misinformation, illegal adult content, or spam. \n\nC: Tweet describes content moderation as NEUTRAL if they do not emphasize negative or positive effects of content moderation. For example if they simply report on the content moderation activity of social media platforms without linking them to potential advantages or disadvantages for users or stakeholders.", "Now, is the following tweet describing content moderation as a PROBLEM, as a SOLUTION, or NEUTRAL?\n\n{text}"]


        frame_specific_prompt_code_2024_few = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as describing content moderation as a problem, as a solution, or neither:\n\nA: Tweet describes content moderation as a PROBLEM if they emphasize negative effects of it, such as restrictions to free speech, censorship, or the biases that can emerge from decisions regarding what users are allowed to post.\n\nB: Tweet describes content moderation as a SOLUTION if they emphasize positive effects of it, such as protecting users from harmful content such as hate speech, misinformation, illegal adult content, or spam. \n\nC: Tweet describes content moderation as NEUTRAL if they do not emphasize negative or positive effects of content moderation. For example if they simply report on the content moderation activity of social media platforms without linking them to potential advantages or disadvantages for users or stakeholders.\n\nThe following tweets are some examples of 'PROBLEM', 'SOLUTION' or 'NEUTRAL' texts about content moderation in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "@amerxmirza I have reported him to make sure his account gets suspended. Disgusting and derogation remarks shouldn't be tolerated. Please report. Thanks"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Channel link  https://t.co/9ms7ukCI84 please block this channel this content very harmful https://t.co/zeJ512JPPx"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Finally could restore my account. 🤬 Why twitter even require phone number in the first place? Aren't we supposed to have a choice not to show it thus not being banned after, if twitter bots decide you're one of them? #twitter #ElonMusk #FreedomInternet"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Of all the issues we’re facing now, why’s social media regulation the goal for this people? Freedom of speech is a basic human right and any attempt to take it from Nigerians should be rejected #RejectNITDAbill #Notosocialmediaregulation"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "@BotGerad @R6Jena @zxdd account suspended xd"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "@elonmusk @ellagirwin  @esthercrawford there seems to have been some mistake, renowned internet love-monger @nickjfuentes was suspended from Twitter. Please do the right thing and re-instate his account for the sake of humanity 🙏😔"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Now, is the following tweet describing content moderation as a PROBLEM, as a SOLUTION, or NEUTRAL?\n\n{text}"
            }
        ]
        
        frame_specific_prompt_code_2024_cot = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as describing content moderation as a problem, as a solution, or neither:\n\nA: Tweet describes content moderation as a PROBLEM if they emphasize negative effects of it, such as restrictions to free speech, censorship, or the biases that can emerge from decisions regarding what users are allowed to post.\n\nB: Tweet describes content moderation as a SOLUTION if they emphasize positive effects of it, such as protecting users from harmful content such as hate speech, misinformation, illegal adult content, or spam. \n\nC: Tweet describes content moderation as NEUTRAL if they do not emphasize negative or positive effects of content moderation. For example if they simply report on the content moderation activity of social media platforms without linking them to potential advantages or disadvantages for users or stakeholders.\n\nThe following tweets are some examples of 'PROBLEM', 'SOLUTION' or 'NEUTRAL' texts about content moderation in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "@amerxmirza I have reported him to make sure his account gets suspended. Disgusting and derogation remarks shouldn't be tolerated. Please report. Thanks"
            },
            {
                "role": "assistant",
                "content": """The given text can be classified as describing content moderation as a SOLUTION. It emphasizes the positive effect of content moderation by advocating for reporting and suspending the account of a user who made "disgusting and derogatory remarks." The text acknowledges the need to not tolerate such behavior and encourages others to report it, highlighting the protective role of content moderation in addressing harmful content."""
            },
            {
                "role": "user",
                "content": "Channel link  https://t.co/9ms7ukCI84 please block this channel this content very harmful https://t.co/zeJ512JPPx"
            },
            {
                "role": "assistant",
                "content": "The given text can be classified as describing content moderation as a SOLUTION. It emphasizes the positive effect of content moderation by requesting the blocking of a specific channel due to its harmful content. The text recognizes the need to take action and protect users from potentially harmful content, aligning with the protective role of content moderation in addressing harmful material."
            },
            {
                "role": "user",
                "content": "Finally could restore my account. 🤬 Why twitter even require phone number in the first place? Aren't we supposed to have a choice not to show it thus not being banned after, if twitter bots decide you're one of them? #twitter #ElonMusk #FreedomInternet"
            },
            {
                "role": "assistant",
                "content": "The given text can be classified as describing content moderation as a PROBLEM. It criticizes the requirement of providing a phone number by Twitter and questions the lack of choice in not showing it. The text implies frustration with the potential consequences of content moderation decisions made by Twitter bots, suggesting concerns about restrictions to freedom and potential biases in the platform's moderation practices."
            },
            {
                "role": "user",
                "content": "Of all the issues we’re facing now, why’s social media regulation the goal for this people? Freedom of speech is a basic human right and any attempt to take it from Nigerians should be rejected #RejectNITDAbill #Notosocialmediaregulation"
            },
            {
                "role": "assistant",
                "content": "The given text can be classified as describing content moderation as a PROBLEM. It emphasizes the negative effects of social media regulation, specifically mentioning the potential infringement on freedom of speech as a basic human right. The text opposes any attempt to take away this right from Nigerians, suggesting concerns about restrictions and censorship associated with content moderation."
            },
            {
                "role": "user",
                "content": "@BotGerad @R6Jena @zxdd account suspended xd"
            },
            {
                "role": "assistant",
                "content": "Based on the given text, it does not explicitly emphasize the negative or positive effects of content moderation. Instead, it simply reports the suspension of an account without further elaboration. Therefore, the text can be classified as describing content moderation as NEUTRAL, as it does not emphasize either the problems or the solutions associated with content moderation."
            },
            {
                "role": "user",
                "content": "@elonmusk @ellagirwin  @esthercrawford there seems to have been some mistake, renowned internet love-monger @nickjfuentes was suspended from Twitter. Please do the right thing and re-instate his account for the sake of humanity 🙏😔"
            },
            {
                "role": "assistant",
                "content": "Based on the given text, it is possible to interpret it differently. While the text does request the reinstatement of a suspended account, it does not explicitly mention any negative effects or problems related to content moderation. Therefore, an alternative classification could be that the text describes content moderation as NEUTRAL since it does not emphasize negative or positive effects. It simply requests the reinstatement of a specific account without further elaboration on the broader implications of content moderation."
            },
            {
                "role": "user",
                "content": "Now, is the following tweet describing content moderation as a PROBLEM, as a SOLUTION, or NEUTRAL?\n\n{text}"
            }
        ]

        # frame_specific_promot_paper_2024 = ""


        return [
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (code) tweets_2020_2021 frame',
                    'compatible_output_mapping': ['gilardi frame'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": gilardi_frame_prompt_code
                        },
                        {
                            "role": "user",
                            "content": tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (code) tweets_2020_2021 frame (no system prompt)',
                    'compatible_output_mapping': ['gilardi frame'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": gilardi_frame_prompt_code + "\n" + tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets_2020_2021 frame',
                    'compatible_output_mapping': ['gilardi frame'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": gilardi_frame_prompt_paper
                        },
                        {
                            "role": "user",
                            "content": tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets_2020_2021 frame (no system prompt)',
                    'compatible_output_mapping': ['gilardi frame'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": gilardi_frame_prompt_paper + "\n" + tweet_instruction_prompt

                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets_2020_2021 frame',
                    'compatible_output_mapping': [
                        'gilardi frame (extended with A and B and C)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": frame_specific_prompt_code_2024_zero[0]
                        },
                        {
                            "role": "user",
                            "content": frame_specific_prompt_code_2024_zero[1]
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets_2020_2021 frame zero-shot (no system prompt)',
                    'compatible_output_mapping': [
                        'gilardi frame (extended with A and B and C)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": frame_specific_prompt_code_2024_zero[0] + "\n\n" + frame_specific_prompt_code_2024_zero[1]
                        },
                    ],
                },
                {
                    'description': '[original] [few-shot-detailed] gilardi 2024 (code) tweets_2020_2021 frame',
                    'compatible_output_mapping': [
                        'gilardi frame (extended with A and B and C)'],
                    'prompt_text': frame_specific_prompt_code_2024_few,
                },
            ]

    def get_all_output_mappings(self):

        return {
            'gilardi frame': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.frame_output_mapping},
            },
            'gilardi frame (extended with A and B and C)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.frame_output_mapping_extended_A_and_B_and_C},
            },
        }


    def get_dataset_name(self):
        """Get the name of the dataset."""
        return "Gilardi2023_Data1Task2_Tweets_2020_2021"

    def load_full_dataset(self, return_statistics_for_plotting=False, return_dataset_used_by_gilardi=False):
        """Load and prepare the Gilardi 2023 Frame dataset."""
        # Load data from Excel files
        df_raw = pd.read_excel("data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/research_assistants_data/problem_solution_task/annotation_data_ra_completed.xlsx")
        df = df_raw.dropna(subset=['relevant_fabio', 'relevant_paula'], how='all')
        # Filter to only relevant tweets and process for frame task
        df_relevant = df[
            (df['relevant_fabio'] == 1) &
            (df['relevant_paula'] == 1)
        ]

        # Now drop duplicates
        df_frame_without_duplicates = df_relevant.drop_duplicates(subset=['status_id'], keep='first')
        nr_of_duplicates_based_on_id = len(df_relevant) - len(df_frame_without_duplicates)
        if not return_dataset_used_by_gilardi:
            df_frame_without_duplicates = df_frame_without_duplicates.drop_duplicates(subset=['text'], keep='first')
        print(f"\nRows after dropping duplicates: {len(df_frame_without_duplicates)}")

        df_frame = df_frame_without_duplicates.dropna(
            subset=['problem_frame_fabio', 'problem_frame_paula',
                    'solution_frame_fabio', 'solution_frame_paula']
        )

        df_frame_not_both = df_frame[(
            ~(
                ((df_frame['problem_frame_fabio'] == 1) & (df_frame['solution_frame_fabio'] == 1)) |
                ((df_frame['problem_frame_paula'] == 1) & (df_frame['solution_frame_paula'] == 1))
            )
        )]

        # Apply frame classification functions
        df_frame_not_both['ground_truth_fabio'] = df_frame_not_both.apply(
            lambda row: self.get_frame_gt(row, 'fabio'), axis=1
        )
        df_frame_not_both['ground_truth_paula'] = df_frame_not_both.apply(
            lambda row: self.get_frame_gt(row, 'paula'), axis=1
        )

        # calculate trained annotators agreement
        trained_annotators_agreement = self.intercoder_agreement_percentage(df_frame_not_both, ['ground_truth_fabio', 'ground_truth_paula'])
        trained_annotators_agreement_krippendorff = self.intercoder_agreement_krippendorff(df_frame_not_both, ['ground_truth_fabio', 'ground_truth_paula'])
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement: {trained_annotators_agreement:.4f}')
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement_krippendorff: {trained_annotators_agreement_krippendorff:.4f}')

        # Keep only rows where both annotators agree
        df_frame_with_gt = df_frame_not_both[
            (df_frame_not_both['ground_truth_fabio'] ==
                df_frame_not_both['ground_truth_paula'])
        ]
        df_frame_with_gt['ground_truth'] = copy.deepcopy(
            df_frame_with_gt['ground_truth_fabio']
        )

        nr_of_duplicates = len(df_relevant) - len(df_frame_without_duplicates)
        nr_of_rows_with_missing_annotations = len(df_frame_without_duplicates) - len(df_frame)
        nr_of_rows_with_invalid_annotations = len(df_frame) - len(df_frame_not_both)
        nr_of_rows_with_disagreement = len(df_frame_not_both) - len(df_frame_with_gt)

        dataset_statistics = {
            'Dataset name': self.get_dataset_name(),
            'Total rows (raw)': len(df_raw),
            'Total rows': len(df),
            'Total relevant rows': len(df_relevant),
            'Rows without duplicate values': len(df_frame_without_duplicates),
            'Rows without GT NaN values': len(df_frame),
            'Rows without both values (problem and solution)': len(df_frame_not_both),
            'Trained annotators agreement': trained_annotators_agreement,
            'Trained annotators agreement (krippendorff)': trained_annotators_agreement_krippendorff,
            'Rows with duplicates (based on id)': nr_of_duplicates_based_on_id,
            'Rows with duplicates': nr_of_duplicates,
            'Rows with missing annotations': nr_of_rows_with_missing_annotations,
            'Rows with invalid annotations': nr_of_rows_with_invalid_annotations,
            'Rows with disagreement': nr_of_rows_with_disagreement,
            'Final dataset size': len(df_frame_with_gt),
            'Value counts': df_frame_with_gt['ground_truth'].value_counts(),
        }
        for k, v in dataset_statistics.items():
            print(f"    {k}: {v}")

        df_frame_with_gt = self.order_df_columns(df_frame_with_gt)

        if return_statistics_for_plotting:
            return df_frame_with_gt, dataset_statistics
        else:
            return df_frame_with_gt

    def _get_mturk_file_path(self):
        return 'data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/mTurk_data/batch_results_problem_solution_final.csv'



class Gilardi2023_Data1Task3Tweets_2020_2021(Gilardi2023):

    def __init__(self, data_directory='gilardi_et_al_pnas', config_fn='config_framesII_tweets.yaml'):
        super().__init__(data_directory, config_fn)

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following tweet is mainly about the topic {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Tweet: {text}

Probability:'''

    def get_prompts(self):
        # prompt copied from paper Appendix F: https://www.pnas.org/lookup/suppl/doi:10.1073/pnas.2305016120/-/DCSupplemental
        frame_problem_solution_prompt = "Content moderation, as described above, can be linked to various other topics, such as health, crime, or equality.\nFor each tweet in the sample, follow these instructions:\n1. Carefully read the text of the tweet, paying close attention to details.\n2. Classify the tweet into one of the topics defined below.\nThe topics are defined as follows:\n• ECONOMY: The costs, benefits, or monetary/financial implications of the issue (to an individual, family, community, or to the economy as a whole).\n• Capacity and resources: The lack of or availability of physical, geographical, spatial, human, and financial resources, or the capacity of existing systems and resources to implement or carry out policy goals.\n• MORALITY: Any perspective—or policy objective or action (including proposed action)that is compelled by religious doctrine or interpretation, duty, honor, righteousness or any other sense of ethics or social responsibility.\n• FAIRNESS AND EQUALITY: Equality or inequality with which laws, punishment, rewards, and resources are applied or distributed among individuals or groups. Also the balance between the rights or interests of one individual or group compared to another individual or group.\n• CONSTITUTIONALITY AND JURISPRUDENCE: The constraints imposed on or freedoms granted to individuals, government, and corporations via the Constitution, Bill of Rights and other amendments, or judicial interpretation. This deals specifically with the authority of government to regulate, and the authority of individuals/corporations to act independently of government.\n• POLICY PRESCRIPTION AND EVALUATION: Particular policies proposed for addressing an identified problem, and figuring out if certain policies will work, or if existing policies are effective.\n• LAW AND ORDER, CRIME AND JUSTICE: Specific policies in practice and their enforcement, incentives, and implications. Includes stories about enforcement and interpretation of laws by individuals and law enforcement, breaking laws, loopholes, fines, sentencing and punishment. Increases or reductions in crime.\n• SECURITY AND DEFENSE: Security, threats to security, and protection of one’s person, family, in-group, nation, etc. Generally an action or a call to action that can be taken to protect the welfare of a person, group, nation sometimes from a not yet manifested threat.\n• HEALTH AND SAFETY: Health care access and effectiveness, illness, disease, sanitation, obesity, mental health effects, prevention of or perpetuation of gun violence, infrastructure and building safety.\n• QUALITY OF LIFE: The effects of a policy on individuals’ wealth, mobility, access to resources, happiness, social structures, ease of day-to-day routines, quality of community life, etc.\n• CULTURAL IDENTITY: The social norms, trends, values and customs constituting culture(s), as they relate to a specific policy issue.\n• PUBLIC OPINION: References to general social attitudes, polling and demographic information, as well as implied or actual consequences of diverging from or “getting ahead of” public opinion or polls.\n• POLITICAL: Any political considerations surrounding an issue. Issue actions or efforts or stances that are political, such as partisan filibusters, lobbyist involvement, bipartisan efforts, deal-making and vote trading, appealing to one’s base, mentions of political maneuvering. Explicit statements that a policy issue is good or bad for a particular political party.\n• EXTERNAL REGULATION AND REPUTATION: The United States’ external relations with another nation; the external relations of one state with another; or relations between groups. This includes trade agreements and outcomes, comparisons of policy outcomes or desired policy outcomes.\n• OTHER: Any topic that does not fit into the above categories.\n"
        

        gilardi_frame_prompt_paper = self.gilardi_content_moderation_prompt_paper + \
            frame_problem_solution_prompt

        tweet_instruction_prompt = "Here's the tweet I picked, please label it as 'Economy', 'Capacity and resources', 'Morality', 'Fairness and Equality', 'Constitutionality and Jurisprudence', 'Policy Prescription and Evaluation', 'Law and Order, Crime and Justice', 'Security and Defense', 'Health and Safety', 'Quality of Life', 'Cultural Identity', 'Public Opinion', 'Political', 'External Regulation and Reputation', or 'Other' by answering with one word:\n{text}"



        # the following prompts are copied from Gilardi 2024 code: data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv
        # available for download at: https://osf.io/adkun/files/osfstorage
        # a=pd.read_csv("data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv")
        frame_specific_prompt_code_2024_zero = ["“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as one of the frames defined below:\n\nECONOMY: The costs, benefits, or monetary/financial implications of the issue (to an individual, family, community, or to the economy as a whole).\nMORALITY: Any perspective—or policy objective or action (including proposed action)that is compelled by religious doctrine or interpretation, duty, honor, righteousness or any other sense of ethics or social responsibility.\nFAIRNESS AND EQUALITY: Equality or inequality with which laws, punishment, rewards, and resources are applied or distributed among individuals or groups. Also the balance between the rights or interests of one individual or group compared to another individual or group.\nPOLICY PRESCRIPTION AND EVALUATION: Particular policies proposed for addressing an identified problem, and figuring out if certain policies will work, or if existing policies are effective.\nLAW AND ORDER, CRIME AND JUSTICE: Specific policies in practice and their enforcement, incentives, and implications. Includes stories about enforcement and interpretation of laws by individuals and law enforcement, breaking laws, loopholes, fines, sentencing and punishment. Increases or reductions in crime.\nSECURITY AND DEFENSE: Security, threats to security, and protection of one’s person, family, in-group, nation, etc. Generally an action or a call to action that can be taken to protect the welfare of a person, group, nation sometimes from a not yet manifested threat.\nHEALTH AND SAFETY: Health care access and effectiveness, illness, disease, sanitation, obesity, mental health effects, prevention of or perpetuation of gun violence, infrastructure and building safety.\nQUALITY OF LIFE: The effects of a policy on individuals’ wealth, mobility, access to resources, happiness, social structures, ease of day-to-day routines, quality of community life, etc.\nPOLITICAL: Any political considerations surrounding an issue. Issue actions or efforts or stances that are political, such as partisan filibusters, lobbyist involvement, bipartisan efforts, deal-making and vote trading, appealing to one's base, mentions of political maneuvering. Explicit statements that a policy issue is good or bad for a particular political party.\nEXTERNAL REGULATION AND REPUTATION: The United States’ external relations with another nation; the external relations of one state with another; or relations between groups. This includes trade agreements and outcomes, comparisons of policy outcomes or desired policy outcomes.\nOTHER: Any topic that does not fit into the above categories.", "Now, which of the bellow frames best fit the following tweet? Answer with only the option below that is most accurate and nothing else.\n\nA: ECONOMY \nB: MORALITY\nC: FAIRNESS AND EQUALITY\nD: POLICY PRESCRIPTION AND EVALUATION \nE: LAW AND ORDER, CRIME AND JUSTICE\nF: SECURITY AND DEFENSE\nG: HEALTH AND SAFETY\nH: QUALITY OF LIFE\nI: POLITICAL\nJ: EXTERNAL REGULATION AND REPUTATION\nK: OTHER\n\n{text}"]



        frame_specific_prompt_code_2024_few = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as one of the frames defined below:\n\nECONOMY: The costs, benefits, or monetary/financial implications of the issue (to an individual, family, community, or to the economy as a whole).\nMORALITY: Any perspective—or policy objective or action (including proposed action)that is compelled by religious doctrine or interpretation, duty, honor, righteousness or any other sense of ethics or social responsibility.\nFAIRNESS AND EQUALITY: Equality or inequality with which laws, punishment, rewards, and resources are applied or distributed among individuals or groups. Also the balance between the rights or interests of one individual or group compared to another individual or group.\nPOLICY PRESCRIPTION AND EVALUATION: Particular policies proposed for addressing an identified problem, and figuring out if certain policies will work, or if existing policies are effective.\nLAW AND ORDER, CRIME AND JUSTICE: Specific policies in practice and their enforcement, incentives, and implications. Includes stories about enforcement and interpretation of laws by individuals and law enforcement, breaking laws, loopholes, fines, sentencing and punishment. Increases or reductions in crime.\nSECURITY AND DEFENSE: Security, threats to security, and protection of one’s person, family, in-group, nation, etc. Generally an action or a call to action that can be taken to protect the welfare of a person, group, nation sometimes from a not yet manifested threat.\nHEALTH AND SAFETY: Health care access and effectiveness, illness, disease, sanitation, obesity, mental health effects, prevention of or perpetuation of gun violence, infrastructure and building safety.\nQUALITY OF LIFE: The effects of a policy on individuals’ wealth, mobility, access to resources, happiness, social structures, ease of day-to-day routines, quality of community life, etc.\nPOLITICAL: Any political considerations surrounding an issue. Issue actions or efforts or stances that are political, such as partisan filibusters, lobbyist involvement, bipartisan efforts, deal-making and vote trading, appealing to one's base, mentions of political maneuvering. Explicit statements that a policy issue is good or bad for a particular political party.\nEXTERNAL REGULATION AND REPUTATION: The United States’ external relations with another nation; the external relations of one state with another; or relations between groups. This includes trade agreements and outcomes, comparisons of policy outcomes or desired policy outcomes.\nOTHER: Any topic that does not fit into the above categories.\n\nThe following tweets are some examples of these frames in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "TY @AGSchneiderman for your investigation into red light camera violations by school bus companies. Our children’s safety is paramount. https://t.co/CVLj7Lk6r4"
            },
            {
                "role": "assistant",
                "content": "G"
            },
            {
                "role": "user",
                "content": "Had a huge turnout for the #GrowWithGoogle digital skills event, which I hosted with @google today in Charleston! Programs like this one will help strengthen our workforce &amp; create a brighter economic future for West Virginians. https://t.co/3ma7Mv1EBR"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "On the floor about to offer and speak on my amendment to eliminate subsistence fees at BOP halfway houses, which house DC Code felons."
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": "It is refreshing that the #JointAddress tonight was a call to unite the country around a strategy for a stronger, brighter future."
            },
            {
                "role": "assistant",
                "content": "I"
            },
            {
                "role": "user",
                "content": "Today we remember and honor the men and women of our Armed Forces who remain missing in action or prisoners of war. #POWMIARecognitionDay https://t.co/D9z1akkjKW"
            },
            {
                "role": "assistant",
                "content": "F"
            },
            {
                "role": "user",
                "content": "No longer can POTUS pretend that Putin’s Russia was not responsible for cyberattacks targeting the 2016 election. Today’s indictments are another example of why the Mueller investigation must continue unimpeded. The American people need to learn the truth. https://t.co/mYwE4p4jR4"
            },
            {
                "role": "assistant",
                "content": "J"
            },
            {
                "role": "user",
                "content": "Women have a valuable place in STEAM fields—let’s make sure they have a seat at the table. https://t.co/LhOawvSszP"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": """BREAKING: @HouseIntelComm Chairman Conaway just announced that the Committee is closing the "Russian collusion" investigation, having found zero evidence of any collusion between the Trump campaign and Russians. Case closed. It's time we return focus to the people's agenda."""
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": """The Trump-Sessions "zero tolerance" family separation border policies are not required, right or moral. https://t.co/aAFX8Q6eKT"""
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": """Glad to work with @TomRooney @RepSeanMaloney @RepStefanik on this bipartisan bill to help our homeless veterans. #HousingOurHeroes"""
            },
            {
                "role": "assistant",
                "content": "H"
            },
            {
                "role": "user",
                "content": "Prayers to my dear friend @SteveScalise &amp;all injured after this morning's horrific shooting.Thinking of their families in this shocking time"
            },
            {
                "role": "assistant",
                "content": "K"
            },
            {
                "role": "user",
                "content": "Now, which of the bellow frames best fit the following tweet? Answer with only the option below that is most accurate and nothing else.\n\nA: ECONOMY \nB: MORALITY\nC: FAIRNESS AND EQUALITY\nD: POLICY PRESCRIPTION AND EVALUATION \nE: LAW AND ORDER, CRIME AND JUSTICE\nF: SECURITY AND DEFENSE\nG: HEALTH AND SAFETY\nH: QUALITY OF LIFE\nI: POLITICAL\nJ: EXTERNAL REGULATION AND REPUTATION\nK: OTHER\n\n{text}"
            }
        ]


        frame_specific_prompt_code_2024_cot = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as one of the frames defined below:\n\nECONOMY: The costs, benefits, or monetary/financial implications of the issue (to an individual, family, community, or to the economy as a whole).\nMORALITY: Any perspective—or policy objective or action (including proposed action)that is compelled by religious doctrine or interpretation, duty, honor, righteousness or any other sense of ethics or social responsibility.\nFAIRNESS AND EQUALITY: Equality or inequality with which laws, punishment, rewards, and resources are applied or distributed among individuals or groups. Also the balance between the rights or interests of one individual or group compared to another individual or group.\nPOLICY PRESCRIPTION AND EVALUATION: Particular policies proposed for addressing an identified problem, and figuring out if certain policies will work, or if existing policies are effective.\nLAW AND ORDER, CRIME AND JUSTICE: Specific policies in practice and their enforcement, incentives, and implications. Includes stories about enforcement and interpretation of laws by individuals and law enforcement, breaking laws, loopholes, fines, sentencing and punishment. Increases or reductions in crime.\nSECURITY AND DEFENSE: Security, threats to security, and protection of one’s person, family, in-group, nation, etc. Generally an action or a call to action that can be taken to protect the welfare of a person, group, nation sometimes from a not yet manifested threat.\nHEALTH AND SAFETY: Health care access and effectiveness, illness, disease, sanitation, obesity, mental health effects, prevention of or perpetuation of gun violence, infrastructure and building safety.\nQUALITY OF LIFE: The effects of a policy on individuals’ wealth, mobility, access to resources, happiness, social structures, ease of day-to-day routines, quality of community life, etc.\nPOLITICAL: Any political considerations surrounding an issue. Issue actions or efforts or stances that are political, such as partisan filibusters, lobbyist involvement, bipartisan efforts, deal-making and vote trading, appealing to one's base, mentions of political maneuvering. Explicit statements that a policy issue is good or bad for a particular political party.\nEXTERNAL REGULATION AND REPUTATION: The United States’ external relations with another nation; the external relations of one state with another; or relations between groups. This includes trade agreements and outcomes, comparisons of policy outcomes or desired policy outcomes.\nOTHER: Any topic that does not fit into the above categories.\n\nThe following tweets are some examples of these frames in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "TY @AGSchneiderman for your investigation into red light camera violations by school bus companies. Our children’s safety is paramount. https://t.co/CVLj7Lk6r4"
            },
            {
                "role": "assistant",
                "content": """The frame that best fits this text is "HEALTH AND SAFETY". This is because the text is expressing concern for the safety of children riding on school buses, which is a health and safety issue. Additionally, the tweet is thanking Attorney General Eric Schneiderman for investigating red light camera violations by school bus companies, implying that there may be safety violations happening that need to be addressed."""
            },
            {
                "role": "user",
                "content": "Had a huge turnout for the #GrowWithGoogle digital skills event, which I hosted with @google today in Charleston! Programs like this one will help strengthen our workforce &amp; create a brighter economic future for West Virginians. https://t.co/3ma7Mv1EBR"
            },
            {
                "role": "assistant",
                "content": "The best-fitted frame for the given text is ECONOMY. The text talks about a digital skills event hosted by the author along with Google to strengthen the workforce, which indicates that the author is concerned about economic growth and creating a better future for West Virginians."
            },
            {
                "role": "user",
                "content": "On the floor about to offer and speak on my amendment to eliminate subsistence fees at BOP halfway houses, which house DC Code felons."
            },
            {
                "role": "assistant",
                "content": "Based on the provided text, the most accurate frame would be: POLICY PRESCRIPTION AND EVALUATION. The text mentions the speaker's intention to offer an amendment to eliminate subsistence fees at BOP halfway houses that house DC Code felons. This indicates a specific policy proposal aimed at addressing a problem related to subsistence fees for felons in halfway houses. Therefore, it falls under the frame of policy prescription and evaluation."
            },
            {
                "role": "user",
                "content": "It is refreshing that the #JointAddress tonight was a call to unite the country around a strategy for a stronger, brighter future."
            },
            {
                "role": "assistant",
                "content": "Based on the provided text, the most accurate frame would be: POLITICAL. The text refers to the #JointAddress, which typically refers to a political event where the President of the United States addresses the nation. It mentions the call to unite the country around a strategy for a stronger, brighter future, which indicates a political consideration aimed at rallying support and fostering unity. Therefore, the text falls under the frame of political considerations."
            },
            {
                "role": "user",
                "content": "Today we remember and honor the men and women of our Armed Forces who remain missing in action or prisoners of war. #POWMIARecognitionDay https://t.co/D9z1akkjKW"
            },
            {
                "role": "assistant",
                "content": "The text you provided can be classified under the frame of SECURITY AND DEFENSE. It is a statement that acknowledges and honors the men and women of the Armed Forces who are missing in action or prisoners of war. It emphasizes the importance of remembering and honoring their sacrifice, which relates to the protection and welfare of individuals and the nation's security."
            },
            {
                "role": "user",
                "content": "No longer can POTUS pretend that Putin’s Russia was not responsible for cyberattacks targeting the 2016 election. Today’s indictments are another example of why the Mueller investigation must continue unimpeded. The American people need to learn the truth. https://t.co/mYwE4p4jR4"
            },
            {
                "role": "assistant",
                "content": "The following frame is the best fit: EXTERNAL REGULATION AND REPUTATION: The text highlights the responsibility of Putin's Russia for cyberattacks targeting the 2016 election and emphasizes the need for the Mueller investigation to continue unimpeded. It mentions the truth that the American people need to learn. This relates to the external relations and reputation of the United States and its interactions with another nation (Russia) in the context of cyberattacks and election interference."
            },
            {
                "role": "user",
                "content": "Women have a valuable place in STEAM fields—let’s make sure they have a seat at the table. https://t.co/LhOawvSszP"
            },
            {
                "role": "assistant",
                "content": "Based on the provided text, the most accurate classification would be: FAIRNESS AND EQUALITY. The text emphasizes the importance of ensuring women have a seat at the table in STEAM (Science, Technology, Engineering, Arts, and Mathematics) fields. It addresses the equality or inequality of opportunities and representation for women in these fields, highlighting the need for fairness and equal participation."
            },
            {
                "role": "user",
                "content": """BREAKING: @HouseIntelComm Chairman Conaway just announced that the Committee is closing the "Russian collusion" investigation, having found zero evidence of any collusion between the Trump campaign and Russians. Case closed. It's time we return focus to the people's agenda."""
            },
            {
                "role": "assistant",
                "content": """The text can be classified under the following frame: LAW AND ORDER, CRIME AND JUSTICE. The text mentions the investigation into "Russian collusion" and the finding of zero evidence of collusion between the Trump campaign and Russians. It highlights the closure of the investigation and suggests a shift in focus to the people's agenda. This classification relates to specific policies in practice (investigation), enforcement (finding of evidence), and implications (closure) within the realm of law and order, crime, and justice."""
            },
            {
                "role": "user",
                "content": """The Trump-Sessions "zero tolerance" family separation border policies are not required, right or moral. https://t.co/aAFX8Q6eKT"""
            },
            {
                "role": "assistant",
                "content": """Based on the provided text, the best fit among the defined frames would be: MORALITY. The text expresses a perspective on the Trump-Sessions "zero tolerance" family separation border policies, stating that they are not required, right, or moral. This indicates a moral evaluation of the policies based on ethics and social responsibility."""
            },
            {
                "role": "user",
                "content": "Glad to work with @TomRooney @RepSeanMaloney @RepStefanik on this bipartisan bill to help our homeless veterans. #HousingOurHeroes"
            },
            {
                "role": "assistant",
                "content": """The text highlights a bipartisan bill aimed at helping homeless veterans, which directly relates to improving the quality of life for these individuals. The focus is on the effects of the policy on individuals' well-being, access to resources, and housing stability.\nTherefore, the frame "QUALITY OF LIFE" is the most accurate classification for the given text."""
            },
            {
                "role": "user",
                "content": "Prayers to my dear friend @SteveScalise &amp;all injured after this morning's horrific shooting.Thinking of their families in this shocking time"
            },
            {
                "role": "assistant",
                "content": """The text expresses sympathy and concern for the injured individuals and their families after a shooting incident, but it doesn't explicitly address the broader implications or aspects related to economy, morality, fairness and equality, policy prescription and evaluation, law and order, crime and justice, health and safety, quality of life, political considerations, external regulation and reputation, or any other specific frame. Therefore, the most accurate classification for this text would be "OTHER," as it doesn't fit into the predefined frames provided."""
            },
            {
                "role": "user",
                "content": "Now, which of the bellow frames best fit the following tweet? Answer with only the option below that is most accurate and nothing else.\n\nA: ECONOMY \nB: MORALITY\nC: FAIRNESS AND EQUALITY\nD: POLICY PRESCRIPTION AND EVALUATION \nE: LAW AND ORDER, CRIME AND JUSTICE\nF: SECURITY AND DEFENSE\nG: HEALTH AND SAFETY\nH: QUALITY OF LIFE\nI: POLITICAL\nJ: EXTERNAL REGULATION AND REPUTATION\nK: OTHER\n\n{text}"
            }
        ]

        # frame_specific_promot_paper_2024 = ""


        return [
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets_2020_2021 frame',
                    'compatible_output_mapping': ['gilardi political frame'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": gilardi_frame_prompt_paper
                        },
                        {
                            "role": "user",
                            "content": tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets_2020_2021 frame (no system prompt)',
                    'compatible_output_mapping': ['gilardi political frame'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": gilardi_frame_prompt_paper + "\n" + tweet_instruction_prompt

                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets_2020_2021 frame',
                    'compatible_output_mapping': [
                        # 'gilardi political frame', 
                        'gilardi political frame (extended with A to K)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": frame_specific_prompt_code_2024_zero[0]
                        },
                        {
                            "role": "user",
                            "content": frame_specific_prompt_code_2024_zero[1]
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets_2020_2021 frame (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi political frame', 
                        'gilardi political frame (extended with A to K)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": frame_specific_prompt_code_2024_zero[0] + "\n\n" + frame_specific_prompt_code_2024_zero[1]
                        },
                    ],
                },
                {
                    'description': '[original] [few-shot-detailed] gilardi 2024 (code) tweets_2020_2021 frame',
                    'compatible_output_mapping': [
                        # 'gilardi political frame', 
                        'gilardi political frame (extended with A to K)'],
                    'prompt_text': frame_specific_prompt_code_2024_few,
                },
            ]

    def get_all_output_mappings(self):

        return {
            'gilardi political frame': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.frame_political_output_mapping},
            },
            'gilardi political frame (extended with A to K)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.frame_political_output_mapping_extended_A_to_K},
            },
        }

    def get_dataset_name(self):
        """Get the name of the dataset."""
        return "Gilardi2023_Data1Task3Tweets_2020_2021"



    def load_full_dataset(self, return_statistics_for_plotting=False, return_dataset_used_by_gilardi=False):
        """Load and prepare the Gilardi2023_Data1Task3Tweets_2020_2021 dataset."""
        # Load data from Excel files
        df_raw = pd.read_excel("data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/research_assistants_data/frames_task/tweets_annotation_data_ra_completed.xlsx")

        df = df_raw.dropna(subset=['relevant'], how='all')

        # Filter to only relevant tweets and process for frame task
        df_relevant = df[df['relevant'] == 1]

        # Now drop duplicates
        df_frame_without_duplicates = df_relevant.drop_duplicates(subset=['status_id'], keep='first')
        nr_of_duplicates_based_on_id = len(df_relevant) - len(df_frame_without_duplicates)
        if not return_dataset_used_by_gilardi:
            df_frame_without_duplicates = df_frame_without_duplicates.drop_duplicates(subset=['text'], keep='first')
        print(f"\nRows after dropping duplicates: {len(df_frame_without_duplicates)}")

        df_frame = df_frame_without_duplicates.dropna(subset=['frame_name_primary_fabio', 'frame_name_primary_paula'])

        # calculate trained annotators agreement
        trained_annotators_agreement = self.intercoder_agreement_percentage(df_frame, ['frame_name_primary_fabio', 'frame_name_primary_paula'])
        trained_annotators_agreement_krippendorff = self.intercoder_agreement_krippendorff(df_frame, ['frame_name_primary_fabio', 'frame_name_primary_paula'])
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement: {trained_annotators_agreement:.4f}')
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement_krippendorff: {trained_annotators_agreement_krippendorff:.4f}')

        # Keep only rows where both annotators agree
        df_frame_with_gt = df_frame[
            (df_frame['frame_name_primary_fabio'] == df_frame['frame_name_primary_paula'])
        ]
        df_frame_with_gt['ground_truth'] = copy.deepcopy(
            df_frame_with_gt['frame_name_primary_fabio']
        )

        nr_of_duplicates = len(df_relevant) - len(df_frame_without_duplicates)
        nr_of_rows_with_missing_annotations = len(df_frame_without_duplicates) - len(df_frame)
        nr_of_rows_with_disagreement = len(df_frame) - len(df_frame_with_gt)

        dataset_statistics = {
            'Dataset name': self.get_dataset_name(),
            'Total rows': len(df),
            'Total rows (raw)': len(df_raw),
            'Total relevant rows': len(df_relevant),
            'Rows without duplicate values': len(df_frame_without_duplicates),
            'Rows without GT NaN values': len(df_frame),
            # 'Rows without both values (problem and solution)': len(df_frame),
            'Trained annotators agreement': trained_annotators_agreement,
            'Trained annotators agreement (krippendorff)': trained_annotators_agreement_krippendorff,
            'Rows with duplicates (based on id)': nr_of_duplicates_based_on_id,
            'Rows with duplicates': nr_of_duplicates,
            'Rows with missing annotations': nr_of_rows_with_missing_annotations,
            'Rows with disagreement': nr_of_rows_with_disagreement,
            'Final dataset size': len(df_frame_with_gt),
            'Value counts': df_frame_with_gt['ground_truth'].value_counts(),
        }
        for k, v in dataset_statistics.items():
            print(f"    {k}: {v}")

        df_frame_with_gt['ground_truth'] = df_frame_with_gt.apply(self.get_detailed_frames_ground_truth, axis=1)

        df_frame_with_gt = self.order_df_columns(df_frame_with_gt)

        if return_statistics_for_plotting:
            return df_frame_with_gt, dataset_statistics
        else:
            return df_frame_with_gt



class Gilardi2023_Data1Task4Tweets_2020_2021(Gilardi2023):

    def __init__(self, data_directory='gilardi_et_al_pnas', config_fn='config_stance_tweets.yaml'):
        super().__init__(data_directory, config_fn)

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the stance of the following tweet toward Section 230 is {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Tweet: {text}

Probability:'''

    def get_prompts(self):
        # prompt copied from paper Appendix H: https://www.pnas.org/lookup/suppl/doi:10.1073/pnas.2305016120/-/DCSupplemental
        stance_prompt = "\nIn the context of content moderation, Section 230 is a law in the United States that protects websites and other online platforms from being held legally responsible for the content posted by their users. This means that if someone posts something illegal or harmful on a website, the website itself cannot be sued for allowing it to be posted. However, websites can still choose to moderate content and remove anything that violates their own policies.\nFor each tweet in the sample, follow these instructions:\n1. Carefully read the text of the tweet, paying close attention to details.\n2. Classify the tweet as having a positive stance towards Section 230, a negative stance, or a neutral stance."

        gilardi_stance_prompt_code = self.gilardi_content_moderation_prompt_code + \
            stance_prompt
        gilardi_stance_prompt_paper = self.gilardi_content_moderation_prompt_paper + \
            stance_prompt

        tweet_instruction_prompt = "Here's the tweet I picked, please label it as 'Positive', 'Negative', or 'Neutral' by answering with one word:\n{text}"


        # the following prompts are copied from Gilardi 2024 code: data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv
        # available for download at: https://osf.io/adkun/files/osfstorage
        # a=pd.read_csv("data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv")
        stance_specific_prompt_code_2024_zero = ["“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines. In the context of content moderation, Section 230 is a law in the United States that protects websites and other online platforms from being held legally responsible for the content posted by their users. This means that if someone posts something illegal or harmful on a website, the website itself cannot be sued for allowing it to be posted. However, websites can still choose to moderate content and remove anything that violates their own policies.\n\nI will ask you to classify a tweet as 'IN FAVOR OF', 'AGAINST', or 'NEUTRAL' about Section 230:\n\nA: “IN FAVOR  OF” expresses approval for Section 230 and/or advocates keeping Section 230\nB: “AGAINST” expresses disapproval towards Section 230 and/or advocates repealing Section 230\nC: “NEUTRAL” discusses Section 230 without expressing approval or disapproval towards it", "Now, is the following tweet IN FAVOR OF, AGAINST, or NEUTRAL about Section 230?\n\n{text}"]


        stance_specific_prompt_code_2024_few = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines. In the context of content moderation, Section 230 is a law in the United States that protects websites and other online platforms from being held legally responsible for the content posted by their users. This means that if someone posts something illegal or harmful on a website, the website itself cannot be sued for allowing it to be posted. However, websites can still choose to moderate content and remove anything that violates their own policies.\n\nI will ask you to classify a tweet as 'IN FAVOR OF', 'AGAINST', or 'NEUTRAL' about Section 230:\n\nA: “IN FAVOR  OF” expresses approval for Section 230 and/or advocates keeping Section 230\nB: “AGAINST” expresses disapproval towards Section 230 and/or advocates repealing Section 230\nC: “NEUTRAL” discusses Section 230 without expressing approval or disapproval towards it\n\nThe following tweets are some examples of texts “IN FAVOR OF” ,“AGAINST” or “NEUTRAL” about section230 in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "#Section230 is poorly understood by lawmakers on both sides of the aisle, and it is dangerous for them to use it as a political football. To save online free speech, we must #Protect230 Contact lawmakers: https://t.co/ldSL75knH4"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Facebook and Twitter CEOs warn against demolishing Section 230, the law that shields tech giants https://t.co/CItuLmTTxE by @alexiskweed https://t.co/7Y6eG19YoZ"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "How do we get big tech companies like Twitter to abide by the spirit of the 1st Amendment, or moderate content by a set of clear and definable standards? Canyon Brimhall joins in the third episode of our series on big tech, free speech, and Section 230. https://t.co/RfoJVuQPEh https://t.co/FheCcceTMr"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "I sat down with the Meme King himself @bennyjohnson in Washington, DC to discuss Big Tech, Social Media Censorship and its impact on the election, Section 230, \nPLEASE DO NOT MISS THIS! https://t.co/ETWKY16J8d"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "@JoeBiden Trump MUST be re-elected. We MUST vote in GOP in the House, keep the Senate and we MUST eliminate Section 230 protections for big tech social media! Social media &amp; Corporate America will kill our free speech! BLM is a Marxist group looking to DESTROY USA! https://t.co/R37Qi2rlNy"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "BREAKING: Supreme Court Justice Clarence Thomas finds social media companies do not have First Amendment right to ban protected speech and that Section 230 is unconstitutional"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Now, is the following tweet IN FAVOR OF, AGAINST, or NEUTRAL about Section 230?\n\n{text}"
            }
        ]
        
        stance_specific_prompt_code_2024_cot = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines. In the context of content moderation, Section 230 is a law in the United States that protects websites and other online platforms from being held legally responsible for the content posted by their users. This means that if someone posts something illegal or harmful on a website, the website itself cannot be sued for allowing it to be posted. However, websites can still choose to moderate content and remove anything that violates their own policies.\n\nI will ask you to classify a tweet as 'IN FAVOR OF', 'AGAINST', or 'NEUTRAL' about Section 230:\n\nA: “IN FAVOR  OF” expresses approval for Section 230 and/or advocates keeping Section 230\nB: “AGAINST” expresses disapproval towards Section 230 and/or advocates repealing Section 230\nC: “NEUTRAL” discusses Section 230 without expressing approval or disapproval towards it\n\nThe following tweets are some examples of texts “IN FAVOR OF” ,“AGAINST” or “NEUTRAL” about section230 in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "#Section230 is poorly understood by lawmakers on both sides of the aisle, and it is dangerous for them to use it as a political football. To save online free speech, we must #Protect230 Contact lawmakers: https://t.co/ldSL75knH4"
            },
            {
                "role": "assistant",
                "content": """The provided text is in favor of Section 230 and advocates keeping it. The use of the hashtag #Protect230 and the statement "To save online free speech, we must #Protect230" indicate support for the law and its role in preserving online free speech. Therefore, the text can be classified as "IN FAVOR OF" Section 230."""
            },
            {
                "role": "user",
                "content": "Facebook and Twitter CEOs warn against demolishing Section 230, the law that shields tech giants https://t.co/CItuLmTTxE by @alexiskweed https://t.co/7Y6eG19YoZ"
            },
            {
                "role": "assistant",
                "content": """The provided text is in favor of Section 230. It mentions that the CEOs of Facebook and Twitter are warning against demolishing Section 230, indicating their support for the law. Therefore, the text can be classified as "IN FAVOR OF" Section 230."""
            },
            {
                "role": "user",
                "content": "How do we get big tech companies like Twitter to abide by the spirit of the 1st Amendment, or moderate content by a set of clear and definable standards? Canyon Brimhall joins in the third episode of our series on big tech, free speech, and Section 230. https://t.co/RfoJVuQPEh https://t.co/FheCcceTMr"
            },
            {
                "role": "assistant",
                "content": """The provided text discusses the need for big tech companies to abide by the spirit of the First Amendment and moderate content according to clear and definable standards. It mentions Section 230 in the context of a discussion about big tech, free speech, and content moderation. The text does not express explicit approval or disapproval towards Section 230 itself but raises the question of how to ensure big tech companies adhere to the principles of the First Amendment. Therefore, the text can be classified as "NEUTRAL" about Section 230."""
            },
            {
                "role": "user",
                "content": "I sat down with the Meme King himself @bennyjohnson in Washington, DC to discuss Big Tech, Social Media Censorship and its impact on the election, Section 230, \nPLEASE DO NOT MISS THIS! https://t.co/ETWKY16J8d"
            },
            {
                "role": "assistant",
                "content": """The provided text does not clearly express whether it is in favor of or against Section 230. It discusses various topics such as Big Tech, social media censorship, Section 230, and its impact on the election without explicitly stating approval or disapproval towards Section 230. Therefore, the text can be classified as "NEUTRAL" about Section 230."""
            },
            {
                "role": "user",
                "content": "@JoeBiden Trump MUST be re-elected. We MUST vote in GOP in the House, keep the Senate and we MUST eliminate Section 230 protections for big tech social media! Social media &amp; Corporate America will kill our free speech! BLM is a Marxist group looking to DESTROY USA! https://t.co/R37Qi2rlNy"
            },
            {
                "role": "assistant",
                "content": """The provided text is against Section 230. It advocates for eliminating Section 230 protections for big tech social media and expresses concerns about the impact of social media and corporate America on free speech. Therefore, the text can be classified as "AGAINST" Section 230."""
            },
            {
                "role": "user",
                "content": "BREAKING: Supreme Court Justice Clarence Thomas finds social media companies do not have First Amendment right to ban protected speech and that Section 230 is unconstitutional"
            },
            {
                "role": "assistant",
                "content": """The provided text expresses a critical stance towards Section 230 by stating that Supreme Court Justice Clarence Thomas finds it unconstitutional and that social media companies do not have a First Amendment right to ban protected speech. Therefore, the text can be classified as "AGAINST" Section 230."""
            },
            {
                "role": "user",
                "content": "Now, is the following tweet IN FAVOR OF, AGAINST, or NEUTRAL about Section 230?\n\n{text}"
            }
        ]

        # stance_specific_promot_paper_2024 = ""


        return [
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (code) tweets_2020_2021 stance',
                    'compatible_output_mapping': ['gilardi stance 2023'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": gilardi_stance_prompt_code
                        },
                        {
                            "role": "user",
                            "content": tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (code) tweets_2020_2021 stance (no system prompt)',
                    'compatible_output_mapping': ['gilardi stance 2023'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": gilardi_stance_prompt_code + "\n" + tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets_2020_2021 stance',
                    'compatible_output_mapping': ['gilardi stance 2023'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": gilardi_stance_prompt_paper
                        },
                        {
                            "role": "user",
                            "content": tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets_2020_2021 stance (no system prompt)',
                    'compatible_output_mapping': ['gilardi stance 2023'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": gilardi_stance_prompt_paper + "\n" + tweet_instruction_prompt

                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets_2020_2021 stance',
                    'compatible_output_mapping': [
                        # 'gilardi stance 2024', 
                        'gilardi stance 2024 classes (extended with A and B and C)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": stance_specific_prompt_code_2024_zero[0]
                        },
                        {
                            "role": "user",
                            "content": stance_specific_prompt_code_2024_zero[1]
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets_2020_2021 stance (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi stance 2024', 
                        'gilardi stance 2024 classes (extended with A and B and C)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": stance_specific_prompt_code_2024_zero[0] + "\n\n" + stance_specific_prompt_code_2024_zero[1]
                        },
                    ],
                },
                {
                    'description': '[original] [few-shot-detailed] gilardi 2024 (code) tweets_2020_2021 stance',
                    'compatible_output_mapping': [
                        # 'gilardi stance 2024', 
                        'gilardi stance 2024 classes (extended with A and B and C)'],
                    'prompt_text': stance_specific_prompt_code_2024_few,
                },
                # {
                #     'description': 'gilardi 2024 (code) tweets_2020_2021 stance cot',
                #     'compatible_output_mapping': [
                #         # 'gilardi stance 2024', 
                #         'gilardi stance 2024 classes (extended with A and B and C)'],
                #     'prompt_text': stance_specific_prompt_code_2024_cot,
                # },
            ]

    def get_all_output_mappings(self):

        return {
            'gilardi stance 2023': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.stance_2023_output_mapping},
            },
            'gilardi stance 2024': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.stance_2024_classes_output_mapping},
            },
            'gilardi stance 2024 classes (extended with A and B and C)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.stance_2024_classes_output_mapping_extended_A_and_B_and_C},
            },
        }


    def get_dataset_name(self):
        """Get the name of the dataset."""
        return "Gilardi2023_Data1Task4Tweets_2020_2021"

    def load_full_dataset(self, return_statistics_for_plotting=False, return_dataset_used_by_gilardi=False):
        """Load and prepare the Gilardi 2023 stance dataset."""
        # Load data from Excel files
        df_raw = pd.read_excel("data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/research_assistants_data/stance_task/section230_training_data_completed.xlsx")
        df = df_raw.dropna(subset=['paula_stance_pro',
                    'fabio_stance_pro',
                    'paula_stance_neutral',
                    'fabio_stance_neutral',
                    'paula_stance_contra',
                    'fabio_stance_contra'
            ], how='all')

        # Now drop duplicates
        df_stance_without_duplicates = df.drop_duplicates(subset=['status_id'], keep='first')
        nr_of_duplicates_based_on_id = len(df) - len(df_stance_without_duplicates)
        if not return_dataset_used_by_gilardi:
            df_stance_without_duplicates = df_stance_without_duplicates.drop_duplicates(subset=['text'], keep='first')
        print(f"\nRows after dropping duplicates: {len(df_stance_without_duplicates)}")

        df_stance = df_stance_without_duplicates.dropna(
            subset=['paula_stance_pro',
                    'fabio_stance_pro',
                    'paula_stance_neutral',
                    'fabio_stance_neutral',
                    'paula_stance_contra',
                    'fabio_stance_contra'
            ])

        df_stance['sum_of_stances_fabio'] = df_stance['fabio_stance_pro'] + df_stance['fabio_stance_neutral'] + df_stance['fabio_stance_contra']
        df_stance['sum_of_stances_paula'] = df_stance['paula_stance_pro'] + df_stance['paula_stance_neutral'] + df_stance['paula_stance_contra']

        df_stance_not_both = df_stance[(
            (
                (df_stance['sum_of_stances_fabio'] == 1) &
                (df_stance['sum_of_stances_paula'] == 1)
            )
        )]

        # Apply stance classification functions
        df_stance_not_both['ground_truth_fabio'] = df_stance_not_both.apply(
            lambda row: self.get_stance_gt(row, 'fabio'), axis=1
        )
        df_stance_not_both['ground_truth_paula'] = df_stance_not_both.apply(
            lambda row: self.get_stance_gt(row, 'paula'), axis=1
        )

        # calculate trained annotators agreement
        trained_annotators_agreement = self.intercoder_agreement_percentage(df_stance_not_both, ['ground_truth_fabio', 'ground_truth_paula'])
        trained_annotators_agreement_krippendorff = self.intercoder_agreement_krippendorff(df_stance_not_both, ['ground_truth_fabio', 'ground_truth_paula'])
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement: {trained_annotators_agreement:.4f}')
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement_krippendorff: {trained_annotators_agreement_krippendorff:.4f}')

        # Keep only rows where both annotators agree
        df_stance_with_gt = df_stance_not_both[
            (df_stance_not_both['ground_truth_fabio'] ==
                df_stance_not_both['ground_truth_paula'])
        ]
        df_stance_with_gt['ground_truth'] = copy.deepcopy(
            df_stance_with_gt['ground_truth_fabio']
        )

        nr_of_duplicates = len(df) - len(df_stance_without_duplicates)
        nr_of_rows_with_missing_annotations = len(df_stance_without_duplicates) - len(df_stance)
        nr_of_rows_with_invalid_annotations = len(df_stance) - len(df_stance_not_both)
        nr_of_rows_with_disagreement = len(df_stance_not_both) - len(df_stance_with_gt)

        dataset_statistics = {
            'Dataset name': self.get_dataset_name(),
            'Total rows (raw)': len(df_raw),
            'Total rows': len(df),
            'Rows without duplicate values': len(df_stance_without_duplicates),
            'Rows without GT NaN values': len(df_stance),
            'Rows without both values (problem and solution)': len(df_stance_not_both),
            'Trained annotators agreement': trained_annotators_agreement,
            'Trained annotators agreement (krippendorff)': trained_annotators_agreement_krippendorff,
            'Rows with duplicates (based on id)': nr_of_duplicates_based_on_id,
            'Rows with duplicates': nr_of_duplicates,
            'Rows with missing annotations': nr_of_rows_with_missing_annotations,
            'Rows with invalid annotations': nr_of_rows_with_invalid_annotations,
            'Rows with disagreement': nr_of_rows_with_disagreement,
            'Final dataset size': len(df_stance_with_gt),
            'Value counts': df_stance_with_gt['ground_truth'].value_counts(),
        }
        for k, v in dataset_statistics.items():
            print(f"    {k}: {v}")

        df_stance_with_gt = self.order_df_columns(df_stance_with_gt)

        if return_statistics_for_plotting:
            return df_stance_with_gt, dataset_statistics
        else:
            return df_stance_with_gt


    def _get_mturk_file_path(self):
        return 'data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/mTurk_data/batch_results_stance_final.csv'


    def _normalize_mturk_annotation(self, x):
        mapping = {
            'POSITIVE': 'in favor of',
            'NEGATIVE': 'against',
            'NEUTRAL': 'neutral'
        }
        return mapping.get(x, str(x).strip().lower())


class Gilardi2023_Data1Task5Tweets_2020_2021(Gilardi2023):

    def __init__(self, data_directory='gilardi_et_al_pnas', config_fn='config_topic_tweets.yaml'):
        super().__init__(data_directory, config_fn)

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following tweet's topic is {previous_answer_placeholder} in the context of content-moderation discussions?

Output only a single number between 0 and 1, without any context or explanation.

Tweet: {text}

Probability:'''

    def get_prompts(self):
        # prompt copied from paper Appendix I: https://www.pnas.org/lookup/suppl/doi:10.1073/pnas.2305016120/-/DCSupplemental
        topic_prompt = "Tweets about content moderation may also discuss other related topics, such as:\n1. Section 230, which is a law in the United States that protects\nwebsites and other online platforms from being held legally responsible for the content posted by their users (SECTION 230).\n2. The decision by many social media platforms, such as Twitter and Facebook, to suspend Donald Trump’s account (TRUMP BAN).\n3. Requests directed to Twitter’s support account or help center (TWITTER SUPPORT).\n4. Social media platforms’ policies and practices, such as community guidelines or terms of service (PLATFORM POLICIES).\n5. Complaints about platform’s policy and practices in deplatforming and content moderation or suggestions to suspend particular accounts, or complaints about accounts being suspended or reported (COMPLAINTS).\n6. If a text is not about the SECTION 230, COMPLAINTS, TRUMP BAN, TWITTER SUPPORT, and PLATFORM POLICIES, then it should be classified in OTHER class (OTHER).\nFor each tweet in the sample, follow these instructions:\n1. Carefully read the text of the tweet, paying close attention to details.\n2. Please classify the following text according to topic (defined by function of the text, author’s purpose and form of the text). You can choose from the following classes: SECTION 230, TRUMP BAN, COMPLAINTS, TWITTER SUPPORT, PLATFORM POLICIES, and OTHER"
        

        gilardi_topic_prompt_paper = self.gilardi_content_moderation_prompt_paper + \
            topic_prompt

        tweet_instruction_prompt = "Here's the tweet I picked, please label it as 'Section 230', 'Trump Ban', 'Complaints', 'Twitter Support', 'Platform Policies', and 'Other' by answering with one word:\n{text}"


        # the following prompts are copied from Gilardi 2024 code: data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv
        # available for download at: https://osf.io/adkun/files/osfstorage
        # a=pd.read_csv("data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv")
        topic_specific_prompt_code_2024_zero = ["“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as of the topics described below:\n\nA: Section 230, which is a law in the United States that protects websites and other online platforms from being held legally responsible for the content posted by their users (SECTION 230). \nB: Trump ban, the decision by many social media platforms, such as Twitter and Facebook, to suspend Donald Trump’s account.\nC: Twitter Support, requests directed to Twitter’s support account or help center.\nD: Platform Policies, which is social media platforms’ policies and practices, such as community guidelines or terms of service.\nE: Complaint, which is general or personal complaints about platform’s policy and practices in deplatforming and content moderation or suggestions to suspend particular accounts, or complaints about accounts being suspended or reported.\nF: Other, if a text is not about the SECTION 230, COMPLAINTS, TRUMP BAN, TWITTER SUPPORT, and PLATFORM POLICIES, then it should be classified in OTHER class.", "Now, is the following tweet about SECTION 230, TRUMP BAN, COMPLAINTS, TWITTER SUPPORT, PLATFORM POLICIES, or OTHER?\n\n{text}"]


        topic_specific_prompt_code_2024_few = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as of the topics described below:\n\nA: Section 230, which is a law in the United States that protects websites and other online platforms from being held legally responsible for the content posted by their users (SECTION 230). \nB: Trump ban, the decision by many social media platforms, such as Twitter and Facebook, to suspend Donald Trump’s account.\nC: Twitter Support, requests directed to Twitter’s support account or help center.\nD: Platform Policies, which is social media platforms’ policies and practices, such as community guidelines or terms of service.\nE: Complaint, which is general or personal complaints about platform’s policy and practices in deplatforming and content moderation or suggestions to suspend particular accounts, or complaints about accounts being suspended or reported.\nF: Other, if a text is not about the SECTION 230, COMPLAINTS, TRUMP BAN, TWITTER SUPPORT, and PLATFORM POLICIES, then it should be classified in OTHER class.\n\nThe following tweets are some examples of the topics described above in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "@TangledUpInDead @soupmaned @jkosseff @jenniferm_q The referenced case has absolutely nothing to do with Section 230."
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "@LindseyGrahamSC The fact that you call it a “demand” is abhorrent! The American people deserve the support of their government and $2,000 pp is well overdue. The fact it will be held hostage unless Section 230 is addressed is horseshit! #humanroulette #americansaredyingtrumpkeepslying"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "@YourAnonCentral Hack Twitter and Ban Trump"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Russian-made #Telegram messenger shoots to top of US app charts, amid fears of wider social media crackdown following Trump ban — RT Russia &amp; Former Soviet Union https://t.co/3HGLCNNJ2T"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Twitter decided to ban Mike Lindell, who founded bedding company My Pillow, due to “repeated violations” of its civic integrity policy, a spokesperson said in a statement. The policy was implemented last September and is targeted at fighting disinformation https://t.co/lM2FeHuv3f"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": "Twitter To Introduce This New Feature To Curb The Spread Of Fake, Harmful Content. https://t.co/TqHeINqeQzhttps://t.co/SrwaCC5fdL So many #cherries to choose from this season! Choose your favourite or get them all on https://t.co/gAhHib40nQ. CherrySeason #FreshFruits #superplum #FarmFresh #VocalForLocal #StellaCherries #OrderOnline #HomeDelivery #BlackCherries #MerchantCherries https://t.co/WNpIDh72p3"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": "This post has been up 24 minutes and has 12 impressions @Twitter @TwitterSupport. I have 3800 followers. This number is about 95% off where it should be Why are you secretly shadow banning me? https://t.co/l1oF7lqraJ"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Using a criminal's picture insulting our history and culture this is where I should report this user to @Twitter @TwitterSupport for hateful content  and threatening #CyberSecurity https://t.co/KdIinpgMXf"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "You may have agreed or disagreed with JD from NY, but I don't think anyone should celebrate deplatforming"
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": "@dbongino I found out Twitter is shadowbanning me and I'm essentially a nobody.  How many other people with viewpoints they don't like and being placed behind a wall?"
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": "@propjoesays Twitter needs to issue a blanket ban on this kind of fearmongering. Anyone engaging in voter suppression or questioning the integrity of elections needs to have their account suspended."
            },
            {
                "role": "assistant",
                "content": "F"
            },
            {
                "role": "user",
                "content": "Lots of people have been noticing that recently. They are big time shadow banning. It’s so obvious. There are people I follow who tweet a couple times and hour and thought they were off twitter because I haven’t seen a tweet in months https://t.co/IQkZ2346VB"
            },
            {
                "role": "assistant",
                "content": "F"
            },
            {
                "role": "user",
                "content": "Now, is the following tweet about SECTION 230, TRUMP BAN, COMPLAINTS, TWITTER SUPPORT, PLATFORM POLICIES, or OTHER?\n\n{text}"
            }
        ]


        topic_specific_prompt_code_2024_cot = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as of the topics described below:\n\nA: Section 230, which is a law in the United States that protects websites and other online platforms from being held legally responsible for the content posted by their users (SECTION 230). \nB: Trump ban, the decision by many social media platforms, such as Twitter and Facebook, to suspend Donald Trump’s account.\nC: Twitter Support, requests directed to Twitter’s support account or help center.\nD: Platform Policies, which is social media platforms’ policies and practices, such as community guidelines or terms of service.\nE: Complaint, which is general or personal complaints about platform’s policy and practices in deplatforming and content moderation or suggestions to suspend particular accounts, or complaints about accounts being suspended or reported.\nF: Other, if a text is not about the SECTION 230, COMPLAINTS, TRUMP BAN, TWITTER SUPPORT, and PLATFORM POLICIES, then it should be classified in OTHER class.\n\nThe following tweets are some examples of the topics described above in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "@TangledUpInDead @soupmaned @jkosseff @jenniferm_q The referenced case has absolutely nothing to do with Section 230."
            },
            {
                "role": "assistant",
                "content": """Based on the provided text, it falls under the "SECTION 230" category. The text explicitly mentions Section 230 and states that the referenced case is unrelated to it."""
            },
            {
                "role": "user",
                "content": "@LindseyGrahamSC The fact that you call it a “demand” is abhorrent! The American people deserve the support of their government and $2,000 pp is well overdue. The fact it will be held hostage unless Section 230 is addressed is horseshit! #humanroulette #americansaredyingtrumpkeepslying"
            },
            {
                "role": "assistant",
                "content": """Based on the provided text, it falls under the "SECTION 230" category. The text mentions Section 230 and expresses frustration about it being used as leverage to address other issues, specifically relating to the demand for $2,000 per person."""
            },
            {
                "role": "user",
                "content": "@YourAnonCentral Hack Twitter and Ban Trump"
            },
            {
                "role": "assistant",
                "content": """Based on the given text, it would be classified as a statement related to "TRUMP BAN," specifically mentioning hacking Twitter and advocating for the banning of Trump."""
            },
            {
                "role": "user",
                "content": "Russian-made #Telegram messenger shoots to top of US app charts, amid fears of wider social media crackdown following Trump ban — RT Russia &amp; Former Soviet Union https://t.co/3HGLCNNJ2T"
            },
            {
                "role": "assistant",
                "content": """Based on the given text, it would be classified as a statement related to "TRUMP BAN." The text discusses fears of a wider social media crackdown following Trump's ban and mentions the rise of the Russian-made Telegram messenger."""
            },
            {
                "role": "user",
                "content": "Twitter decided to ban Mike Lindell, who founded bedding company My Pillow, due to “repeated violations” of its civic integrity policy, a spokesperson said in a statement. The policy was implemented last September and is targeted at fighting disinformation https://t.co/lM2FeHuv3f"
            },
            {
                "role": "assistant",
                "content": """This tweet appears to be more closely related to "PLATFORM POLICIES". The text discusses Twitter's decision to ban Mike Lindell based on violations of its civic integrity policy, which falls under the category of social media platforms' policies and practices."""
            },
            {
                "role": "user",
                "content": "Twitter To Introduce This New Feature To Curb The Spread Of Fake, Harmful Content. https://t.co/TqHeINqeQzhttps://t.co/SrwaCC5fdL So many #cherries to choose from this season! Choose your favourite or get them all on https://t.co/gAhHib40nQ. CherrySeason #FreshFruits #superplum #FarmFresh #VocalForLocal #StellaCherries #OrderOnline #HomeDelivery #BlackCherries #MerchantCherries https://t.co/WNpIDh72p3"
            },
            {
                "role": "assistant",
                "content": """This tweet should be classified as "PLATFORM POLICIES". The text mentions Twitter's introduction of a new feature to curb the spread of fake and harmful content. This aspect relates to social media platform policies aimed at content moderation."""
            },
            {
                "role": "user",
                "content": "This post has been up 24 minutes and has 12 impressions @Twitter @TwitterSupport. I have 3800 followers. This number is about 95% off where it should be Why are you secretly shadow banning me? https://t.co/l1oF7lqraJ"
            },
            {
                "role": "assistant",
                "content": """The text is about TWITTER SUPPORT. The author is addressing Twitter and TwitterSupport and complaining about a decrease in his/her number of followers, suggesting that he/she is being shadowbanned by Twitter. The use of the words "impressions" and "followers" suggests that the author is concerned with the platform's policies and practices regarding content moderation and deplatforming."""
            },
            {
                "role": "user",
                "content": "Using a criminal's picture insulting our history and culture this is where I should report this user to @Twitter @TwitterSupport for hateful content  and threatening #CyberSecurity https://t.co/KdIinpgMXf"
            },
            {
                "role": "assistant",
                "content": "The tweet is about TWITTER SUPPORT because it mentions reporting a user to @Twitter and @TwitterSupport for hateful content and threatening #CyberSecurity. This indicates a request for assistance from Twitter's support team to moderate and remove potentially harmful content shared on the platform."
            },
            {
                "role": "user",
                "content": "You may have agreed or disagreed with JD from NY, but I don't think anyone should celebrate deplatforming"
            },
            {
                "role": "assistant",
                "content": """This text falls under the "COMPLAINTS" class as it expresses dissatisfaction with deplatforming and suggests that it should not be celebrated, but it does not refer specifically to any social media platform, policy, or practice."""
            },
            {
                "role": "user",
                "content": "@dbongino I found out Twitter is shadowbanning me and I'm essentially a nobody.  How many other people with viewpoints they don't like and being placed behind a wall?"
            },
            {
                "role": "assistant",
                "content": """The given text is about COMPLAINTS. The text talks about an individual named "@dbongino" who has found out that Twitter has shadowbanned them, resulting in their tweets being placed behind a wall. The user expresses concern about how many other people with differing viewpoints might also be affected by Twitter's policies. This text is a complaint about Twitter's content moderation and deplatforming practices, and hence it falls under the COMPLAINTS category."""
            },
            {
                "role": "user",
                "content": "@propjoesays Twitter needs to issue a blanket ban on this kind of fearmongering. Anyone engaging in voter suppression or questioning the integrity of elections needs to have their account suspended."
            },
            {
                "role": "assistant",
                "content": "The text is not about the topics mentioned in the instructions, Instead, it is focused on a specific suggestion that Twitter should issue a blanket ban on certain types of content related to voter suppression and questioning the integrity of elections. As it does not fit into any of the given topic categories, it should be classified under OTHER."
            },
            {
                "role": "user",
                "content": "Lots of people have been noticing that recently. They are big time shadow banning. It’s so obvious. There are people I follow who tweet a couple times and hour and thought they were off twitter because I haven’t seen a tweet in months https://t.co/IQkZ2346VB"
            },
            {
                "role": "assistant",
                "content": """The tweet is more appropriately classified as "OTHER." It does not specifically mention Section 230, the Trump ban, Twitter support, or platform policies. Instead, it discusses shadow banning and the author's observation of decreased visibility of tweets from certain accounts. Since it does not align with any of the specified topics, "OTHER" is the most suitable classification."""
            },
            {
                "role": "user",
                "content": "Now, is the following tweet about SECTION 230, TRUMP BAN, COMPLAINTS, TWITTER SUPPORT, PLATFORM POLICIES, or OTHER?\n\n{text}"
            }
        ]

        # topic_specific_promot_paper_2024 = ""


        return [
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets_2020_2021 topic',
                    'compatible_output_mapping': ['gilardi political topic'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": gilardi_topic_prompt_paper
                        },
                        {
                            "role": "user",
                            "content": tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets_2020_2021 topic (no system prompt)',
                    'compatible_output_mapping': ['gilardi political topic'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": gilardi_topic_prompt_paper + "\n" + tweet_instruction_prompt

                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets_2020_2021 topic',
                    'compatible_output_mapping': [
                        # 'gilardi political topic', 
                        'gilardi political topic (extended with A to F)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": topic_specific_prompt_code_2024_zero[0]
                        },
                        {
                            "role": "user",
                            "content": topic_specific_prompt_code_2024_zero[1]
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets_2020_2021 topic (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi political topic', 
                        'gilardi political topic (extended with A to F)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": topic_specific_prompt_code_2024_zero[0] + "\n\n" + topic_specific_prompt_code_2024_zero[1]
                        },
                    ],
                },
                {
                    'description': '[original] [few-shot-detailed] gilardi 2024 (code) tweets_2020_2021 topic',
                    'compatible_output_mapping': [
                        # 'gilardi political topic', 
                        'gilardi political topic (extended with A to F)'],
                    'prompt_text': topic_specific_prompt_code_2024_few,
                },
                # {
                #     'description': 'gilardi 2024 (code) tweets_2020_2021 topic cot',
                #     'compatible_output_mapping': [
                #         # 'gilardi political topic', 
                #         'gilardi political topic (extended with A to F)'],
                #     'prompt_text': topic_specific_prompt_code_2024_cot,
                # },
            ]

    def get_all_output_mappings(self):

        return {
            'gilardi political topic': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.topic_political_output_mapping},
            },
            'gilardi political topic (extended with A to F)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.topic_political_output_mapping_extended_A_to_F},
            },
        }

    def get_dataset_name(self):
        """Get the name of the dataset."""
        return "Gilardi2023_Data1Task5Tweets_2020_2021"

    def load_full_dataset(self, return_statistics_for_plotting=False, return_dataset_used_by_gilardi=False):
        """Load and prepare the Gilardi2023_Data1Task5Tweets_2020_2021 dataset."""

        df_raw = pd.read_excel("data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/research_assistants_data/frames_task/tweets_annotation_data_ra_completed.xlsx")
        df = df_raw.dropna(subset=['topic_name_fabio', 'topic_name_paula'], how='all')

        # Filter to only relevant tweets and process for topic task
        df_relevant = df[df['relevant'] == 1]

        # The following aggregates the two complaint classes before checking agreement
        # gilardi et al. made the mistake to filter out the rows where both annotators agree that it's complaint but disagreed on the type of complaint (personal/general)
        # df_relevant['topic_name_fabio'] = df_relevant['topic_name_fabio'].apply(
        #     lambda x: 'complaint' if x in ['personal complaint', 'general complaint'] else x
        # )
        # df_relevant['topic_name_paula'] = df_relevant['topic_name_paula'].apply(
        #     lambda x: 'complaint' if x in ['personal complaint', 'general complaint'] else x
        # )

        # Now drop duplicates
        df_topic_without_duplicates = df_relevant.drop_duplicates(subset=['status_id'], keep='first')
        nr_of_duplicates_based_on_id = len(df_relevant) - len(df_topic_without_duplicates)
        if not return_dataset_used_by_gilardi:
            df_topic_without_duplicates = df_topic_without_duplicates.drop_duplicates(subset=['text'], keep='first')
        print(f"\nRows after dropping duplicates: {len(df_topic_without_duplicates)}")

        df_topic = df_topic_without_duplicates.dropna(subset=['topic_name_fabio', 'topic_name_paula'])

        # calculate trained annotators agreement
        trained_annotators_agreement = self.intercoder_agreement_percentage(df_topic, ['topic_name_fabio', 'topic_name_paula'])
        trained_annotators_agreement_krippendorff = self.intercoder_agreement_krippendorff(df_topic, ['topic_name_fabio', 'topic_name_paula'])
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement: {trained_annotators_agreement:.4f}')
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement_krippendorff: {trained_annotators_agreement_krippendorff:.4f}')

        # Keep only rows where both annotators agree
        df_topic_with_gt = df_topic[
            (df_topic['topic_name_fabio'] == df_topic['topic_name_paula'])
        ]
        df_topic_with_gt['ground_truth'] = copy.deepcopy(
            df_topic_with_gt['topic_name_fabio']
        )
        df_topic_with_gt_final = copy.deepcopy(df_topic_with_gt)
        # aggregate persona complaint and general complaint into complaint only
        df_topic_with_gt_final['ground_truth'] = df_topic_with_gt_final['ground_truth'].apply(
            lambda x: 'complaint' if x in ['personal complaint', 'general complaint'] else x
        )

        nr_of_duplicates = len(df_relevant) - len(df_topic_without_duplicates)
        nr_of_rows_with_missing_annotations = len(df_topic_without_duplicates) - len(df_topic)
        nr_of_rows_with_disagreement = len(df_topic) - len(df_topic_with_gt)

        dataset_statistics = {
            'Dataset name': self.get_dataset_name(),
            'Total rows (raw)': len(df_raw),
            'Total rows': len(df),
            'Total relevant rows': len(df_relevant),
            'Rows without duplicate values': len(df_topic_without_duplicates),
            'Rows without GT NaN values': len(df_topic),
            # 'Rows without both values (problem and solution)': len(df_topic),
            'Trained annotators agreement': trained_annotators_agreement,
            'Trained annotators agreement (krippendorff)': trained_annotators_agreement_krippendorff,
            'Rows with duplicates (based on id)': nr_of_duplicates_based_on_id,
            'Rows with duplicates': nr_of_duplicates,
            'Rows with missing annotations': nr_of_rows_with_missing_annotations,
            'Rows with disagreement': nr_of_rows_with_disagreement,
            'Final dataset size': len(df_topic_with_gt),
            'Value counts (before class aggregation)': df_topic_with_gt['ground_truth'].value_counts(),
            'Value counts': df_topic_with_gt_final['ground_truth'].value_counts(),
        }
        for k, v in dataset_statistics.items():
            print(f"    {k}: {v}")

        df_topic_with_gt_final = self.order_df_columns(df_topic_with_gt_final)

        if return_statistics_for_plotting:
            return df_topic_with_gt_final, dataset_statistics
        else:
            return df_topic_with_gt_final


    def _get_mturk_file_path(self):
        return 'data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/mTurk_data/batch_results_topics_final.csv'


class Gilardi2023_Data1Task6Tweets_2020_2021(Gilardi2023):
    # throw not implemented error
    def __init__(self, config=None, debug=False, results_folder=None):
        raise NotImplementedError("This task is not implemented yet as it has only been used in the 2024 paper but not in the 2023 PNAS paper.")
    



class Gilardi2023_Data2Task1_Tweets2023Relevance(Gilardi2023):
    """Implementation for Gilardi 2023 Relevance Task."""

    def __init__(self, data_directory='gilardi_et_al_pnas', config_fn='config_relevance_tweets23.yaml'):
        super().__init__(data_directory, config_fn)

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following tweet is {previous_answer_placeholder} to the topic of content moderation?

Output only a single number between 0 and 1, without any context or explanation.

Tweet: {text}

Probability:'''

    def get_prompts(self):
        # the following prompts are copied from Gilardi 2024 code: data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv
        # available for download at: https://osf.io/adkun/files/osfstorage
        # a=pd.read_csv("data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv")
        relevance_specific_prompt_code_2024_zero = ["“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Text is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like flagging, shadowbanning, or account suspension.\n\nB: Text is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.", "Now, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"]
        
        ban_prompt_rel1 = "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be banned or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Text is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like banning, flagging, shadowbanning, or account suspension.\n\nB: Text is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nNow, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
        ban_prompt_rel2 = "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be banned or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Text is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like banning, flagging, shadowbanning, or account suspension.\n\nB: Text is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nNotice that whenever a tweet mentions 'ban' or 'banning', it is typically RELEVANT to content moderation.\n\nNow, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
        ban_prompt_irrel1 = "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Text is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like flagging, shadowbanning, or account suspension.\n\nB: Text is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nNotice that whenever a tweet mentions 'ban' or 'banning', it is typically IRRELEVANT to content moderation.\n\nNow, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
        ban_prompt_irrel2 = "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Text is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like flagging, or account suspension.\n\nB: Text is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nNotice that whenever a tweet mentions 'ban' or 'banning', it is typically IRRELEVANT to content moderation.\n\nNow, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
        
        relevance_specific_prompt_code_2024_few = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Text is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like flagging, shadowbanning, or account suspension.\n\nB: Text is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nThe tweets below are some examples of 'RELEVANT' and 'IRRELEVANT' texts to content moderation in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "@GillianBateman5 @ankertilbrewer @BreesAnna She blocked me on my old account but added me on this one for some reason 🤣"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "This is how will #SriLanka debt restructuring look like? accordingly to VERITÉ RESEARCH SRI LANKA POLICY GROUP @VeriteResearch @CBSL @StanChart\nhttps://t.co/adjf5cKzxnhttps://t.co/Er1nP9a4jh"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Dear @Twitter @TwitterMENA @TwitterSupport @SalmaMMMT account has been suspended 1 day ago And without any reason ☹️💔"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Finally could restore my account. 🤬 Why twitter even require phone number in the first place? Aren't we supposed to have a choice not to show it thus not being banned after, if twitter bots decide you're one of them? #twitter #ElonMusk #FreedomInternet"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Now, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
            }
        ]

        relevance_specific_prompt_code_2024_cot = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Text is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like flagging, shadowbanning, or account suspension.\n\nB: Text is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nThe tweets below are some examples of 'RELEVANT' and 'IRRELEVANT' texts to content moderation in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "@GillianBateman5 @ankertilbrewer @BreesAnna She blocked me on my old account but added me on this one for some reason 🤣"
            },
            {
                "role": "assistant",
                "content": "Based on the criteria provided, the text provided is IRRELEVANT to content moderation. It does not discuss social media platforms' content moderation rules and practices, censorship, governments' regulation of online content moderation, or any forms of content moderation such as flagging, shadowbanning, or account suspension. Instead, it appears to be a casual conversation or observation about someone blocking and adding people on social media."
            },
            {
                "role": "user",
                "content": "This is how will #SriLanka debt restructuring look like? accordingly to VERITÉ RESEARCH SRI LANKA POLICY GROUP @VeriteResearch @CBSL @StanChart\nhttps://t.co/adjf5cKzxnhttps://t.co/Er1nP9a4jh"
            },
            {
                "role": "assistant",
                "content": "Based on the given criteria, the text provided is IRRELEVANT to content moderation. It does not discuss social media platforms' content moderation rules and practices, censorship, governments' regulation of online content moderation, or any mild forms of content moderation. Instead, the text appears to be about the debt restructuring in Sri Lanka, which is unrelated to content moderation on social media sites."
            },
            {
                "role": "user",
                "content": "Dear @Twitter @TwitterMENA @TwitterSupport @SalmaMMMT account has been suspended 1 day ago And without any reason ☹️💔"
            },
            {
                "role": "assistant",
                "content": "Based on the given criteria, the text provided is RELEVANT to content moderation. It mentions the suspension of an account on Twitter (@SalmaMMMT) without any given reason. This relates to content moderation as it touches upon the action of account suspension, which is a mild form of content moderation employed by social media platforms."
            },
            {
                "role": "user",
                "content": "Finally could restore my account. 🤬 Why twitter even require phone number in the first place? Aren't we supposed to have a choice not to show it thus not being banned after, if twitter bots decide you're one of them? #twitter #ElonMusk #FreedomInternet"
            },
            {
                "role": "assistant",
                "content": "The text provided can be considered RELEVANT to content moderation. Although it primarily discusses personal account issues and frustrations with Twitter's phone number requirement, it also raises concerns about being banned based on the decisions of Twitter bots. This alludes to the practice of content moderation, where automated systems are often involved in flagging or suspending accounts based on certain criteria.\n\nNow, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
            },
            {
                "role": "user",
                "content": "Now, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
            }
        ]

        # relevance_specific_promot_paper_2024 = ""

        return [
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (code) tweets2023 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with 0 and 1)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": self.content_moderation_relevance_specific_prompt_code
                        },
                        {
                            "role": "user",
                            "content": " here's the tweet I picked, please label it as 'Relevant' or 'Irrelevant' by answering with one word:\n{text}"
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (code) tweets2023 relevance (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with 0 and 1)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": self.content_moderation_relevance_specific_prompt_code + " here's the tweet I picked, please label it as 'Relevant' or 'Irrelevant' by answering with one word:\n{text}"
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets2023 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with 0 and 1)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": self.content_moderation_relevance_specific_prompt_paper
                        },
                        {
                            "role": "user",
                            "content": "Here's the tweet I picked, please label it as 'Relevant' or 'Irrelevant' by answering with one word:\n{text}"
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets2023 relevance (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with 0 and 1)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": self.content_moderation_relevance_specific_prompt_paper + " Here's the tweet I picked, please label it as 'Relevant' or 'Irrelevant' by answering with one word:\n{text}"
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets2023 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with A and B)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": relevance_specific_prompt_code_2024_zero[0]
                        },
                        {
                            "role": "user",
                            "content": relevance_specific_prompt_code_2024_zero[1]
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets2023 relevance (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with A and B)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": relevance_specific_prompt_code_2024_zero[0] + "\n\n" + relevance_specific_prompt_code_2024_zero[1]
                        },
                    ],
                },
                {
                    'description': '[original] [few-shot-detailed] gilardi 2024 (code) tweets2023 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with A and B)'],
                    'prompt_text': relevance_specific_prompt_code_2024_few,
                },
            ]

    def get_all_output_mappings(self):

        return {
            'gilardi relevance': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.relevance_output_mapping},
            },
            'gilardi relevance (extended with 0 and 1)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.relevance_output_mapping_extended_0_and_1},
            },
            'gilardi relevance (extended with A and B)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.relevance_output_mapping_extended_A_and_B},
            },
        }

    def get_dataset_name(self):
        """Get the name of the dataset."""
        return "gilardi_tweet23_relevance"

    def load_full_dataset(self, return_statistics_for_plotting=False, return_dataset_used_by_gilardi=False):
        """Load and prepare the Gilardi 2023 Relevance dataset."""
        # Load data from Excel files
        df_raw = pd.read_excel(
            'data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/research_assistants_data/problem_solution_task/annotation_data_tweets23_ra_completed.xlsx')

        df = df_raw.dropna(subset=['relevant_fabio', 'relevant_paula'], how='all')

        # Process for relevance task
        df_relevance = df.dropna(
            subset=['relevant_fabio', 'relevant_paula'])

        df_relevance_without_duplicates = df_relevance.drop_duplicates(subset=['status_id'], keep='first')
        nr_of_duplicates_based_on_id = len(df_relevance) - len(df_relevance_without_duplicates)
        if not return_dataset_used_by_gilardi:
            df_relevance_without_duplicates = df_relevance_without_duplicates.drop_duplicates(subset=['text'], keep='first')
        print(f"\nRows after dropping duplicates: {len(df_relevance_without_duplicates)}")

        # calculate trained annotators agreement
        trained_annotators_agreement = self.intercoder_agreement_percentage(df_relevance_without_duplicates, ['relevant_fabio', 'relevant_paula'])
        trained_annotators_agreement_krippendorff = self.intercoder_agreement_krippendorff(df_relevance_without_duplicates, ['relevant_fabio', 'relevant_paula'])
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement: {trained_annotators_agreement:.4f}')
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement_krippendorff: {trained_annotators_agreement_krippendorff:.4f}')

        df_relevance_with_gt = df_relevance_without_duplicates[
            df_relevance_without_duplicates['relevant_fabio'] == df_relevance_without_duplicates['relevant_paula']
        ]
        df_relevance_with_gt['ground_truth'] = copy.deepcopy(
            df_relevance_with_gt['relevant_fabio']
        )

        nr_of_duplicates = len(df) - len(df_relevance)
        nr_of_rows_with_missing_annotations = len(df_relevance) - len(df_relevance_without_duplicates)
        nr_of_rows_with_disagreement = len(df_relevance_without_duplicates) - len(df_relevance_with_gt)

        dataset_statistics = {
            'Dataset name': self.get_dataset_name(),
            'Total rows': len(df),
            'Total rows (raw)': len(df_raw),
            'Rows without GT NaN values': len(df_relevance),
            'Rows without duplicate values': len(df_relevance_without_duplicates),
            'Trained annotators agreement': trained_annotators_agreement,
            'Trained annotators agreement (krippendorff)': trained_annotators_agreement_krippendorff,
            'Rows with duplicates (based on id)': nr_of_duplicates_based_on_id,
            'Rows with duplicates': nr_of_duplicates,
            'Rows with missing annotations': nr_of_rows_with_missing_annotations,
            'Rows with disagreement': nr_of_rows_with_disagreement,
            'Final dataset size': len(df_relevance_with_gt),
            # 'Rows which all trained annotators agree [relevant=1]': sum(df_relevance_with_gt['ground_truth']==1),
            'Value counts': df_relevance_with_gt['ground_truth'].value_counts(),
        }
        for k, v in dataset_statistics.items():
            print(f"    {k}: {v}")

        df_relevance_with_gt = self.order_df_columns(df_relevance_with_gt)
        df_relevance_with_gt['ground_truth'] = df_relevance_with_gt['ground_truth'].map({0: 'irrelevant', 1: 'relevant'})

        if return_statistics_for_plotting:
            return df_relevance_with_gt, dataset_statistics
        else:
            return df_relevance_with_gt


    def _get_mturk_file_path(self):
        return 'data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/mTurk_data/batch_results_tweets23_relevance_final.csv'

    def _get_mturk_delimiter(self):
        return ';'

class Gilardi2023_Data2Task2_Tweets2023Frame(Gilardi2023):
    """Implementation for Gilardi 2023 Problem/Solution Frame Task."""

    def __init__(self, data_directory='gilardi_et_al_pnas', config_fn='config_framesI_tweets23.yaml'):
        super().__init__(data_directory, config_fn)

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following tweet describes content moderation as a {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Tweet: {text}

Probability:'''

    def get_prompts(self):
        # prompt copied from paper Appendix E: https://www.pnas.org/lookup/suppl/doi:10.1073/pnas.2305016120/-/DCSupplemental
        frame_problem_solution_prompt = "\nContent moderation can be seen from two different perspectives:\n• Content moderation can be seen as a PROBLEM; for example, as a restriction of free speech\n• Content moderation can be seen as a SOLUTION; for example, as a protection from harmful speech\nFor each tweet in the sample, follow these instructions:\n1. Carefully read the text of the tweet, paying close attention to details.\n2. Classify the tweet as describing content moderation as a problem, as a solution, or neither.\nTweets should be classified as describing content moderation as a PROBLEM if they emphasize negative effects of content moderation, such as restrictions to free speech, or the biases that can emerge from decisions regarding what users are allowed to post.\nTweets should be classified as describing content moderation as a SOLUTION if they emphasize positive effects of content moderation, such as protecting users from various kinds of harmful content, including hate speech, misinformation, illegal adult content, or spam.\nTweets should be classified as describing content moderation as NEUTRAL if they do not emphasize possible negative or positive effects of content moderation, for example if they simply report on the content moderation activity of social media platforms without linking them to potential advantages or disadvantages for users or stakeholders."

        gilardi_frame_prompt_code = self.gilardi_content_moderation_prompt_code + \
            frame_problem_solution_prompt
        gilardi_frame_prompt_paper = self.gilardi_content_moderation_prompt_paper + \
            frame_problem_solution_prompt

        tweet_instruction_prompt = "Here's the tweet I picked, please label it as 'Problem', 'Solution', or 'Neutral' by answering with one word:\n{text}"


        # the following prompts are copied from Gilardi 2024 code: data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv
        # available for download at: https://osf.io/adkun/files/osfstorage
        # a=pd.read_csv("data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv")
        frame_specific_prompt_code_2024_zero = ["“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as describing content moderation as a problem, as a solution, or neither:\n\nA: Text describes content moderation as a PROBLEM if they emphasize negative effects of it, such as restrictions to free speech, censorship, or the biases that can emerge from decisions regarding what users are allowed to post.\n\nB: Text describes content moderation as a SOLUTION if they emphasize positive effects of it, such as protecting users from harmful content such as hate speech, misinformation, illegal adult content, or spam. \n\nC: Text describes content moderation as NEUTRAL if they do not emphasize negative or positive effects of content moderation. For example if they simply report on the content moderation activity of social media platforms without linking them to potential advantages or disadvantages for users or stakeholders.", "Now, is the following tweet describing content moderation as a PROBLEM, as a SOLUTION, or NEUTRAL?\n\n{text}"]


        frame_specific_prompt_code_2024_few = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as describing content moderation as a problem, as a solution, or neither:\n\nA: Text describes content moderation as a PROBLEM if they emphasize negative effects of it, such as restrictions to free speech, censorship, or the biases that can emerge from decisions regarding what users are allowed to post.\n\nB: Text describes content moderation as a SOLUTION if they emphasize positive effects of it, such as protecting users from harmful content such as hate speech, misinformation, illegal adult content, or spam. \n\nC: Text describes content moderation as NEUTRAL if they do not emphasize negative or positive effects of content moderation. For example if they simply report on the content moderation activity of social media platforms without linking them to potential advantages or disadvantages for users or stakeholders.\n\nThe following texts are some examples of 'PROBLEM', 'SOLUTION' or 'NEUTRAL' texts about content moderation in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "@amerxmirza I have reported him to make sure his account gets suspended. Disgusting and derogation remarks shouldn't be tolerated. Please report. Thanks"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Channel link https://t.co/9ms7ukCI84 please block this channel this content very harmful https://t.co/zeJ512JPPx"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Finally could restore my account. 🤬 Why twitter even require phone number in the first place? Aren't we supposed to have a choice not to show it thus not being banned after, if twitter bots decide you're one of them? \n#twitter #ElonMusk #FreedomInternet"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Of all the issues we’re facing now, why’s social media regulation the goal for this people? Freedom of speech is a basic human right and any attempt to take it from Nigerians should be rejected #RejectNITDAbill #Notosocialmediaregulation"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "@BotGerad @R6Jena @zxdd account suspended xd"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "@elonmusk @ellagirwin  @esthercrawford there seems to have been some mistake, renowned internet love-monger @nickjfuentes was suspended from Twitter. Please do the right thing and re-instate his account for the sake of humanity 🙏😔"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Now, is the following tweet describing content moderation as a PROBLEM, as a SOLUTION, or NEUTRAL?\n\n{text}"
            }
        ]


        
        frame_specific_prompt_code_2024_cot = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as describing content moderation as a problem, as a solution, or neither:\n\nA: Text describes content moderation as a PROBLEM if they emphasize negative effects of it, such as restrictions to free speech, censorship, or the biases that can emerge from decisions regarding what users are allowed to post.\n\nB: Text describes content moderation as a SOLUTION if they emphasize positive effects of it, such as protecting users from harmful content such as hate speech, misinformation, illegal adult content, or spam. \n\nC: Text describes content moderation as NEUTRAL if they do not emphasize negative or positive effects of content moderation. For example if they simply report on the content moderation activity of social media platforms without linking them to potential advantages or disadvantages for users or stakeholders.\n\nThe following texts are some examples of 'PROBLEM', 'SOLUTION' or 'NEUTRAL' texts about content moderation in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "@amerxmirza I have reported him to make sure his account gets suspended. Disgusting and derogation remarks shouldn't be tolerated. Please report. Thanks"
            },
            {
                "role": "assistant",
                "content": """The given text can be classified as describing content moderation as a SOLUTION. It emphasizes the positive effect of content moderation by advocating for reporting and suspending the account of a user who made "disgusting and derogatory remarks." The text acknowledges the need to not tolerate such behavior and encourages others to report it, highlighting the protective role of content moderation in addressing harmful content."""
            },
            {
                "role": "user",
                "content": "Channel link https://t.co/9ms7ukCI84 please block this channel this content very harmful https://t.co/zeJ512JPPx"
            },
            {
                "role": "assistant",
                "content": "The given text can be classified as describing content moderation as a SOLUTION. It emphasizes the positive effect of content moderation by requesting the blocking of a specific channel due to its harmful content. The text recognizes the need to take action and protect users from potentially harmful content, aligning with the protective role of content moderation in addressing harmful material."
            },
            {
                "role": "user",
                "content": "Finally could restore my account. 🤬 Why twitter even require phone number in the first place? Aren't we supposed to have a choice not to show it thus not being banned after, if twitter bots decide you're one of them? \n#twitter #ElonMusk #FreedomInternet"
            },
            {
                "role": "assistant",
                "content": "The given text can be classified as describing content moderation as a PROBLEM. It criticizes the requirement of providing a phone number by Twitter and questions the lack of choice in not showing it. The text implies frustration with the potential consequences of content moderation decisions made by Twitter bots, suggesting concerns about restrictions to freedom and potential biases in the platform's moderation practices."
            },
            {
                "role": "user",
                "content": "Of all the issues we’re facing now, why’s social media regulation the goal for this people? Freedom of speech is a basic human right and any attempt to take it from Nigerians should be rejected #RejectNITDAbill #Notosocialmediaregulation"
            },
            {
                "role": "assistant",
                "content": "The given text can be classified as describing content moderation as a PROBLEM. It emphasizes the negative effects of social media regulation, specifically mentioning the potential infringement on freedom of speech as a basic human right. The text opposes any attempt to take away this right from Nigerians, suggesting concerns about restrictions and censorship associated with content moderation."
            },
            {
                "role": "user",
                "content": "@BotGerad @R6Jena @zxdd account suspended xd"
            },
            {
                "role": "assistant",
                "content": "Based on the given text, it does not explicitly emphasize the negative or positive effects of content moderation. Instead, it simply reports the suspension of an account without further elaboration. Therefore, the text can be classified as describing content moderation as NEUTRAL, as it does not emphasize either the problems or the solutions associated with content moderation."
            },
            {
                "role": "user",
                "content": "@elonmusk @ellagirwin  @esthercrawford there seems to have been some mistake, renowned internet love-monger @nickjfuentes was suspended from Twitter. Please do the right thing and re-instate his account for the sake of humanity 🙏😔"
            },
            {
                "role": "assistant",
                "content": "Based on the given text, it is possible to interpret it differently. While the text does request the reinstatement of a suspended account, it does not explicitly mention any negative effects or problems related to content moderation. Therefore, an alternative classification could be that the text describes content moderation as NEUTRAL since it does not emphasize negative or positive effects. It simply requests the reinstatement of a specific account without further elaboration on the broader implications of content moderation."
            },
            {
                "role": "user",
                "content": "Now, is the following tweet describing content moderation as a PROBLEM, as a SOLUTION, or NEUTRAL?\n\n{text}"
            }
        ]

        # frame_specific_promot_paper_2024 = ""


        return [
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (code) tweets2023 frame',
                    'compatible_output_mapping': ['gilardi frame'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": gilardi_frame_prompt_code
                        },
                        {
                            "role": "user",
                            "content": tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (code) tweets2023 frame (no system prompt)',
                    'compatible_output_mapping': ['gilardi frame'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": gilardi_frame_prompt_code + "\n" + tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets2023 frame',
                    'compatible_output_mapping': ['gilardi frame'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": gilardi_frame_prompt_paper
                        },
                        {
                            "role": "user",
                            "content": tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets2023 frame (no system prompt)',
                    'compatible_output_mapping': ['gilardi frame'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": gilardi_frame_prompt_paper + "\n" + tweet_instruction_prompt

                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets2023 frame',
                    'compatible_output_mapping': [
                        'gilardi frame (extended with A and B and C)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": frame_specific_prompt_code_2024_zero[0]
                        },
                        {
                            "role": "user",
                            "content": frame_specific_prompt_code_2024_zero[1]
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets2023 frame (no system prompt)',
                    'compatible_output_mapping': [
                        'gilardi frame (extended with A and B and C)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": frame_specific_prompt_code_2024_zero[0] + "\n\n" + frame_specific_prompt_code_2024_zero[1]
                        },
                    ],
                },
                {
                    'description': '[original] [few-shot-detailed] gilardi 2024 (code) tweets2023 frame',
                    'compatible_output_mapping': [
                        'gilardi frame (extended with A and B and C)'],
                    'prompt_text': frame_specific_prompt_code_2024_few,
                },
            ]

    def get_all_output_mappings(self):

        return {
            'gilardi frame': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.frame_output_mapping},
            },
            'gilardi frame (extended with A and B and C)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.frame_output_mapping_extended_A_and_B_and_C},
            },
        }

    def get_dataset_name(self):
        """Get the name of the dataset."""
        return "gilardi_tweet23_frame"

    def load_full_dataset(self, return_statistics_for_plotting=False, return_dataset_used_by_gilardi=False):
        """Load and prepare the Gilardi 2023 Frame dataset."""
        # Load data from Excel files
        df_raw = pd.read_excel(
            'data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/research_assistants_data/problem_solution_task/annotation_data_tweets23_ra_completed.xlsx')

        df = df_raw.dropna(subset=['relevant_fabio', 'relevant_paula'], how='all')

        # Filter to only relevant tweets and process for frame task
        df_relevant = df[
            (df['relevant_fabio'] == 1) &
            (df['relevant_paula'] == 1)
        ]

        # Now drop duplicates
        df_frame_without_duplicates = df_relevant.drop_duplicates(subset=['status_id'], keep='first')
        nr_of_duplicates_based_on_id = len(df_relevant) - len(df_frame_without_duplicates)
        if not return_dataset_used_by_gilardi:
            df_frame_without_duplicates = df_frame_without_duplicates.drop_duplicates(subset=['text'], keep='first')
        print(f"\nRows after dropping duplicates: {len(df_frame_without_duplicates)}")


        df_frame = df_frame_without_duplicates.dropna(
            subset=['problem_frame_fabio', 'problem_frame_paula',
                    'solution_frame_fabio', 'solution_frame_paula']
        )

        df_frame_not_both = df_frame[(
            ~(
                ((df_frame['problem_frame_fabio'] == 1) & (df_frame['solution_frame_fabio'] == 1)) |
                ((df_frame['problem_frame_paula'] == 1) & (df_frame['solution_frame_paula'] == 1))
            )
        )]

        # Apply frame classification functions
        df_frame_not_both['ground_truth_fabio'] = df_frame_not_both.apply(
            lambda row: self.get_frame_gt(row, 'fabio'), axis=1
        )
        df_frame_not_both['ground_truth_paula'] = df_frame_not_both.apply(
            lambda row: self.get_frame_gt(row, 'paula'), axis=1
        )

        # calculate trained annotators agreement
        trained_annotators_agreement = self.intercoder_agreement_percentage(df_frame_not_both, ['ground_truth_fabio', 'ground_truth_paula'])
        trained_annotators_agreement_krippendorff = self.intercoder_agreement_krippendorff(df_frame_not_both, ['ground_truth_fabio', 'ground_truth_paula'])
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement: {trained_annotators_agreement:.4f}')
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement_krippendorff: {trained_annotators_agreement_krippendorff:.4f}')

        # Keep only rows where both annotators agree
        df_frame_with_gt = df_frame_not_both[
            (df_frame_not_both['ground_truth_fabio'] ==
             df_frame_not_both['ground_truth_paula'])
        ]
        df_frame_with_gt['ground_truth'] = copy.deepcopy(
            df_frame_with_gt['ground_truth_fabio']
        )

        nr_of_duplicates = len(df_relevant) - len(df_frame_without_duplicates)
        nr_of_rows_with_missing_annotations = len(df_frame_without_duplicates) - len(df_frame)
        nr_of_rows_with_invalid_annotations = len(df_frame) - len(df_frame_not_both)
        nr_of_rows_with_disagreement = len(df_frame_not_both) - len(df_frame_with_gt)

        dataset_statistics = {
            'Dataset name': self.get_dataset_name(),
            'Total rows (raw)': len(df_raw),
            'Total rows': len(df),
            'Total relevant rows': len(df_relevant),
            'Rows without duplicate values': len(df_frame_without_duplicates),
            'Rows without GT NaN values': len(df_frame),
            'Rows without both values (problem and solution)': len(df_frame_not_both),
            'Trained annotators agreement': trained_annotators_agreement,
            'Trained annotators agreement (krippendorff)': trained_annotators_agreement_krippendorff,
            'Rows with duplicates (based on id)': nr_of_duplicates_based_on_id,
            'Rows with duplicates': nr_of_duplicates,
            'Rows with missing annotations': nr_of_rows_with_missing_annotations,
            'Rows with invalid annotations': nr_of_rows_with_invalid_annotations,
            'Rows with disagreement': nr_of_rows_with_disagreement,
            'Final dataset size': len(df_frame_with_gt),
            'Value counts': df_frame_with_gt['ground_truth'].value_counts(),
        }
        for k, v in dataset_statistics.items():
            print(f"    {k}: {v}")

        df_frame_with_gt = self.order_df_columns(df_frame_with_gt)

        if return_statistics_for_plotting:
            return df_frame_with_gt, dataset_statistics
        else:
            return df_frame_with_gt


    def _get_mturk_file_path(self):
        return 'data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/mTurk_data/batch_results_tweets23_problem_solution_final.csv'


class Gilardi2023_Data3Task1_CongressTweets_2017_2022_Relevance(Gilardi2023):
    """Implementation for Gilardi 2023 Relevance Task."""

    def __init__(self, data_directory='gilardi_et_al_pnas', config_fn='config_relevance_tweets17.yaml'):
        super().__init__(data_directory, config_fn)

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following tweet is {previous_answer_placeholder} with respect to containing political content?

Output only a single number between 0 and 1, without any context or explanation.

Tweet: {text}

Probability:'''

    def get_prompts(self):
        # Define relevance-specific prompt details
        # This part of the prompt is specific to the relevance task

        # define prompt details

        # 2nd part of prompt copied from paper Appendix D: https://www.pnas.org/lookup/suppl/doi:10.1073/pnas.2305016120/-/DCSupplemental
        relevance_specific_prompt_paper = self.gilardi_political_content_prompt_paper + \
            "\nFor each tweet in the sample, follow these instructions:\n1. Carefully read the text of the tweet, paying close attention to details.\n2. Classify the tweet as either relevant (1) or irrelevant (0)\nTweets should be coded as RELEVANT if they include POLITICAL CONTENT, as defined above. Tweets should be coded as IRRELEVANT if they do NOT include POLITICAL CONTENT, as defined above."

        # the following prompts are copied from Gilardi 2024 code: data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv
        # available for download at: https://osf.io/adkun/files/osfstorage
        # a=pd.read_csv("data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv")
        relevance_specific_prompt_code_2024_zero = ["“Political content” refers to a text that pertains to politics or government policies at the local, national, or international level. This can include political figures, events, or issues, as well as text that uses political language or hashtags. \n\nI will ask you to classify a tweet as relevant or irrelevant to the political content:\n\nA: Text is RELEVANT if it uses political keywords or hashtags, mentions political figures or events, discusses policy issues such as immigration, abortion, foreign policy, health care, tax, or police shootings, or includes a link to well-known news outlets or other political sources such as think tanks, political pundits or journalists, the White House, or the US Congress.\n\nB: Text is IRRELEVANT if it does not fit the criteria above\n", "Now, is the following tweet RELEVANT or IRRELEVANT to political content?\n\n{text}"]
        
        
        relevance_specific_prompt_code_2024_few = [
            {
                "role": "system",
                "content": "“Political content” refers to a text that pertains to politics or government policies at the local, national, or international level. This can include political figures, events, or issues, as well as text that uses political language or hashtags. \n\nI will ask you to classify a tweet as relevant or irrelevant to the political content:\n\nA: Text is RELEVANT if it uses political keywords or hashtags, mentions political figures or events, discusses policy issues such as immigration, abortion, foreign policy, health care, tax, or police shootings, or includes a link to well-known news outlets or other political sources such as think tanks, political pundits or journalists, the White House, or the US Congress.\n\nB: Text is IRRELEVANT if it does not fit the criteria above\n\nThe following tweets are some examples of RELEVANT or IRRELEVANT texts to political contents in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "What inspires me: being a warrior for human dignity, human potential. https://t.co/k6NXxcThaD"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "I hope this time brings you the opportunity to reflect, to #hope, and to know that you're loved by God.\n#MerryChristmas, #GA09! https://t.co/cOfwb9TxWa"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Pres Trump wants another 1 percent for the richest 1 percent. 100 percent irresponsible. https://t.co/CawU4SpUTo"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "I spoke on the floor of the House this morning to honor the life of longtime Syracuse Police Capt Richard Walsh --&gt; https://t.co/6CdNF5jcLc"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Now, is the following tweet RELEVANT or IRRELEVANT to political content?\n\n{text}"
            }
        ]
        
        relevance_specific_prompt_code_2024_cot = [
            {
                "role": "system",
                "content": "“Political content” refers to a text that pertains to politics or government policies at the local, national, or international level. This can include political figures, events, or issues, as well as text that uses political language or hashtags. \n\nI will ask you to classify a tweet as relevant or irrelevant to the political content:\n\nA: Text is RELEVANT if it uses political keywords or hashtags, mentions political figures or events, discusses policy issues such as immigration, abortion, foreign policy, health care, tax, or police shootings, or includes a link to well-known news outlets or other political sources such as think tanks, political pundits or journalists, the White House, or the US Congress.\n\nB: Text is IRRELEVANT if it does not fit the criteria above\n\nThe following tweets are some examples of RELEVANT or IRRELEVANT texts to political contents in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "What inspires me: being a warrior for human dignity, human potential. https://t.co/k6NXxcThaDaaa"
            },
            {
                "role": "assistant",
                "content": "Based on the given criteria, the text you provided is IRRELEVANT to political content. It does not include any political keywords or hashtags, mention political figures or events, discuss policy issues, or provide links to political sources. Instead, it focuses on concepts such as human dignity and human potential.aaa"
            },
            {
                "role": "user",
                "content": "I hope this time brings you the opportunity to reflect, to #hope, and to know that you're loved by God. #MerryChristmas, #GA09! https://t.co/cOfwb9TxWaaaa"
            },
            {
                "role": "assistant",
                "content": "Based on the provided criteria, the text you provided is IRRELEVANT to political content. While it includes hashtags such as #hope, #MerryChristmas, and #GA09, it does not use any political keywords or discuss political figures, events, or policy issues. The focus of the text is on reflection, hope, and sending Christmas wishes.aaa"
            },
            {
                "role": "user",
                "content": "Pres Trump wants another 1 percent for the richest 1 percent. 100 percent irresponsible. https://t.co/CawU4SpUToaaa"
            },
            {
                "role": "assistant",
                "content": """Based on the provided criteria, the text you provided is relevant to political content. It mentions "Pres Trump" as a political figure, discusses a policy issue related to wealth distribution and taxation. Therefore, it meets the criteria for RELEVANT to political content.aaa"""
            },
            {
                "role": "user",
                "content": "I spoke on the floor of the House this morning to honor the life of longtime Syracuse Police Capt Richard Walsh --&gt; https://t.co/6CdNF5jcLcaaa"
            },
            {
                "role": "assistant",
                "content": "Based on the provided criteria, the text you provided is RELEVANT to political content. It mentions speaking on the floor of the House, honoring the life of a police captain, and includes a link to a source (https://t.co/6CdNF5jcLc). This text pertains to government activities, a political event (speech on the House floor), and mentions a political figure (Syracuse Police Capt Richard Walsh).aaa"
            },
            {
                "role": "user",
                "content": "Now, is the following tweet RELEVANT or IRRELEVANT to political content?\n\n{text}"
            }
        ]
        

        # relevance_specific_promot_paper_2024 = ""

        return [
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets_2017_2022 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with 0 and 1)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": relevance_specific_prompt_paper
                        },
                        {
                            "role": "user",
                            "content": "Here's the tweet I picked, please label it as 'Relevant' or 'Irrelevant' by answering with one word:\n{text}"
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets_2017_2022 relevance (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with 0 and 1)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": relevance_specific_prompt_paper + " Here's the tweet I picked, please label it as 'Relevant' or 'Irrelevant' by answering with one word:\n{text}"
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets_2017_2022 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with A and B)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": relevance_specific_prompt_code_2024_zero[0]
                        },
                        {
                            "role": "user",
                            "content": relevance_specific_prompt_code_2024_zero[1]
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets_2017_2022 relevance (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with A and B)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": relevance_specific_prompt_code_2024_zero[0] + "\n\n" + relevance_specific_prompt_code_2024_zero[1]
                        },
                    ],
                },
                {
                    'description': '[original] [few-shot-detailed] gilardi 2024 (code) tweets_2017_2022 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with A and B)'],
                    'prompt_text': relevance_specific_prompt_code_2024_few,
                },
            ]

    def get_all_output_mappings(self):

        return {
            'gilardi relevance': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.relevance_output_mapping},
            },
            'gilardi relevance (extended with 0 and 1)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.relevance_output_mapping_extended_0_and_1},
            },
            'gilardi relevance (extended with A and B)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.relevance_output_mapping_extended_A_and_B},
            },
        }

    def get_dataset_name(self):
        """Get the name of the dataset."""
        return "Gilardi2023_Data3Task1_CongressTweets_2017_2022_Relevance"

    def load_full_dataset(self, return_statistics_for_plotting=False, return_dataset_used_by_gilardi=False):
        """Load and prepare the Gilardi2023_Data3Task1_CongressTweets_2017_2022_Relevance dataset."""

        df_raw = pd.read_csv("data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/mTurk_data/training_data_tweetscongress_relevance.csv")

        # drop rows with NaN values in relevant columns
        df = df_raw.dropna(subset=['relevant_fabio', 'relevant_paula'], how='all')

        # Process for relevance task
        df_relevance = df.dropna(subset=['relevant_fabio', 'relevant_paula'])

        # Now drop duplicates
        df_relevance_without_duplicates = df_relevance.drop_duplicates(subset=['status_id'], keep='first')
        nr_of_duplicates_based_on_id = len(df_relevance) - len(df_relevance_without_duplicates)
        if not return_dataset_used_by_gilardi:
            df_relevance_without_duplicates = df_relevance_without_duplicates.drop_duplicates(subset=['text'], keep='first')
        print(f"\nRows after dropping duplicates: {len(df_relevance_without_duplicates)}")

        # calculate trained annotators agreement
        trained_annotators_agreement = self.intercoder_agreement_percentage(df_relevance_without_duplicates, ['relevant_fabio', 'relevant_paula'])
        trained_annotators_agreement_krippendorff = self.intercoder_agreement_krippendorff(df_relevance_without_duplicates, ['relevant_fabio', 'relevant_paula'])
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement: {trained_annotators_agreement:.4f}')
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement_krippendorff: {trained_annotators_agreement_krippendorff:.4f}')

        df_relevance_with_gt = df_relevance_without_duplicates[
            df_relevance_without_duplicates['relevant_fabio'] == df_relevance_without_duplicates['relevant_paula']
        ]
        df_relevance_with_gt['ground_truth'] = copy.deepcopy(
            df_relevance_with_gt['relevant_fabio']
        )

        nr_of_rows_with_missing_annotations = len(df) - len(df_relevance)
        nr_of_duplicates = len(df_relevance) - len(df_relevance_without_duplicates)
        nr_of_rows_with_disagreement = len(df_relevance_without_duplicates) - len(df_relevance_with_gt)

        dataset_statistics = {
            'Dataset name': self.get_dataset_name(),
            'Total rows': len(df),
            'Total rows (raw)': len(df_raw),
            'Rows without GT NaN values': len(df_relevance),
            'Rows without duplicate values': len(df_relevance_without_duplicates),
            'Trained annotators agreement': trained_annotators_agreement,
            'Trained annotators agreement (krippendorff)': trained_annotators_agreement_krippendorff,
            'Rows with missing annotations': nr_of_rows_with_missing_annotations,
            'Rows with duplicates (based on id)': nr_of_duplicates_based_on_id,
            'Rows with duplicates': nr_of_duplicates,
            'Rows with disagreement': nr_of_rows_with_disagreement,
            'Final dataset size': len(df_relevance_with_gt),
            # 'Rows which all trained annotators agree [relevant=1]': sum(df_relevance_with_gt['ground_truth']==1),
            'Value counts': df_relevance_with_gt['ground_truth'].value_counts(),
        }
        for k, v in dataset_statistics.items():
            print(f"    {k}: {v}")

        df_relevance_with_gt = self.order_df_columns(df_relevance_with_gt)
        df_relevance_with_gt['ground_truth'] = df_relevance_with_gt['ground_truth'].map({0: 'irrelevant', 1: 'relevant'})

        if return_statistics_for_plotting:
            return df_relevance_with_gt, dataset_statistics
        else:
            return df_relevance_with_gt


    def _get_mturk_file_path(self):
        return 'data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/mTurk_data/batch_results_tweetscongress_relevance_final.csv'


class Gilardi2023_Data3Task2_CongressTweets_2017_2022PoliticalFrame(Gilardi2023):
    """Implementation for Gilardi 2023 Problem/Solution Frame Task."""

    def __init__(self, data_directory='gilardi_et_al_pnas', config_fn='config_framesII_tweets17.yaml'):
        super().__init__(data_directory, config_fn)

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following tweet is mainly about the topic {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Tweet: {text}

Probability:'''

    def get_prompts(self):
        # prompt copied from paper Appendix G: https://www.pnas.org/lookup/suppl/doi:10.1073/pnas.2305016120/-/DCSupplemental
        frame_problem_solution_prompt = "Political content, as described above, can be linked to various other topics, such as health, crime, or equality.\nFor each tweet in the sample, follow these instructions:\n1. Carefully read the text of the tweet, paying close attention to details.\n2. Classify the tweet into one of the topics defined below.\nThe topics are defined as follows:\n• ECONOMY: The costs, benefits, or monetary/financial implications of the issue (to an individual, family, community, or to the economy as a whole).\n• Capacity and resources: The lack of or availability of physical, geographical, spatial, human, and financial resources, or the capacity of existing systems and resources to implement or carry out policy goals.\n• MORALITY: Any perspective—or policy objective or action (including proposed action)that is compelled by religious doctrine or interpretation, duty, honor, righteousness or any other sense of ethics or social responsibility.\n• FAIRNESS AND EQUALITY: Equality or inequality with which laws, punishment, rewards, and resources are applied or distributed among individuals or groups. Also the balance between the rights or interests of one individual or group compared to another individual or group.\n• CONSTITUTIONALITY AND JURISPRUDENCE: The constraints imposed on or freedoms granted to individuals, government, and corporations via the Constitution, Bill of Rights and other amendments, or judicial interpretation. This deals specifically with the authority of government to regulate, and the authority of individuals/corporations to act independently of government.\n• POLICY PRESCRIPTION AND EVALUATION: Particular policies proposed for addressing an identified problem, and figuring out if certain policies will work, or if existing policies are effective.\n• LAW AND ORDER, CRIME AND JUSTICE: Specific policies in practice and their enforcement, incentives, and implications. Includes stories about enforcement and interpretation of laws by individuals and law enforcement, breaking laws, loopholes, fines, sentencing and punishment. Increases or reductions in crime.\n• SECURITY AND DEFENSE: Security, threats to security, and protection of one’s person, family, in-group, nation, etc. Generally an action or a call to action that can be taken to protect the welfare of a person, group, nation sometimes from a not yet manifested threat.\n• HEALTH AND SAFETY: Health care access and effectiveness, illness, disease, sanitation, obesity, mental health effects, prevention of or perpetuation of gun violence, infrastructure and building safety.\n• QUALITY OF LIFE: The effects of a policy on individuals’ wealth, mobility, access to resources, happiness, social structures, ease of day-to-day routines, quality of community life, etc.\n• CULTURAL IDENTITY: The social norms, trends, values and customs constituting culture(s), as they relate to a specific policy issue.\n• PUBLIC OPINION: References to general social attitudes, polling and demographic information, as well as implied or actual consequences of diverging from or “getting ahead of” public opinion or polls.\n• POLITICAL: Any political considerations surrounding an issue. Issue actions or efforts or stances that are political, such as partisan filibusters, lobbyist involvement, bipartisan efforts, deal-making and vote trading, appealing to one’s base, mentions of political maneuvering. Explicit statements that a policy issue is good or bad for a particular political party.\n• EXTERNAL REGULATION AND REPUTATION: The United States’ external relations with another nation; the external relations of one state with another; or relations between groups. This includes trade agreements and outcomes, comparisons of policy outcomes or desired policy outcomes.\n• OTHER: Any topic that does not fit into the above categories."
        

        gilardi_frame_prompt_paper = self.gilardi_political_content_prompt_paper + \
            frame_problem_solution_prompt

        tweet_instruction_prompt = "Here's the tweet I picked, please label it as 'Economy', 'Capacity and resources', 'Morality', 'Fairness and Equality', 'Constitutionality and Jurisprudence', 'Policy Prescription and Evaluation', 'Law and Order, Crime and Justice', 'Security and Defense', 'Health and Safety', 'Quality of Life', 'Cultural Identity', 'Public Opinion', 'Political', 'External Regulation and Reputation', or 'Other' by answering with one word:\n{text}"


        # the following prompts are copied from Gilardi 2024 code: data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv
        # available for download at: https://osf.io/adkun/files/osfstorage
        # a=pd.read_csv("data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv")


        frame_specific_prompt_code_2024_zero = ["“Political content” refers to a text that pertains to politics or government policies at the local, national, or international level. This can include political figures, events, or issues, as well as text that uses political language or hashtags. \n\nI will ask you to classify a tweet as one of the frames defined below:\n\nECONOMY: The costs, benefits, or monetary/financial implications of the issue (to an individual, family, community, or to the economy as a whole).\nMORALITY: Any perspective—or policy objective or action (including proposed action)that is compelled by religious doctrine or interpretation, duty, honor, righteousness or any other sense of ethics or social responsibility.\nFAIRNESS AND EQUALITY: Equality or inequality with which laws, punishment, rewards, and resources are applied or distributed among individuals or groups. Also the balance between the rights or interests of one individual or group compared to another individual or group.\nPOLICY PRESCRIPTION AND EVALUATION: Particular policies proposed for addressing an identified problem, and figuring out if certain policies will work, or if existing policies are effective.\nLAW AND ORDER, CRIME AND JUSTICE: Specific policies in practice and their enforcement, incentives, and implications. Includes stories about enforcement and interpretation of laws by individuals and law enforcement, breaking laws, loopholes, fines, sentencing and punishment. Increases or reductions in crime.\nSECURITY AND DEFENSE: Security, threats to security, and protection of one’s person, family, in-group, nation, etc. Generally an action or a call to action that can be taken to protect the welfare of a person, group, nation sometimes from a not yet manifested threat.\nHEALTH AND SAFETY: Health care access and effectiveness, illness, disease, sanitation, obesity, mental health effects, prevention of or perpetuation of gun violence, infrastructure and building safety.\nQUALITY OF LIFE: The effects of a policy on individuals’ wealth, mobility, access to resources, happiness, social structures, ease of day-to-day routines, quality of community life, etc.\nPOLITICAL: Any political considerations surrounding an issue. Issue actions or efforts or stances that are political, such as partisan filibusters, lobbyist involvement, bipartisan efforts, deal-making and vote trading, appealing to one's base, mentions of political maneuvering. Explicit statements that a policy issue is good or bad for a particular political party.\nEXTERNAL REGULATION AND REPUTATION: The United States’ external relations with another nation; the external relations of one state with another; or relations between groups. This includes trade agreements and outcomes, comparisons of policy outcomes or desired policy outcomes.\nOTHER: Any topic that does not fit into the above categories.", "Now, which of the above frames best fit the following tweet? Answer with only the option below that is most accurate and nothing else.\n\nA: ECONOMY \nB: MORALITY\nC: FAIRNESS AND EQUALITY\nD: POLICY PRESCRIPTION AND EVALUATION \nE: LAW AND ORDER, CRIME AND JUSTICE\nF: SECURITY AND DEFENSE\nG: HEALTH AND SAFETY\nH: QUALITY OF LIFE\nI: POLITICAL\nJ: EXTERNAL REGULATION AND REPUTATION\nK: OTHER\n\n{text}"]



        frame_specific_prompt_code_2024_few = [
            {
                "role": "system",
                "content": "“Political content” refers to a text that pertains to politics or government policies at the local, national, or international level. This can include political figures, events, or issues, as well as text that uses political language or hashtags. \n\nI will ask you to classify a tweet as one of the frames defined below:\n\nECONOMY: The costs, benefits, or monetary/financial implications of the issue (to an individual, family, community, or to the economy as a whole).\nMORALITY: Any perspective—or policy objective or action (including proposed action)that is compelled by religious doctrine or interpretation, duty, honor, righteousness or any other sense of ethics or social responsibility.\nFAIRNESS AND EQUALITY: Equality or inequality with which laws, punishment, rewards, and resources are applied or distributed among individuals or groups. Also the balance between the rights or interests of one individual or group compared to another individual or group.\nPOLICY PRESCRIPTION AND EVALUATION: Particular policies proposed for addressing an identified problem, and figuring out if certain policies will work, or if existing policies are effective.\nLAW AND ORDER, CRIME AND JUSTICE: Specific policies in practice and their enforcement, incentives, and implications. Includes stories about enforcement and interpretation of laws by individuals and law enforcement, breaking laws, loopholes, fines, sentencing and punishment. Increases or reductions in crime.\nSECURITY AND DEFENSE: Security, threats to security, and protection of one’s person, family, in-group, nation, etc. Generally an action or a call to action that can be taken to protect the welfare of a person, group, nation sometimes from a not yet manifested threat.\nHEALTH AND SAFETY: Health care access and effectiveness, illness, disease, sanitation, obesity, mental health effects, prevention of or perpetuation of gun violence, infrastructure and building safety.\nQUALITY OF LIFE: The effects of a policy on individuals’ wealth, mobility, access to resources, happiness, social structures, ease of day-to-day routines, quality of community life, etc.\nPOLITICAL: Any political considerations surrounding an issue. Issue actions or efforts or stances that are political, such as partisan filibusters, lobbyist involvement, bipartisan efforts, deal-making and vote trading, appealing to one's base, mentions of political maneuvering. Explicit statements that a policy issue is good or bad for a particular political party.\nEXTERNAL REGULATION AND REPUTATION: The United States’ external relations with another nation; the external relations of one state with another; or relations between groups. This includes trade agreements and outcomes, comparisons of policy outcomes or desired policy outcomes.\nOTHER: Any topic that does not fit into the above categories.\n\nThe following tweets are some examples of these frames in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "HURRY!!! Today is the last day to sign up for health insurance that begins on Jan 1. Visit https://t.co/rrKeGJOFBA to #GetCoveredNow. #ACA https://t.co/LCMQNHjCMN"
            },
            {
                "role": "assistant",
                "content": "G"
            },
            {
                "role": "user",
                "content": "The #CHOICEAct provides regulatory relief for community banks &amp; credit unions promoting more economic opportunity → https://t.co/uOBmHKhrxkhttps://t.co/64WGHA1D2R"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "The #REINSAct signals our commitment to modeling reform that puts smart policy above tired politicking. https://t.co/GpOcD1NZO7"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": "Tonight it was my distinct privilege to speak on the Senate floor in support of my friend &amp; our Attorney General Jeff Sessions. https://t.co/UoIYp1R3ES"
            },
            {
                "role": "assistant",
                "content": "I"
            },
            {
                "role": "user",
                "content": "Thanks @Astro_Kate7 for speaking w/students at her Alma mater @VHS_Crusheds about her groundbreaking work on the International Space Station https://t.co/UXnh8STwaN"
            },
            {
                "role": "assistant",
                "content": "K"
            },
            {
                "role": "user",
                "content": "I always thought the best soldiers end up at Joint Base Lewis-McChord, but here's proof. Congrats to the 1st Special Forces Group (Airborne) sniper team! @JBLM_PAO @TaskandPurpose https://t.co/x8nX6HyYOQ"
            },
            {
                "role": "assistant",
                "content": "F"
            },
            {
                "role": "user",
                "content": "As I told #SouthKorea leaders during my visit in Dec, US is committed to a strong alliance despite political turmoil https://t.co/8orrFs8atv"
            },
            {
                "role": "assistant",
                "content": "J"
            },
            {
                "role": "user",
                "content": "Proud to #StandWithGavin and all transgender students. Every child deserves to go to school &amp; live as who they are free from discrimination. https://t.co/4uqpuHzbCd"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "The prosecution of corruption by high ranking government officials, even years after the crimes were committed, is critical to..."
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": """The Trump-Sessions "zero tolerance" family separation border policies are not required, right or moral. https://t.co/aAFX8Q6eKT"""
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Wisconsin is full of great role models and leaders. Congratulations to all of the outstanding women honored by the La Crosse YWCA, and thank you for making the coulee region a better place to live! https://t.co/mj1HK4PwzI"
            },
            {
                "role": "assistant",
                "content": "H"
            },
            {
                "role": "user",
                "content": "Now, which of the above frames best fit the following tweet? Answer with only the option below that is most accurate and nothing else.\n\nA: ECONOMY \nB: MORALITY\nC: FAIRNESS AND EQUALITY\nD: POLICY PRESCRIPTION AND EVALUATION \nE: LAW AND ORDER, CRIME AND JUSTICE\nF: SECURITY AND DEFENSE\nG: HEALTH AND SAFETY\nH: QUALITY OF LIFE\nI: POLITICAL\nJ: EXTERNAL REGULATION AND REPUTATION\nK: OTHER\n\n{text}"
            }
        ]


        frame_specific_prompt_code_2024_cot = [
            {
                "role": "system",
                "content": "“Political content” refers to a text that pertains to politics or government policies at the local, national, or international level. This can include political figures, events, or issues, as well as text that uses political language or hashtags. \n\nI will ask you to classify a tweet as one of the frames defined below:\n\nECONOMY: The costs, benefits, or monetary/financial implications of the issue (to an individual, family, community, or to the economy as a whole).\nMORALITY: Any perspective—or policy objective or action (including proposed action)that is compelled by religious doctrine or interpretation, duty, honor, righteousness or any other sense of ethics or social responsibility.\nFAIRNESS AND EQUALITY: Equality or inequality with which laws, punishment, rewards, and resources are applied or distributed among individuals or groups. Also the balance between the rights or interests of one individual or group compared to another individual or group.\nPOLICY PRESCRIPTION AND EVALUATION: Particular policies proposed for addressing an identified problem, and figuring out if certain policies will work, or if existing policies are effective.\nLAW AND ORDER, CRIME AND JUSTICE: Specific policies in practice and their enforcement, incentives, and implications. Includes stories about enforcement and interpretation of laws by individuals and law enforcement, breaking laws, loopholes, fines, sentencing and punishment. Increases or reductions in crime.\nSECURITY AND DEFENSE: Security, threats to security, and protection of one’s person, family, in-group, nation, etc. Generally an action or a call to action that can be taken to protect the welfare of a person, group, nation sometimes from a not yet manifested threat.\nHEALTH AND SAFETY: Health care access and effectiveness, illness, disease, sanitation, obesity, mental health effects, prevention of or perpetuation of gun violence, infrastructure and building safety.\nQUALITY OF LIFE: The effects of a policy on individuals’ wealth, mobility, access to resources, happiness, social structures, ease of day-to-day routines, quality of community life, etc.\nPOLITICAL: Any political considerations surrounding an issue. Issue actions or efforts or stances that are political, such as partisan filibusters, lobbyist involvement, bipartisan efforts, deal-making and vote trading, appealing to one's base, mentions of political maneuvering. Explicit statements that a policy issue is good or bad for a particular political party.\nEXTERNAL REGULATION AND REPUTATION: The United States’ external relations with another nation; the external relations of one state with another; or relations between groups. This includes trade agreements and outcomes, comparisons of policy outcomes or desired policy outcomes.\nOTHER: Any topic that does not fit into the above categories.\n\nThe following tweets are some examples of these frames in the form of question and answer:"
            },
            {
                "role": "user",
                "content": "HURRY!!! Today is the last day to sign up for health insurance that begins on Jan 1. Visit https://t.co/rrKeGJOFBA to #GetCoveredNow. #ACA https://t.co/LCMQNHjCMN"
            },
            {
                "role": "assistant",
                "content": """The frame that best fits the given text is "HEALTH AND SAFETY" as the text is about the last day to sign up for health insurance and urging people to get covered. The hashtag #ACA refers to the Affordable Care Act, which is a health care policy."""
            },
            {
                "role": "user",
                "content": "The #CHOICEAct provides regulatory relief for community banks &amp; credit unions promoting more economic opportunity → https://t.co/uOBmHKhrxkhttps://t.co/64WGHA1D2R"
            },
            {
                "role": "assistant",
                "content": "The best fitting frame for the given text is ECONOMY as it pertains to financial implications and regulatory relief for community banks and credit unions, promoting economic opportunity. The text explicitly mentions the #CHOICEAct, which is a bill designed to provide regulatory relief to community banks and credit unions. Therefore, the text is focused on the economic benefits and implications of a particular policy proposal."
            },
            {
                "role": "user",
                "content": "The #REINSAct signals our commitment to modeling reform that puts smart policy above tired politicking. https://t.co/GpOcD1NZO7"
            },
            {
                "role": "assistant",
                "content": "The text you provided falls under the frame of POLICY PRESCRIPTION AND EVALUATION, as it refers to the REINS Act and emphasizes the commitment to modeling reform and smart policy."
            },
            {
                "role": "user",
                "content": "Tonight it was my distinct privilege to speak on the Senate floor in support of my friend &amp; our Attorney General Jeff Sessions. https://t.co/UoIYp1R3ES"
            },
            {
                "role": "assistant",
                "content": """The best-fit frame for the given text is POLITICAL. It explicitly mentions the speaker's support for the Attorney General, who is a political figure, and the mention of speaking on "Senate floor" implies a political setting. Additionally, the mention of "privilege" indicates the political significance of the event for the speaker."""
            },
            {
                "role": "user",
                "content": "Thanks @Astro_Kate7 for speaking w/students at her Alma mater @VHS_Crusheds about her groundbreaking work on the International Space Station https://t.co/UXnh8STwaN"
            },
            {
                "role": "assistant",
                "content": """The best fit for this text is the "OTHER" frame because it doesn't seem to fit into any of the categories listed above. It doesn't pertain to any political figures, events, or issues, nor does it use political language or hashtags. This text is about education and career and may inspire students. Therefore, it can be categorized as "OTHER"."""
            },
            {
                "role": "user",
                "content": "I always thought the best soldiers end up at Joint Base Lewis-McChord, but here's proof. Congrats to the 1st Special Forces Group (Airborne) sniper team! @JBLM_PAO @TaskandPurpose https://t.co/x8nX6HyYOQ"
            },
            {
                "role": "assistant",
                "content": "The frame that best fits the following text is SECURITY AND DEFENSE. The text mentions Joint Base Lewis-McChord, which is a military installation. It also congratulates a sniper team from the 1st Special Forces Group, which is a military unit. Therefore, the text is related to military personnel and their accomplishments, indicating a focus on security and defense."
            },
            {
                "role": "user",
                "content": "As I told #SouthKorea leaders during my visit in Dec, US is committed to a strong alliance despite political turmoil https://t.co/8orrFs8atv"
            },
            {
                "role": "assistant",
                "content": """The frame that best fits the following text is "EXTERNAL REGULATION AND REPUTATION". The reason being, the text talks about the relations between the United States and South Korea, which falls under the external relations of one state with another. It also mentions a commitment to a strong alliance which is relevant to trade agreements and outcomes, comparisons of policy outcomes, or desired policy outcomes."""
            },
            {
                "role": "user",
                "content": "Proud to #StandWithGavin and all transgender students. Every child deserves to go to school &amp; live as who they are free from discrimination. https://t.co/4uqpuHzbCd"
            },
            {
                "role": "assistant",
                "content": """Based on the provided text, the best frame that fits the text is "FAIRNESS AND EQUALITY." The text expresses support for transgender students and emphasizes the importance of ensuring that every child can go to school and live free from discrimination. This relates to the equality or inequality with which laws and resources are applied or distributed among individuals or groups. In this case, it pertains to the rights and interests of transgender students compared to others, highlighting the need for fairness and equal treatment."""
            },
            {
                "role": "user",
                "content": "The prosecution of corruption by high ranking government officials, even years after the crimes were committed, is critical to..."
            },
            {
                "role": "assistant",
                "content": """Based on the provided text, the best frame that fits the text is "LAW AND ORDER, CRIME AND JUSTICE." The text discusses the prosecution of corruption by high-ranking government officials, which falls under specific policies in practice, their enforcement, incentives, and implications. It involves the enforcement and interpretation of laws, the accountability of government officials, and the pursuit of justice in addressing corruption."""
            },
            {
                "role": "user",
                "content": """The Trump-Sessions "zero tolerance" family separation border policies are not required, right or moral. https://t.co/aAFX8Q6eKT"""
            },
            {
                "role": "assistant",
                "content": """Based on the provided text, the frame that best fits is: MORALITY. The text expresses a moral perspective on the Trump-Sessions "zero tolerance" family separation border policies, stating that they are not required, right, or moral. It addresses the ethical dimension and social responsibility associated with these policies."""
            },
            {
                "role": "user",
                "content": "Wisconsin is full of great role models and leaders. Congratulations to all of the outstanding women honored by the La Crosse YWCA, and thank you for making the coulee region a better place to live! https://t.co/mj1HK4PwzI"
            },
            {
                "role": "assistant",
                "content": "Based on the provided text, the frame that best fits is: QUALITY OF LIFE. The text highlights the positive impact of the outstanding women honored by the La Crosse YWCA in making the coulee region a better place to live. It emphasizes the effects of their contributions on the quality of life in the community"
            },
            {
                "role": "user",
                "content": "Now, which of the above frames best fit the following tweet? Answer with only the option below that is most accurate and nothing else.\n\nA: ECONOMY \nB: MORALITY\nC: FAIRNESS AND EQUALITY\nD: POLICY PRESCRIPTION AND EVALUATION \nE: LAW AND ORDER, CRIME AND JUSTICE\nF: SECURITY AND DEFENSE\nG: HEALTH AND SAFETY\nH: QUALITY OF LIFE\nI: POLITICAL\nJ: EXTERNAL REGULATION AND REPUTATION\nK: OTHER\n\n{text}"
            }
        ]

        # frame_specific_promot_paper_2024 = ""


        return [
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets_2017_2022 frame',
                    'compatible_output_mapping': ['gilardi political frame'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": gilardi_frame_prompt_paper
                        },
                        {
                            "role": "user",
                            "content": tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) tweets_2017_2022 frame (no system prompt)',
                    'compatible_output_mapping': ['gilardi political frame'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": gilardi_frame_prompt_paper + "\n" + tweet_instruction_prompt

                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets_2017_2022 frame',
                    'compatible_output_mapping': [
                        # 'gilardi political frame', 
                        'gilardi political frame (extended with A to K)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": frame_specific_prompt_code_2024_zero[0]
                        },
                        {
                            "role": "user",
                            "content": frame_specific_prompt_code_2024_zero[1]
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) tweets_2017_2022 frame (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi political frame', 
                        'gilardi political frame (extended with A to K)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": frame_specific_prompt_code_2024_zero[0] + "\n\n" + frame_specific_prompt_code_2024_zero[1]
                        },
                    ],
                },
                {
                    'description': '[original] [few-shot-detailed] gilardi 2024 (code) tweets_2017_2022 frame',
                    'compatible_output_mapping': [
                        # 'gilardi political frame', 
                        'gilardi political frame (extended with A to K)'],
                    'prompt_text': frame_specific_prompt_code_2024_few,
                },
            ]


    def get_all_output_mappings(self):

        return {
            'gilardi political frame': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.frame_political_output_mapping},
            },
            'gilardi political frame (extended with A to K)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.frame_political_output_mapping_extended_A_to_K},
            },
        }

    def get_dataset_name(self):
        """Get the name of the dataset."""
        return "Gilardi2023_Data3Task2_CongressTweets_2017_2022PoliticalFrame"


    def load_full_dataset(self, return_statistics_for_plotting=False, return_dataset_used_by_gilardi=False):
        """Load and prepare the Gilardi2023_Data3Task2_CongressTweets_2017_2022PoliticalFrame dataset."""
        # Load data from Excel files
        df_raw = pd.read_csv("data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/mTurk_data/training_data_tweetscongress_relevance.csv")

        df = df_raw.dropna(subset=['relevant_fabio', 'relevant_paula'], how='all')

        # Filter to only relevant tweets and process for frame task
        df_relevant = df[
            (df['relevant_fabio'] == 1) &
            (df['relevant_paula'] == 1)
        ]

        # Now drop duplicates
        df_frame_without_duplicates = df_relevant.drop_duplicates(subset=['status_id'], keep='first')
        nr_of_duplicates_based_on_id = len(df_relevant) - len(df_frame_without_duplicates)
        if not return_dataset_used_by_gilardi:
            df_frame_without_duplicates = df_frame_without_duplicates.drop_duplicates(subset=['text'], keep='first')
        print(f"\nRows after dropping duplicates: {len(df_frame_without_duplicates)}")

        df_frame = df_frame_without_duplicates.dropna(
            subset=['frame_fabio', 'frame_paula']
        )

        # calculate trained annotators agreement
        trained_annotators_agreement = self.intercoder_agreement_percentage(df_frame, ['frame_fabio', 'frame_paula'])
        trained_annotators_agreement_krippendorff = self.intercoder_agreement_krippendorff(df_frame, ['frame_fabio', 'frame_paula'])
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement: {trained_annotators_agreement:.4f}')
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement_krippendorff: {trained_annotators_agreement_krippendorff:.4f}')

        # Keep only rows where both annotators agree
        df_frame_with_gt = df_frame[
            (df_frame['frame_fabio'] ==
                df_frame['frame_paula'])
        ]
        df_frame_with_gt['ground_truth'] = copy.deepcopy(
            df_frame_with_gt['frame_fabio']
        )

        nr_of_duplicates = len(df_relevant) - len(df_frame_without_duplicates)
        nr_of_rows_with_missing_annotations = len(df_frame_without_duplicates) - len(df_frame)
        nr_of_rows_with_disagreement = len(df_frame) - len(df_frame_with_gt)

        dataset_statistics = {
            'Dataset name': self.get_dataset_name(),
            'Total rows': len(df),
            'Total rows (raw)': len(df_raw),
            'Total relevant rows': len(df_relevant),
            'Rows without duplicate values': len(df_frame_without_duplicates),
            'Rows without GT NaN values': len(df_frame),
            # 'Rows without both values (problem and solution)': len(df_frame),
            'Trained annotators agreement': trained_annotators_agreement,
            'Trained annotators agreement (krippendorff)': trained_annotators_agreement_krippendorff,
            'Rows with duplicates (based on id)': nr_of_duplicates_based_on_id,
            'Rows with duplicates': nr_of_duplicates,
            'Rows with missing annotations': nr_of_rows_with_missing_annotations,
            'Rows with disagreement': nr_of_rows_with_disagreement,
            'Final dataset size': len(df_frame_with_gt),
            'Value counts': df_frame_with_gt['ground_truth'].value_counts(),
        }
        for k, v in dataset_statistics.items():
            print(f"    {k}: {v}")

        df_frame_with_gt['ground_truth'] = df_frame_with_gt.apply(self.get_detailed_frames_ground_truth, axis=1)

        df_frame_with_gt = self.order_df_columns(df_frame_with_gt)

        if return_statistics_for_plotting:
            return df_frame_with_gt, dataset_statistics
        else:
            return df_frame_with_gt


    def _get_mturk_file_path(self):
        return 'data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/mTurk_data/batch_results_tweetscongress_frames_final.csv'



class Gilardi2023_Data4Task1_News_2020_2021(Gilardi2023):
    """Implementation for Gilardi 2023 Relevance Task."""

    def __init__(self, data_directory='gilardi_et_al_pnas', config_fn='config_relevance_news.yaml'):
        super().__init__(data_directory, config_fn)


    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following tweet is {previous_answer_placeholder} to the topic of content moderation?

Output only a single number between 0 and 1, without any context or explanation.

Tweet: {text}

Probability:'''

    def get_prompts(self):
        # the following prompts are copied from Gilardi 2024 code: data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv
        # available for download at: https://osf.io/adkun/files/osfstorage
        # a=pd.read_csv("data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv")
        relevance_specific_prompt_code_2024_zero = ["“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Text is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like flagging, shadowbanning, or account suspension.\n\nB: Text is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.", "Now, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"]

        relevance_specific_prompt_code_2024_few = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Text is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like flagging, shadowbanning, or account suspension.\n\nB: Text is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nThe following texts are some examples of 'RELEVANT' or 'IRRELEVANT' texts to content moderation in the form of question and answer:"
            },
            {
                "role": "user",
                "content": """TORONTO - Ontario Premier Doug Ford on Monday said the United States had blocked the delivery of nearly three million face masks at the American border over the weekend. Ford said restrictions on shipments at the U.S. border have left the province with just one more week's worth of personal protective equipment for health-care workers fighting the coronavirus outbreak in Ontario. In a statement today, he says Ontario is ramping up its own production of personal protective equipment, but most of those supplies are weeks away from being in the hands of front-line health workers. At least 451 health-care workers in Ontario have tested positive for COVID-19, representing about 10 per cent of all cases in the province. In all, Ontario reported 309 new COVID-19 cases today, including 13 new deaths. There have now been a total of 4,347 cases in the province, including 1,624 patients who have recovered and 132 deaths. Allies of the United States are complaining about its "Wild West" tactics in outbidding or blocking shipments to buyers who have already signed deals for medical equipment. Prime Minister Justin Trudeau sidestepped reporters' questions about the incident on Monday, saying his government was in productive talks with the United States and adding: "We expect"""
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": """A look at the first quarter of the year that was 2019. January 3: China's "Chang'e 4" is the first space probe to land on the far side of the moon. January 11: Macedonia is now North Macedonia, ending a row with Greece and paving the way for NATO membership and EU accession talks. The Greek parliament ratifies the historic name agreement on January 25.January 13: The mayor of the Polish city of Gdansk, Pawel Adamowicz, 53, is stabbed to death by a previously convicted bank robber during a fundraiser. January 15: Attackers claimed by the Somalia-based group al-Shabaab storm the upmarket Dusit hotel in the Kenyan capital Nairobi, killing more than 20. January 18: After four months of political deadlock in Sweden, Social Democratic leader Stefan Lofven wins a vote in parliament to form a government. January 18: At least 109 people are killed when a fuel pipeline explodes in the Mexican city of Tlahuelilpan. January 22: The EU Commission imposes a fine of 570 million euros on the credit card company Mastercard for artificially pushing up the cost of card payments. January 23: Juan Guaido, the head of Venezuela's opposition-dominated National Assembly, declares himself the country's interim president. January 24: Felix Tshisekedi is"""
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Bhubaneswar, Oct. 29 -- New Delhi: The Supreme Court has severely criticised the growing trend of police in different States summoning individuals from far corners of the country over social media postings. Hearing a petition concerning a Delhi resident Roshni Biswas who was reportedly summoned by Bengal Police for posting objectionable content on Facebook, the Bench of Justices DY Chandrachud and Indira Banerjee noted that police's power to issue summons under Section 41A of the Code of Criminal Procedure (CrPC) cannot be used to intimidate, threaten and harass. As per reports, the apex court's comment was prompted by Bengal Police issuing summons to the 29-year-old woman who, in a Facebook post, had criticised the Mamata government for non-enforcement of lockdown norms. The FIR which relies on FB links contains a statement that the posts implied the State administration was going soft on the violation of the lockdown at Rajabazar as the area is predominantly inhabited by a particular community and that the administration is complacent while dealing with lockdown violations caused by a certain segment of the community. Mahesh Jethmalani, learned senior counsel appearing on behalf of the petitioner submitted that the petitioner has stated on oath that she disclaims any association with"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": """Facebook and Instagram took down video tributes to George Floyd posted by the Trump campaign over copyright complaints on Friday, following a similar decision by Twitter - broadening the latest skirmish over the policing of online messages from President Donald Trump and his allies. Facebook and its subsidiary Instagram removed posts by official Trump campaign accounts that included videos narrated by Trump discussing Floyd's death in Minneapolis. The narration is played over a series of photographs and videos that appear to have been taken during recent protests around the country over Floyd's killing, Politico reported. We received a copyright complaint from the creator under the Digital Millennium Copyright Act and have removed the post," Facebook Spokesperson Andy Stone told POLITICO in an email, adding, "Organizations that use original art shared on Instagram are expected to have the right to do so. The move by the companies follows a parallel action by Twitter, which on Thursday morning disabled the same video included in a pair of tweets by @TeamTrump and @TrumpWarRoom 2020 campaign accounts, also citing an unspecified complaint under the Digital Millennium Copyright Act. The company left up the tweets but placed an overlay on the videos that reads, This media has been"""
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Now, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
            }
        ]


        relevance_specific_prompt_code_2024_cot = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as relevant or irrelevant to the content moderation:\n\nA: Text is RELEVANT if it includes: social media platforms’ content moderation rules and practices, censorship, governments’ regulation of online content moderation, and/or mild forms of content moderation like flagging, shadowbanning, or account suspension.\n\nB: Text is IRRELEVANT if they do not refer to content moderation, as defined above. This would include, for example, a tweet by Trump that Twitter has labeled his tweet as “disputed”, or a tweet claiming that something is false.\n\nThe following texts are some examples of 'RELEVANT' or 'IRRELEVANT' texts to content moderation in the form of question and answer:"
            },
            {
                "role": "user",
                "content": """TORONTO - Ontario Premier Doug Ford on Monday said the United States had blocked the delivery of nearly three million face masks at the American border over the weekend. Ford said restrictions on shipments at the U.S. border have left the province with just one more week's worth of personal protective equipment for health-care workers fighting the coronavirus outbreak in Ontario. In a statement today, he says Ontario is ramping up its own production of personal protective equipment, but most of those supplies are weeks away from being in the hands of front-line health workers. At least 451 health-care workers in Ontario have tested positive for COVID-19, representing about 10 per cent of all cases in the province. In all, Ontario reported 309 new COVID-19 cases today, including 13 new deaths. There have now been a total of 4,347 cases in the province, including 1,624 patients who have recovered and 132 deaths. Allies of the United States are complaining about its "Wild West" tactics in outbidding or blocking shipments to buyers who have already signed deals for medical equipment. Prime Minister Justin Trudeau sidestepped reporters' questions about the incident on Monday, saying his government was in productive talks with the United States and adding: "We expect"""
            },
            {
                "role": "assistant",
                "content": "The provided text is 'IRRELEVANT' to content moderation. It discusses the United States blocking the delivery of face masks at the border, Ontario's shortage of personal protective equipment, COVID-19 cases and deaths, and complaints about the United States' tactics in outbidding or blocking shipments. It does not refer to social media platforms, content moderation rules, censorship, or any forms of online content moderation."
            },
            {
                "role": "user",
                "content": """A look at the first quarter of the year that was 2019. January 3: China's "Chang'e 4" is the first space probe to land on the far side of the moon. January 11: Macedonia is now North Macedonia, ending a row with Greece and paving the way for NATO membership and EU accession talks. The Greek parliament ratifies the historic name agreement on January 25.January 13: The mayor of the Polish city of Gdansk, Pawel Adamowicz, 53, is stabbed to death by a previously convicted bank robber during a fundraiser. January 15: Attackers claimed by the Somalia-based group al-Shabaab storm the upmarket Dusit hotel in the Kenyan capital Nairobi, killing more than 20. January 18: After four months of political deadlock in Sweden, Social Democratic leader Stefan Lofven wins a vote in parliament to form a government. January 18: At least 109 people are killed when a fuel pipeline explodes in the Mexican city of Tlahuelilpan. January 22: The EU Commission imposes a fine of 570 million euros on the credit card company Mastercard for artificially pushing up the cost of card payments. January 23: Juan Guaido, the head of Venezuela's opposition-dominated National Assembly, declares himself the country's interim president. January 24: Felix Tshisekedi is"""
            },
            {
                "role": "assistant",
                "content": "The provided text is 'IRRELEVANT' to content moderation. It discusses various events that took place in the first quarter of 2019, including space exploration, political developments, criminal incidents, and international affairs. It does not refer to social media platforms, content moderation rules, censorship, or any forms of online content moderation."
            },
            {
                "role": "user",
                "content": "Bhubaneswar, Oct. 29 -- New Delhi: The Supreme Court has severely criticised the growing trend of police in different States summoning individuals from far corners of the country over social media postings. Hearing a petition concerning a Delhi resident Roshni Biswas who was reportedly summoned by Bengal Police for posting objectionable content on Facebook, the Bench of Justices DY Chandrachud and Indira Banerjee noted that police's power to issue summons under Section 41A of the Code of Criminal Procedure (CrPC) cannot be used to intimidate, threaten and harass. As per reports, the apex court's comment was prompted by Bengal Police issuing summons to the 29-year-old woman who, in a Facebook post, had criticised the Mamata government for non-enforcement of lockdown norms. The FIR which relies on FB links contains a statement that the posts implied the State administration was going soft on the violation of the lockdown at Rajabazar as the area is predominantly inhabited by a particular community and that the administration is complacent while dealing with lockdown violations caused by a certain segment of the community. Mahesh Jethmalani, learned senior counsel appearing on behalf of the petitioner submitted that the petitioner has stated on oath that she disclaims any association with"
            },
            {
                "role": "assistant",
                "content": "The provided text is 'RELEVANT' to content moderation because it discusses the issue of objectionable content posted on social media and the actions taken by law enforcement. It also mentions the power of police to issue summons and highlights the need to avoid using it to harass or threaten individuals. Additionally, the text refers to a specific case involving criticism of the government's enforcement of lockdown rules, which could potentially be linked to content moderation practices on social media platforms."
            },
            {
                "role": "user",
                "content": """Facebook and Instagram took down video tributes to George Floyd posted by the Trump campaign over copyright complaints on Friday, following a similar decision by Twitter - broadening the latest skirmish over the policing of online messages from President Donald Trump and his allies. Facebook and its subsidiary Instagram removed posts by official Trump campaign accounts that included videos narrated by Trump discussing Floyd's death in Minneapolis. The narration is played over a series of photographs and videos that appear to have been taken during recent protests around the country over Floyd's killing, Politico reported. We received a copyright complaint from the creator under the Digital Millennium Copyright Act and have removed the post," Facebook Spokesperson Andy Stone told POLITICO in an email, adding, "Organizations that use original art shared on Instagram are expected to have the right to do so. The move by the companies follows a parallel action by Twitter, which on Thursday morning disabled the same video included in a pair of tweets by @TeamTrump and @TrumpWarRoom 2020 campaign accounts, also citing an unspecified complaint under the Digital Millennium Copyright Act. The company left up the tweets but placed an overlay on the videos that reads, This media has been"""
            },
            {
                "role": "assistant",
                "content": "The provided text is 'RELEVANT' to content moderation. This text discusses the removal of video tributes to George Floyd by Facebook and Instagram due to copyright complaints. The text also mentions the similar decision made by Twitter to disable the same video on its platform. These actions relate to content moderation rules and practices, specifically in terms of copyright infringement."
            },
            {
                "role": "user",
                "content": "Now, is the following tweet RELEVANT or IRRELEVANT to content moderation?\n\n{text}"
            }
        ]

        # relevance_specific_promot_paper_2024 = ""

        return [
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (code) news_2020_2021 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with 0 and 1)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": self.content_moderation_relevance_specific_prompt_code
                        },
                        {
                            "role": "user",
                            "content": " here's the tweet I picked, please label it as 'Relevant' or 'Irrelevant' by answering with one word:\n{text}"
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (code) news_2020_2021 relevance (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with 0 and 1)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": self.content_moderation_relevance_specific_prompt_code + " here's the tweet I picked, please label it as 'Relevant' or 'Irrelevant' by answering with one word:\n{text}"
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) news_2020_2021 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with 0 and 1)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": self.content_moderation_relevance_specific_prompt_paper
                        },
                        {
                            "role": "user",
                            "content": "Here's the tweet I picked, please label it as 'Relevant' or 'Irrelevant' by answering with one word:\n{text}"
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) news_2020_2021 relevance (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with 0 and 1)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": self.content_moderation_relevance_specific_prompt_paper + " Here's the tweet I picked, please label it as 'Relevant' or 'Irrelevant' by answering with one word:\n{text}"
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) news_2020_2021 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with A and B)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": relevance_specific_prompt_code_2024_zero[0]
                        },
                        {
                            "role": "user",
                            "content": relevance_specific_prompt_code_2024_zero[1]
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) news_2020_2021 relevance (no system prompt)',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with A and B)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": relevance_specific_prompt_code_2024_zero[0] + "\n\n" + relevance_specific_prompt_code_2024_zero[1]
                        },
                    ],
                },
                {
                    'description': '[original] [few-shot-detailed] gilardi 2024 (code) news_2020_2021 relevance',
                    'compatible_output_mapping': [
                        # 'gilardi relevance', 
                        'gilardi relevance (extended with A and B)'],
                    'prompt_text': relevance_specific_prompt_code_2024_few,
                },
            ]

    def get_all_output_mappings(self):

        return {
            'gilardi relevance': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.relevance_output_mapping},
            },
            'gilardi relevance (extended with 0 and 1)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.relevance_output_mapping_extended_0_and_1},
            },
            'gilardi relevance (extended with A and B)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.relevance_output_mapping_extended_A_and_B},
            },
        }


    def get_dataset_name(self):
        """Get the name of the dataset."""
        return "Gilardi2023_Data4Task1_News_2020_2021"
    
    def load_full_dataset(self, return_statistics_for_plotting=False, return_dataset_used_by_gilardi=False):
        
        df_raw = pd.read_excel("data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/original_sample/annotation_data_newspapers_raw.xlsx")


        # Process for relevance task
        df_relevance = pd.read_csv("data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/mTurk_data/training_data_newsarticles_relevance.csv")
        df = df_relevance.dropna(subset=['relevant_fabio', 'relevant_paula'], how='all')
        df_relevance = df.dropna(subset=['relevant_fabio', 'relevant_paula'])

        # Now drop duplicates
        df_relevance_without_duplicates = df_relevance.drop_duplicates(subset=['id'], keep='first')
        nr_of_duplicates_based_on_id = len(df_relevance) - len(df_relevance_without_duplicates)
        if not return_dataset_used_by_gilardi:
            df_relevance_without_duplicates = df_relevance_without_duplicates.drop_duplicates(subset=['title_h1', 'text_200'], keep='first')
        print(f"\nRows after dropping duplicates: {len(df_relevance_without_duplicates)}")

        # calculate trained annotators agreement
        trained_annotators_agreement = self.intercoder_agreement_percentage(df_relevance_without_duplicates, ['relevant_fabio', 'relevant_paula'])
        trained_annotators_agreement_krippendorff = self.intercoder_agreement_krippendorff(df_relevance_without_duplicates, ['relevant_fabio', 'relevant_paula'])
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement: {trained_annotators_agreement:.4f}')
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement_krippendorff: {trained_annotators_agreement_krippendorff:.4f}')

        df_relevance_with_gt = df_relevance_without_duplicates[
            df_relevance_without_duplicates['relevant_fabio'] == df_relevance_without_duplicates['relevant_paula']
        ]
        df_relevance_with_gt['ground_truth'] = copy.deepcopy(
            df_relevance_with_gt['relevant_fabio']
        )

        nr_of_rows_with_missing_annotations = len(df) - len(df_relevance)
        nr_of_duplicates = len(df_relevance) - len(df_relevance_without_duplicates)
        nr_of_rows_with_disagreement = len(df_relevance_without_duplicates) - len(df_relevance_with_gt)

        dataset_statistics = {
            'Dataset name': self.get_dataset_name(),
            'Total rows': len(df),
            'Total rows (raw)': len(df_raw),
            'Rows without GT NaN values': len(df_relevance),
            'Rows without duplicate values': len(df_relevance_without_duplicates),
            'Trained annotators agreement': trained_annotators_agreement,
            'Trained annotators agreement (krippendorff)': trained_annotators_agreement_krippendorff,
            'Rows with missing annotations': nr_of_rows_with_missing_annotations,
            'Rows with duplicates (based on id)': nr_of_duplicates_based_on_id,
            'Rows with duplicates': nr_of_duplicates,
            'Rows with disagreement': nr_of_rows_with_disagreement,
            'Final dataset size': len(df_relevance_with_gt),
            # 'Rows which all trained annotators agree [relevant=1]': sum(df_relevance_with_gt['ground_truth']==1),
            'Value counts': df_relevance_with_gt['ground_truth'].value_counts(),
        }
        for k, v in dataset_statistics.items():
            print(f"    {k}: {v}")
        
        df_relevance_with_gt.rename(columns={"text_200": "text"}, inplace=True)

        df_relevance_with_gt = self.order_df_columns(df_relevance_with_gt)
        df_relevance_with_gt['ground_truth'] = df_relevance_with_gt['ground_truth'].map({0: 'irrelevant', 1: 'relevant'})

        if return_statistics_for_plotting:
            return df_relevance_with_gt, dataset_statistics
        else:
            return df_relevance_with_gt


    def _get_mturk_file_path(self):
        return 'data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/mTurk_data/batch_results_newsarticles_relevance_final.csv'

    def _get_mturk_delimiter(self):
        return ';'

    def _get_mturk_text_col(self):
        return 'Input.text_200'


class Gilardi2023_Data4Task2_News_2020_2021(Gilardi2023):
    """Implementation for Gilardi 2023 Frame Task."""

    def __init__(self, data_directory='gilardi_et_al_pnas', config_fn='config_framesI_news.yaml'):
        super().__init__(data_directory, config_fn)

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following tweet describes content moderation as a {previous_answer_placeholder}?

Output only a single number between 0 and 1, without any context or explanation.

Tweet: {text}

Probability:'''

    def get_prompts(self):
        # prompt copied from paper Appendix E: https://www.pnas.org/lookup/suppl/doi:10.1073/pnas.2305016120/-/DCSupplemental
        # 
        frame_problem_solution_prompt = "\nContent moderation can be seen from two different perspectives:\n• Content moderation can be seen as a PROBLEM; for example, as a restriction of free speech\n• Content moderation can be seen as a SOLUTION; for example, as a protection from harmful speech\nFor each tweet in the sample, follow these instructions:\n1. Carefully read the text of the tweet, paying close attention to details.\n2. Classify the tweet as describing content moderation as a problem, as a solution, or neither.\nTweets should be classified as describing content moderation as a PROBLEM if they emphasize negative effects of content moderation, such as restrictions to free speech, or the biases that can emerge from decisions regarding what users are allowed to post.\nTweets should be classified as describing content moderation as a SOLUTION if they emphasize positive effects of content moderation, such as protecting users from various kinds of harmful content, including hate speech, misinformation, illegal adult content, or spam.\nTweets should be classified as describing content moderation as NEUTRAL if they do not emphasize possible negative or positive effects of content moderation, for example if they simply report on the content moderation activity of social media platforms without linking them to potential advantages or disadvantages for users or stakeholders."

        gilardi_frame_prompt_code = self.gilardi_content_moderation_prompt_code + \
            frame_problem_solution_prompt
        gilardi_frame_prompt_paper = self.gilardi_content_moderation_prompt_paper + \
            frame_problem_solution_prompt

        tweet_instruction_prompt = "Here's the tweet I picked, please label it as 'Problem', 'Solution', or 'Neutral' by answering with one word:\n{text}"


        # the following prompts are copied from Gilardi 2024 code: data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv
        # available for download at: https://osf.io/adkun/files/osfstorage
        # a=pd.read_csv("data/gilardi_et_al_pnas/data_raw/replication-files-figures/LLM-Comparisons_export/dataset_task_mappings.csv")
        frame_specific_prompt_code_2024_zero = ["“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as describing content moderation as a problem, as a solution, or neither:\n\nA: Text describes content moderation as a PROBLEM if they emphasize negative effects of it, such as restrictions to free speech, censorship, or the biases that can emerge from decisions regarding what users are allowed to post.\n\nB: Text describes content moderation as a SOLUTION if they emphasize positive effects of it, such as protecting users from harmful content such as hate speech, misinformation, illegal adult content, or spam. \n\nC: Text describes content moderation as NEUTRAL if they do not emphasize negative or positive effects of content moderation. For example if they simply report on the content moderation activity of social media platforms without linking them to potential advantages or disadvantages for users or stakeholders.", "Now, is the following tweet describing content moderation as a PROBLEM, as a SOLUTION, or NEUTRAL?\n\n{text}"]


        frame_specific_prompt_code_2024_few = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as describing content moderation as a problem, as a solution, or neither:\n\nA: Text describes content moderation as a PROBLEM if they emphasize negative effects of it, such as restrictions to free speech, censorship, or the biases that can emerge from decisions regarding what users are allowed to post.\n\nB: Text describes content moderation as a SOLUTION if they emphasize positive effects of it, such as protecting users from harmful content such as hate speech, misinformation, illegal adult content, or spam. \n\nC: Text describes content moderation as NEUTRAL if they do not emphasize negative or positive effects of content moderation. For example if they simply report on the content moderation activity of social media platforms without linking them to potential advantages or disadvantages for users or stakeholders.\n\nThe following texts are some examples of 'PROBLEM', 'SOLUTION' or 'NEUTRAL' texts about content moderation in the form of question and answer:"
            },
            {
                "role": "user",
                "content": """Twitter removed a "misleading" tweet downplaying the efficacy of masks posted by a top coronavirus adviser to President Donald Trump, while U.S. cases surged before the Nov. 3 election, Trend reports citing Reuters. As the Trump administration fends off accusations that its mixed messaging on wearing masks hampered the fight against the coronavirus, Dr. Scott Atlas continued to minimize the importance of masks with a Twitter post on Saturday, saying, "Masks work? NO." Twitter Inc removed the tweet on Sunday, saying it violated its misleading information policy on COVID-19, which targets statements that have been confirmed to be false or misleading by subject-matter experts. The White House had no immediate comment on the decision. New infections have been rising fast in the United States, according to a Reuters analysis, with more than 69,400 reported on Friday, up from 46,000 a month ago. Total U.S. cases have surpassed 8 million. Trump, who was hospitalized with the disease for three nights in early October, has been criss-crossing the country in a surge of 11th-hour campaigning as he lags in many public opinion polls. His rallies draw thousands of supporters in close quarters, with many not wearing masks despite federal coronavirus guidelines. Despite data showing otherwise, Trump has said"""
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": """OAKLAND, Calif. - Facebook has banned an extremist anti-government network loosely associated with the broader "boogaloo" movement, a slang term supporters use to refer to a second Civil War or a collapse of civilization. But the platform didn't try to name the group, underscoring the difficulty of grappling with an amorphous network linked to a string of domestic terror plots that appears to obfuscate its existence. Among other complications, its internet-savvy members tend to keep their distance from one another, frequently change their symbols and catch phrases and mask their intentions with sarcasm. The move by Facebook designates this group as a dangerous organization similar to the Islamic State group and white supremacists, both of which are already banned from its service. The social network is not banning all references to "boogaloo" and said it is only removing groups, accounts and pages when they have a "clear connection to violence or a credible threat to public safety." The loose movement is named after "Breakin' 2: Electric Boogaloo," a 1984 sequel to a movie about breakdancing. "Boogaloo" supporters have shown up at protests over COVID-19 lockdown orders, carrying rifles and wearing tactical gear over Hawaiian shirts - themselves a reference to "big luau," a"""
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Florida Governor Ron DeSantis announced this week that he would fine social media companies that ban political candidates. Every outlet from Fox News to MSNBC fired off missives about the bill. What got lost in the news coverage is that Silicon Valley deplatforms very few politicians, save shock-jocks like Donald Trump and Laura Loomer (if you want to call her a politician). The same cannot be said for sex workers. This month, Centro University released a study estimating that 46 percent of adult influencers reported losing access to Twitter or Instagram in the last year. The bans put a permanent dent in the stars’ income, with Centro estimating sex workers lose $260 million a year due to social media bans. You won’t hear DeSantis, Fox News, Glenn Greenwald, or any other so-called free speech warriors decrying porn stars’ lost incomes, so let me break down how social media companies are screwing over porn stars (and not screwing them in a good way!). Silicon Valley titans have revoked my social media access multiple times. Take my recent Snapchat ban. The Santa Monica-based app barred me from posting on my public account, so I lost the means to communicate with fans who would"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": """TALLAHASSEE – Gov. Ron DeSantis' call for punishing social media sites that deplatformed former President Donald Trump narrowly cleared a Senate committee Monday and soon will be ready for a full vote in the Legislature. Sen. Jeff Brandes, R-St. Petersburg, was the lone Republican who argued against the proposal by fellow Republican Sen. Ray Rodrigues of Naples. Brandes labeled it a "big government bill." "This Senate is currently filled with small government Republicans who do believe that government shouldn't be in the lives of businesses," Brandes said. He added: "This is the exact opposite of the things that we stand for." But Rodrigues argued back that the measure doesn't defy free market principles. The bill (SB 7072) orders social media companies to publish standards with detailed definitions of when someone would be censored or blocked, and makes companies subject to as much as $100,000 fines for deplatforming a Florida candidate. "I'm bringing you good policy supported by your constituents," Rodrigues said. The measure was approved 10-9 by the Appropriations Committee, its last stop before going to the Senate floor. A similar measure is ready for a full House vote. State and federal courts have generally taken a hands-off view involving regulating online platforms. Congress also has not"""
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "A scathing new report released by hedge fund Hindenburg Research claims that start-up Nikola is an 'intricate fraud' based on years of lies and fake products•Hindenburg claimed to have 'extensive evidence' that the company's proprietary technology was purchased from another company•The fund also accused Nikola's founder Trevor Milton of making countless false statements over the last decade and faking a product promotional video•When filming the Nikola One ad, the truck reportedly didn't have an engine•So the company reportedly rolled the prototype along a downhill stretch of a highway and filmed it as if it was being driven•In a tweet, Milton called the report a 'hit job' and asked the public to 'give me a few hours to put together responses to their lies'•Nikola's stock prices plummeted as much as 13 percent Thursday A scathing new report released by hedge fund Hindenburg Research claims that start-up Nikola is an 'intricate fraud' based on years of lies and fake products Hindenburg claimed to have 'extensive evidence' that the company's proprietary technology was purchased from another company The fund also accused Nikola's founder Trevor Milton of making countless false statements over the last decade and faking a product promotional video When filming the Nikola One ad, the"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": """Some of the toughest ads against Donald Trump are driven by lifelong Republicans unable to stomach the direction their party has taken. Washington: Rick Wilson apologises for running late for our phone interview: it's been a frantic morning for the veteran Republican ad-maker and his colleagues at the Lincoln Project. The anti-Trump group has just released its latest advertisement, slamming the US President for suggesting that the November 3 election may need to be delayed. In the half hour since the ad ??? titled We Will Vote ??? went live, it has already racked up more than 250,000 views online. That's nothing unusual for the operatives at the Lincoln Project, who have been pumping out attack ads at a prolific rate over recent months. "We push really fast all the time," Wilson says. "We drive ourselves and our team very hard because we think we are pursuing a worthwhile endeavour and we know it works." The group's co-founders include Steve Schmidt, who ran Republican nominee John McCain's 2008 campaign, and conservative lawyer George Conway, the husband of top Trump aide Kellyanne Conway. Having spent most of their adult lives working to get Republicans elected, they are now producing some of the toughest anti-Trump ads on"""
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Now, is the following tweet describing content moderation as a PROBLEM, as a SOLUTION, or NEUTRAL?\n\n{text}"
            }
        ]
        
        frame_specific_prompt_code_2024_cot = [
            {
                "role": "system",
                "content": "“Content moderation” refers to the practice of screening and monitoring content posted by users on social media sites to determine if the content should be published or not, based on specific rules and guidelines.\n\nI will ask you to classify a tweet as describing content moderation as a problem, as a solution, or neither:\n\nA: Text describes content moderation as a PROBLEM if they emphasize negative effects of it, such as restrictions to free speech, censorship, or the biases that can emerge from decisions regarding what users are allowed to post.\n\nB: Text describes content moderation as a SOLUTION if they emphasize positive effects of it, such as protecting users from harmful content such as hate speech, misinformation, illegal adult content, or spam. \n\nC: Text describes content moderation as NEUTRAL if they do not emphasize negative or positive effects of content moderation. For example if they simply report on the content moderation activity of social media platforms without linking them to potential advantages or disadvantages for users or stakeholders.\n\nThe following texts are some examples of 'PROBLEM', 'SOLUTION' or 'NEUTRAL' texts about content moderation in the form of question and answer:"
            },
            {
                "role": "user",
                "content": """Twitter removed a "misleading" tweet downplaying the efficacy of masks posted by a top coronavirus adviser to President Donald Trump, while U.S. cases surged before the Nov. 3 election, Trend reports citing Reuters. As the Trump administration fends off accusations that its mixed messaging on wearing masks hampered the fight against the coronavirus, Dr. Scott Atlas continued to minimize the importance of masks with a Twitter post on Saturday, saying, "Masks work? NO." Twitter Inc removed the tweet on Sunday, saying it violated its misleading information policy on COVID-19, which targets statements that have been confirmed to be false or misleading by subject-matter experts. The White House had no immediate comment on the decision. New infections have been rising fast in the United States, according to a Reuters analysis, with more than 69,400 reported on Friday, up from 46,000 a month ago. Total U.S. cases have surpassed 8 million. Trump, who was hospitalized with the disease for three nights in early October, has been criss-crossing the country in a surge of 11th-hour campaigning as he lags in many public opinion polls. His rallies draw thousands of supporters in close quarters, with many not wearing masks despite federal coronavirus guidelines. Despite data showing otherwise, Trump has said"""
            },
            {
                "role": "assistant",
                "content": "The provided text can be classified as describing content moderation as a SOLUTION. It highlights how Twitter removed a tweet that downplayed the efficacy of masks, stating that it violated their policy on misleading information related to COVID-19. This demonstrates content moderation as a means to address and mitigate the spread of false or misleading information. By removing the tweet, Twitter aims to protect users from harmful content and ensure accurate information is shared regarding public health during the pandemic."
            },
            {
                "role": "user",
                "content": """OAKLAND, Calif. - Facebook has banned an extremist anti-government network loosely associated with the broader "boogaloo" movement, a slang term supporters use to refer to a second Civil War or a collapse of civilization. But the platform didn't try to name the group, underscoring the difficulty of grappling with an amorphous network linked to a string of domestic terror plots that appears to obfuscate its existence. Among other complications, its internet-savvy members tend to keep their distance from one another, frequently change their symbols and catch phrases and mask their intentions with sarcasm. The move by Facebook designates this group as a dangerous organization similar to the Islamic State group and white supremacists, both of which are already banned from its service. The social network is not banning all references to "boogaloo" and said it is only removing groups, accounts and pages when they have a "clear connection to violence or a credible threat to public safety." The loose movement is named after "Breakin' 2: Electric Boogaloo," a 1984 sequel to a movie about breakdancing. "Boogaloo" supporters have shown up at protests over COVID-19 lockdown orders, carrying rifles and wearing tactical gear over Hawaiian shirts - themselves a reference to "big luau," a"""
            },
            {
                "role": "assistant",
                "content": """Based on the provided text, it can be classified as describing content moderation as a SOLUTION. The text highlights Facebook's action of banning an extremist anti-government network associated with the "boogaloo" movement, which is linked to domestic terror plots. Facebook's move is presented as designating the group as a dangerous organization, similar to the Islamic State group and white supremacists, and removing groups, accounts, and pages with a clear connection to violence or a credible threat to public safety. This portrays content moderation as a measure to protect public safety and prevent the spread of extremist content."""
            },
            {
                "role": "user",
                "content": "Florida Governor Ron DeSantis announced this week that he would fine social media companies that ban political candidates. Every outlet from Fox News to MSNBC fired off missives about the bill. What got lost in the news coverage is that Silicon Valley deplatforms very few politicians, save shock-jocks like Donald Trump and Laura Loomer (if you want to call her a politician). The same cannot be said for sex workers. This month, Centro University released a study estimating that 46 percent of adult influencers reported losing access to Twitter or Instagram in the last year. The bans put a permanent dent in the stars’ income, with Centro estimating sex workers lose $260 million a year due to social media bans. You won’t hear DeSantis, Fox News, Glenn Greenwald, or any other so-called free speech warriors decrying porn stars’ lost incomes, so let me break down how social media companies are screwing over porn stars (and not screwing them in a good way!). Silicon Valley titans have revoked my social media access multiple times. Take my recent Snapchat ban. The Santa Monica-based app barred me from posting on my public account, so I lost the means to communicate with fans who would"
            },
            {
                "role": "assistant",
                "content": "Based on the provided text, it can be classified as describing content moderation as a PROBLEM. The text criticizes social media companies for their content moderation practices, particularly regarding the ban on sex workers and adult influencers. It suggests that these bans have negative consequences, such as the loss of income for sex workers, and highlights the perceived lack of concern or attention from politicians and free speech advocates. The text emphasizes the negative effects of content moderation, including restrictions on speech and the impact on individuals' livelihoods, indicating that it views content moderation as a problem."
            },
            {
                "role": "user",
                "content": """TALLAHASSEE – Gov. Ron DeSantis' call for punishing social media sites that deplatformed former President Donald Trump narrowly cleared a Senate committee Monday and soon will be ready for a full vote in the Legislature. Sen. Jeff Brandes, R-St. Petersburg, was the lone Republican who argued against the proposal by fellow Republican Sen. Ray Rodrigues of Naples. Brandes labeled it a "big government bill." "This Senate is currently filled with small government Republicans who do believe that government shouldn't be in the lives of businesses," Brandes said. He added: "This is the exact opposite of the things that we stand for." But Rodrigues argued back that the measure doesn't defy free market principles. The bill (SB 7072) orders social media companies to publish standards with detailed definitions of when someone would be censored or blocked, and makes companies subject to as much as $100,000 fines for deplatforming a Florida candidate. "I'm bringing you good policy supported by your constituents," Rodrigues said. The measure was approved 10-9 by the Appropriations Committee, its last stop before going to the Senate floor. A similar measure is ready for a full House vote. State and federal courts have generally taken a hands-off view involving regulating online platforms. Congress also has not"""
            },
            {
                "role": "assistant",
                "content": """Based on the provided text, it can be classified as describing content moderation as a PROBLEM. The text highlights the debate and disagreement surrounding a proposed bill that aims to punish social media sites for deplatforming former President Donald Trump. Senator Jeff Brandes argues against the proposal, labeling it a "big government bill" and stating that it goes against the principles of small government Republicans. This indicates a negative view of content moderation, particularly in terms of government intervention and potential restrictions on businesses. While Senator Ray Rodrigues argues in favor of the bill, the emphasis in the text is on the opposition and criticism, suggesting a portrayal of content moderation as a problem."""
            },
            {
                "role": "user",
                "content": "A scathing new report released by hedge fund Hindenburg Research claims that start-up Nikola is an 'intricate fraud' based on years of lies and fake products•Hindenburg claimed to have 'extensive evidence' that the company's proprietary technology was purchased from another company•The fund also accused Nikola's founder Trevor Milton of making countless false statements over the last decade and faking a product promotional video•When filming the Nikola One ad, the truck reportedly didn't have an engine•So the company reportedly rolled the prototype along a downhill stretch of a highway and filmed it as if it was being driven•In a tweet, Milton called the report a 'hit job' and asked the public to 'give me a few hours to put together responses to their lies'•Nikola's stock prices plummeted as much as 13 percent Thursday A scathing new report released by hedge fund Hindenburg Research claims that start-up Nikola is an 'intricate fraud' based on years of lies and fake products Hindenburg claimed to have 'extensive evidence' that the company's proprietary technology was purchased from another company The fund also accused Nikola's founder Trevor Milton of making countless false statements over the last decade and faking a product promotional video When filming the Nikola One ad, the"
            },
            {
                "role": "assistant",
                "content": "The provided text does not explicitly discuss content moderation or its effects. It focuses on a scathing report released by Hindenburg Research that accuses the startup company Nikola of fraud and making false statements. The text does not emphasize the positive or negative effects of content moderation, nor does it link the topic to potential advantages or disadvantages. Therefore, the text can be classified as NEUTRAL regarding the description of content moderation."
            },
            {
                "role": "user",
                "content": """Some of the toughest ads against Donald Trump are driven by lifelong Republicans unable to stomach the direction their party has taken. Washington: Rick Wilson apologises for running late for our phone interview: it's been a frantic morning for the veteran Republican ad-maker and his colleagues at the Lincoln Project. The anti-Trump group has just released its latest advertisement, slamming the US President for suggesting that the November 3 election may need to be delayed. In the half hour since the ad ??? titled We Will Vote ??? went live, it has already racked up more than 250,000 views online. That's nothing unusual for the operatives at the Lincoln Project, who have been pumping out attack ads at a prolific rate over recent months. "We push really fast all the time," Wilson says. "We drive ourselves and our team very hard because we think we are pursuing a worthwhile endeavour and we know it works." The group's co-founders include Steve Schmidt, who ran Republican nominee John McCain's 2008 campaign, and conservative lawyer George Conway, the husband of top Trump aide Kellyanne Conway. Having spent most of their adult lives working to get Republicans elected, they are now producing some of the toughest anti-Trump ads on"""
            },
            {
                "role": "assistant",
                "content": "The provided text describes the Lincoln Project, an anti-Trump group that releases attack ads against the US President. While the text does not explicitly discuss content moderation, it focuses on the activities and efforts of the Lincoln Project in creating and disseminating ads. It does not emphasize the positive or negative effects of content moderation or link it to potential advantages or disadvantages. Therefore, the text can be classified as NEUTRAL regarding the description of content moderation."
            },
            {
                "role": "user",
                "content": "Now, is the following tweet describing content moderation as a PROBLEM, as a SOLUTION, or NEUTRAL?\n\n{text}"
            }
        ]

        return [
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (code) news_2020_2021 frame',
                    'compatible_output_mapping': ['gilardi frame'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": gilardi_frame_prompt_code
                        },
                        {
                            "role": "user",
                            "content": tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (code) news_2020_2021 frame (no system prompt)',
                    'compatible_output_mapping': ['gilardi frame'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": gilardi_frame_prompt_code + "\n" + tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) news_2020_2021 frame',
                    'compatible_output_mapping': ['gilardi frame'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": gilardi_frame_prompt_paper
                        },
                        {
                            "role": "user",
                            "content": tweet_instruction_prompt
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2023 (paper) news_2020_2021 frame (no system prompt)',
                    'compatible_output_mapping': ['gilardi frame'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": gilardi_frame_prompt_paper + "\n" + tweet_instruction_prompt

                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) news_2020_2021 frame',
                    'compatible_output_mapping': [
                        'gilardi frame (extended with A and B and C)'],
                    'prompt_text': [
                        {
                            "role": "system",
                            "content": frame_specific_prompt_code_2024_zero[0]
                        },
                        {
                            "role": "user",
                            "content": frame_specific_prompt_code_2024_zero[1]
                        }
                    ],
                },
                {
                    'description': '[original] [zero-shot-detailed] gilardi 2024 (code) news_2020_2021 frame (no system prompt)',
                    'compatible_output_mapping': [
                        'gilardi frame (extended with A and B and C)'],
                    'prompt_text': [
                        {
                            "role": "user",
                            "content": frame_specific_prompt_code_2024_zero[0] + "\n\n" + frame_specific_prompt_code_2024_zero[1]
                        },
                    ],
                },
                {
                    'description': '[original] [few-shot-detailed] gilardi 2024 (code) news_2020_2021 frame',
                    'compatible_output_mapping': [
                        'gilardi frame (extended with A and B and C)'],
                    'prompt_text': frame_specific_prompt_code_2024_few,
                },
            ]

    def get_all_output_mappings(self):

        return {
            'gilardi frame': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.frame_output_mapping},
            },
            'gilardi frame (extended with A and B and C)': {
                'function': self.default_output_mapping,
                'params': {'mapping_options': self.frame_output_mapping_extended_A_and_B_and_C},
            },
        }


    def get_dataset_name(self):
        """Get the name of the dataset."""
        return "Gilardi2023_Data4Task2_News_2020_2021"

    def load_full_dataset(self, return_statistics_for_plotting=False, return_dataset_used_by_gilardi=False):
        """Load and prepare the Gilardi 2023 Frame dataset."""

        df_raw = pd.read_excel("data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/original_sample/annotation_data_newspapers_raw.xlsx")


        # Process for relevance task
        df_relevant = pd.read_csv("data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/mTurk_data/training_data_newsarticles_problem_solution_final.csv")
        df = df_relevant.dropna(subset=['relevant_fabio', 'relevant_paula'], how='all')
        df_relevant = df_relevant.dropna(subset=['relevant_fabio', 'relevant_paula'])

        # Filter to only relevant tweets and process for frame task
        df_relevant = df_relevant[
            (df_relevant['relevant_fabio'] == 1) &
            (df_relevant['relevant_paula'] == 1)
        ]

        # Now drop duplicates
        df_frame_without_duplicates = df_relevant.drop_duplicates(subset=['id'], keep='first')
        nr_of_duplicates_based_on_id = len(df_relevant) - len(df_frame_without_duplicates)
        if not return_dataset_used_by_gilardi:
            df_frame_without_duplicates = df_relevant.drop_duplicates(subset=['title_h1', 'text_200'], keep='first')
        print(f"\nRows after dropping duplicates: {len(df_frame_without_duplicates)}")


        df_frame = df_frame_without_duplicates.dropna(
            subset=['problem_frame_fabio', 'problem_frame_paula',
                    'solution_frame_fabio', 'solution_frame_paula']
        )

        df_frame_not_both = df_frame[(
            ~(
                ((df_frame['problem_frame_fabio'] == 1) & (df_frame['solution_frame_fabio'] == 1)) |
                ((df_frame['problem_frame_paula'] == 1) & (df_frame['solution_frame_paula'] == 1))
            )
        )]

        # Apply frame classification functions
        df_frame_not_both['ground_truth_fabio'] = df_frame_not_both.apply(
            lambda row: self.get_frame_gt(row, 'fabio'), axis=1
        )
        df_frame_not_both['ground_truth_paula'] = df_frame_not_both.apply(
            lambda row: self.get_frame_gt(row, 'paula'), axis=1
        )

        # calculate trained annotators agreement
        trained_annotators_agreement = self.intercoder_agreement_percentage(df_frame_not_both, ['ground_truth_fabio', 'ground_truth_paula'])
        trained_annotators_agreement_krippendorff = self.intercoder_agreement_krippendorff(df_frame_not_both, ['ground_truth_fabio', 'ground_truth_paula'])
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement: {trained_annotators_agreement:.4f}')
        print('Annotator agreement:', self.get_dataset_name(), f'trained_annotators_agreement_krippendorff: {trained_annotators_agreement_krippendorff:.4f}')

        # Keep only rows where both annotators agree
        df_frame_with_gt = df_frame_not_both[
            (df_frame_not_both['ground_truth_fabio'] ==
                df_frame_not_both['ground_truth_paula'])
        ]
        df_frame_with_gt['ground_truth'] = copy.deepcopy(
            df_frame_with_gt['ground_truth_fabio']
        )

        nr_of_duplicates = len(df_relevant) - len(df_frame_without_duplicates)
        nr_of_rows_with_missing_annotations = len(df_frame_without_duplicates) - len(df_frame)
        nr_of_rows_with_invalid_annotations = len(df_frame) - len(df_frame_not_both)
        nr_of_rows_with_disagreement = len(df_frame_not_both) - len(df_frame_with_gt)

        dataset_statistics = {
            'Dataset name': self.get_dataset_name(),
            'Total rows (raw)': len(df_raw),
            'Total rows': len(df),
            'Total relevant rows': len(df_relevant),
            'Rows without duplicate values': len(df_frame_without_duplicates),
            'Rows without GT NaN values': len(df_frame),
            'Rows without both values (problem and solution)': len(df_frame_not_both),
            'Trained annotators agreement': trained_annotators_agreement,
            'Trained annotators agreement (krippendorff)': trained_annotators_agreement_krippendorff,
            'Rows with duplicates (based on id)': nr_of_duplicates_based_on_id,
            'Rows with duplicates': nr_of_duplicates,
            'Rows with missing annotations': nr_of_rows_with_missing_annotations,
            'Rows with invalid annotations': nr_of_rows_with_invalid_annotations,
            'Rows with disagreement': nr_of_rows_with_disagreement,
            'Final dataset size': len(df_frame_with_gt),
            'Value counts': df_frame_with_gt['ground_truth'].value_counts(),
        }
        for k, v in dataset_statistics.items():
            print(f"    {k}: {v}")

        df_frame_with_gt.rename(columns={"text_200": "text"}, inplace=True)

        df_frame_with_gt = self.order_df_columns(df_frame_with_gt)

        if return_statistics_for_plotting:
            return df_frame_with_gt, dataset_statistics
        else:
            return df_frame_with_gt


    def _get_mturk_file_path(self):
        return 'data/gilardi_et_al_pnas/data_raw/dataverse_repo/data/mTurk_data/batch_results_newsarticles_problem_solution_final.csv'

    def _get_mturk_delimiter(self):
        return ';'

    def _get_mturk_text_col(self):
        return 'Input.text_200'
