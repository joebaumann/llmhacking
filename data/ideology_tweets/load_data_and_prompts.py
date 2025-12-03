from data.data_utils import MyDataLoader
import copy


class IdeologyTweetsDataset(MyDataLoader):
    def __init__(self, data_directory='ideology_tweets', config_fn='config.yaml'):
        super().__init__(data_directory, config_fn)

    def get_all_output_mappings(self):
        dict_with_mapping_options = {
            "Australian Labor Party": "left",  # australia
            "Liberal Party of Australia": "right",  # australia
            "The Liberal Party": "right",  # dk
            "The Social Democratic Party": "left",  # dk
            "CDU": "right",  # germany
            "SPD": "left",  # germany
            "Labour Party": "left",  # nz
            "National Party": "right",  # nz
            "Civic Coalition": "left",  # poland, relatively left?
            "Law and Justice": "right",  # poland
            "Moderates": "right",  # sweden
            "Social Democrats": "left",  # sweden
            "AKP": "right",  # turkey
            "CHP": "left",  # turkey
            "Labour": "left",  # uk
            "Democrat": "left",  # us
            "Republican": "right",  # us
            "Conservative": "right",  # canada and uk
            "Liberal": "left",  # canada
        }
        dict_with_mapping_options_letters = {
            'A': 'left',
            'B': 'right',
            "Australian Labor Party": "left",  # australia
            "Liberal Party of Australia": "right",  # australia
            "The Liberal Party": "right",  # dk
            "The Social Democratic Party": "left",  # dk
            "CDU": "right",  # germany
            "SPD": "left",  # germany
            "Labour Party": "left",  # nz
            "National Party": "right",  # nz
            "Civic Coalition": "left",  # poland, relatively left?
            "Law and Justice": "right",  # poland
            "Moderates": "right",  # sweden
            "Social Democrats": "left",  # sweden
            "AKP": "right",  # turkey
            "CHP": "left",  # turkey
            "Labour": "left",  # uk
            "Democrat": "left",  # us
            "Republican": "right",  # us
            "Conservative": "right",  # canada and uk
            "Liberal": "left",  # canada
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
        gender_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'gender')
        country_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'country')
        region_groupings = self.split_by_column_with_n_most_frequent_values(
            df, 'region')
        return gender_groupings + country_groupings + region_groupings

    def get_cols_to_stratify(self):
        return [
            'ground_truth',
            ('gender', 10),
            ('country', 10),
            ('region', 10),
        ]

    def format_prompt(self, prompt_text, data):
        assert 'text' in data, f"'text' column is not in data. All columns: {data.columns}"
        # check if of type list
        formatted_prompt = copy.deepcopy(prompt_text)
        formatted_prompt = formatted_prompt[-1]['content'][data['country']]
        formatted_prompt = formatted_prompt.format(**data)
        return formatted_prompt

    def get_prompt_for_verbalized_confidence_elicitation(self):
        return '''How likely is it that the following tweet was authored by a {previous_answer_placeholder} politician?

Output only a single number between 0 and 1, without any context or explanation.

Tweet: {text}

Probability:'''

    def get_prompts(self):
        prompt_templates = {
            "AUSTRALIA": "You will be given a set of Twitter posts from different Australian politicians, sent during the two months preceding the 2019 Australian election. Your task is to use your knowledge of Australian politics to make an educated guess on whether the poster is a Australian Labor Party or Liberal Party of Australia. Respond either 'Australian Labor Party' or 'Liberal Party of Australia'. If the message does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "CANADA": "You will be given a set of Twitter posts from different Canadian politicians, sent during the two months preceding the 2021 Canadian election. Your task is to use your knowledge of Canadian politics to make an educated guess on whether the poster is a Liberal or Conservative. Respond either 'Liberal' or 'Conservative'. If the message does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "DENMARK": "You will be given a set of Twitter posts from different Danish politicians, sent during the two months preceding the 2019 Danish election. Your task is to use your knowledge of Danish politics to make an educated guess on whether the poster is a The Social Democratic Party or The Liberal Party. Respond either 'The Social Democratic Party' or 'The Liberal Party'. If the message does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "GERMANY": "You will be given a set of Twitter posts from different German politicians, sent during the two months preceding the 2021 German election. Your task is to use your knowledge of German politics to make an educated guess on whether the poster is a CDU or SPD. Respond either 'CDU' or 'SPD'. If the message does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "NZ": "You will be given a set of Twitter posts from different New Zealand politicians, sent during the two months preceding the 2020 New Zealand election. Your task is to use your knowledge of New Zealand politics to make an educated guess on whether the poster is a Labour Party or National Party. Respond either 'Labour Party' or 'National Party'. If the message does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "POLAND": "You will be given a set of Twitter posts from different Polish politicians, sent during the two months preceding the 2020 Polish election. Your task is to use your knowledge of Polish politics to make an educated guess on whether the poster is a Civic Coalition or Law and Justice. Respond either 'Civic Coalition' or 'Law and Justice'. If the message does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "SWEDEN": "You will be given a set of Twitter posts from different Swedish politicians, sent during the two months preceding the 2018 Swedish election. Your task is to use your knowledge of Swedish politics to make an educated guess on whether the poster is a Social Democrats or Moderates. Respond either 'Social Democrats' or 'Moderates'. If the message does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "TURKEY": "You will be given a set of Twitter posts from different Turkish politicians, sent during the two months preceding the 2018 Turkish election. Your task is to use your knowledge of Turkish politics to make an educated guess on whether the poster is a AKP or CHP. Respond either 'AKP' or 'CHP'. If the message does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "UK": "You will be given a set of Twitter posts from different UK politicians, sent during the two months preceding the 2019 UK election. Your task is to use your knowledge of UK politics to make an educated guess on whether the poster is a Conservative or Labour. Respond either 'Conservative' or 'Labour'. If the message does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "US": "You will be given a set of Twitter posts from different US politicians, sent during the two months preceding the 2020 US election. Your task is to use your knowledge of US politics to make an educated guess on whether the poster is a Democrat or Republican. Respond either 'Democrat' or 'Republican'. If the message does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
        }
        prompt_templates_with_letters = {
            "AUSTRALIA": "You will be given a Twitter post from an Australian politician, sent during the two months preceding the 2019 Australian election. Your task is to use your knowledge of Australian politics to make an educated guess on whether the author of the tweet is a (A) Australian Labor Party or (B) Liberal Party of Australia. Respond either 'A' or 'B'. If the tweet does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "CANADA": "You will be given a Twitter post from a Canadian politician, sent during the two months preceding the 2021 Canadian election. Your task is to use your knowledge of Canadian politics to make an educated guess on whether the author of the tweet is a (A) Liberal or (B) Conservative. Respond either 'A' or 'B'. If the tweet does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "DENMARK": "You will be given a Twitter post from a Danish politician, sent during the two months preceding the 2019 Danish election. Your task is to use your knowledge of Danish politics to make an educated guess on whether the author of the tweet is a (A) The Social Democratic Party or (B) The Liberal Party. Respond either 'A' or 'B'. If the tweet does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "GERMANY": "You will be given a Twitter post from a German politician, sent during the two months preceding the 2021 German election. Your task is to use your knowledge of German politics to make an educated guess on whether the author of the tweet is a (A) SPD or (B) CDU. Respond either 'A' or 'B'. If the tweet does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "NZ": "You will be given a Twitter post from a New Zealand politicians, sent during the two months preceding the 2020 New Zealand election. Your task is to use your knowledge of New Zealand politics to make an educated guess on whether the author of the tweet is a (A) Labour Party or (B) National Party. Respond either 'A' or 'B'. If the tweet does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "POLAND": "You will be given a Twitter post from a Polish politician, sent during the two months preceding the 2020 Polish election. Your task is to use your knowledge of Polish politics to make an educated guess on whether the author of the tweet is a (A) Civic Coalition or (B) Law and Justice. Respond either 'A' or 'B'. If the tweet does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "SWEDEN": "You will be given a Twitter post from a Swedish politician, sent during the two months preceding the 2018 Swedish election. Your task is to use your knowledge of Swedish politics to make an educated guess on whether the author of the tweet is a (A) Social Democrats or (B) Moderates. Respond either 'A' or 'B'. If the tweet does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "TURKEY": "You will be given a Twitter post from a Turkish politician, sent during the two months preceding the 2018 Turkish election. Your task is to use your knowledge of Turkish politics to make an educated guess on whether the author of the tweet is a (A) CHP or (B) AKP. Respond either 'A' or 'B'. If the tweet does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "UK": "You will be given a Twitter post from a UK politician, sent during the two months preceding the 2019 UK election. Your task is to use your knowledge of UK politics to make an educated guess on whether the author of the tweet is a (A) Labour or (B) Conservative. Respond either 'A' or 'B'. If the tweet does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
            "US": "You will be given a Twitter post from a US politician, sent during the two months preceding the 2020 US election. Your task is to use your knowledge of US politics to make an educated guess on whether the author of the tweet is a (A) Democrat or (B) Republican. Respond either 'A' or 'B'. If the tweet does not have enough information for an educated guess, just make your best guess.\n\nTweet: {text}",
        }

        return [
            {
                'description': '[original] [zero-shot-short]',
                'compatible_output_mapping': [
                    'default output mapping'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # using a custom prompt formatter to replicate Törnberg (2024)
                        "content": prompt_templates
                    }
                ],
            },
            {
                'description': '[original] [zero-shot-short] with letters',
                'compatible_output_mapping': [
                    'default output mapping letters'
                ],
                'prompt_text': [
                    {
                        "role": "user",
                        # using a custom prompt formatter to replicate Törnberg (2024)
                        "content": prompt_templates_with_letters
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
                        "content": {
                            "AUSTRALIA": "Below is a tweet from an Australian politician posted in the two-month run-up to the 2019 Australian election. Using your knowledge of Australia’s party landscape, decide whether the author is from the Australian Labor Party or the Liberal Party of Australia. Reply with exactly one of the two strings: 'Australian Labor Party' or 'Liberal Party of Australia'. If the tweet is unclear, make the most informed choice you can.\n\nTweet: {text}",
                            "CANADA": "Below is a tweet from a Canadian politician posted in the two-month run-up to the 2021 Canadian election. Using your knowledge of Canada’s party landscape, decide whether the author is Liberal or Conservative. Reply with exactly one of the two strings: 'Liberal' or 'Conservative'. If the tweet is unclear, make the most informed choice you can.\n\nTweet: {text}",
                            "DENMARK": "Below is a tweet from a Danish politician posted in the two-month run-up to the 2019 Danish election. Using your knowledge of Denmark’s party landscape, decide whether the author is from The Social Democratic Party or The Liberal Party. Reply with exactly one of the two strings: 'The Social Democratic Party' or 'The Liberal Party'. If the tweet is unclear, make the most informed choice you can.\n\nTweet: {text}",
                            "GERMANY": "Below is a tweet from a German politician posted in the two-month run-up to the 2021 German election. Using your knowledge of Germany’s party landscape, decide whether the author is CDU or SPD. Reply with exactly one of the two strings: 'CDU' or 'SPD'. If the tweet is unclear, make the most informed choice you can.\n\nTweet: {text}",
                            "NZ": "Below is a tweet from a New Zealand politician posted in the two-month run-up to the 2020 New Zealand election. Using your knowledge of New Zealand’s party landscape, decide whether the author is from the Labour Party or the National Party. Reply with exactly one of the two strings: 'Labour Party' or 'National Party'. If the tweet is unclear, make the most informed choice you can.\n\nTweet: {text}",
                            "POLAND": "Below is a tweet from a Polish politician posted in the two-month run-up to the 2020 Polish election. Using your knowledge of Poland’s party landscape, decide whether the author is from Civic Coalition or Law and Justice. Reply with exactly one of the two strings: 'Civic Coalition' or 'Law and Justice'. If the tweet is unclear, make the most informed choice you can.\n\nTweet: {text}",
                            "SWEDEN": "Below is a tweet from a Swedish politician posted in the two-month run-up to the 2018 Swedish election. Using your knowledge of Sweden’s party landscape, decide whether the author is from the Social Democrats or the Moderates. Reply with exactly one of the two strings: 'Social Democrats' or 'Moderates'. If the tweet is unclear, make the most informed choice you can.\n\nTweet: {text}",
                            "TURKEY": "Below is a tweet from a Turkish politician posted in the two-month run-up to the 2018 Turkish election. Using your knowledge of Turkey’s party landscape, decide whether the author is AKP or CHP. Reply with exactly one of the two strings: 'AKP' or 'CHP'. If the tweet is unclear, make the most informed choice you can.\n\nTweet: {text}",
                            "UK": "Below is a tweet from a UK politician posted in the two-month run-up to the 2019 UK election. Using your knowledge of the UK’s party landscape, decide whether the author is Conservative or Labour. Reply with exactly one of the two strings: 'Conservative' or 'Labour'. If the tweet is unclear, make the most informed choice you can.\n\nTweet: {text}",
                            "US": "Below is a tweet from a US politician posted in the two-month run-up to the 2020 US election. Using your knowledge of the United States’ party landscape, decide whether the author is Democrat or Republican. Reply with exactly one of the two strings: 'Democrat' or 'Republican'. If the tweet is unclear, make the most informed choice you can.\n\nTweet: {text}"
                        }
                    }
                ],
            },
            {
                'description': '[paraphrase-nr-1] orig:[orig*] [zero-shot-short] with letters',
                'original_description': '[original] [zero-shot-short] with letters',
                'compatible_output_mapping': ['default output mapping letters'],
                'prompt_text': [
                    {
                        "role": "user",
                        "content": {
                            'AUSTRALIA': "Below is a tweet authored by an Australian politician in the two months leading up to the 2019 Australian election. Using your knowledge of Australian politics, identify whether it was written by (A) the Australian Labor Party or (B) the Liberal Party of Australia. Reply with ONLY 'A' or 'B'. If the tweet is not conclusive, choose the option that seems more likely.\n\nTweet: {text}",
                            'CANADA': "Below is a tweet from a Canadian politician sent within two months of the 2021 Canadian election. Draw on your understanding of Canadian politics to decide whether the author belongs to (A) the Liberal Party or (B) the Conservative Party. Respond strictly with 'A' or 'B'. If uncertain, make your best guess.\n\nTweet: {text}",
                            'DENMARK': "Here is a tweet by a Danish politician posted during the two months preceding the 2019 Danish election. Based on your knowledge of Danish politics, determine whether the writer is from (A) The Social Democratic Party or (B) The Liberal Party. Output only 'A' or 'B'. If evidence is limited, choose the more plausible option.\n\nTweet: {text}",
                            'GERMANY': "Here is a tweet from a German politician published in the two months before the 2021 German election. Using your understanding of German politics, indicate whether the poster represents (A) SPD or (B) CDU. Return only 'A' or 'B'. If clues are insufficient, select the answer that seems most likely.\n\nTweet: {text}",
                            'NZ': "Below is a tweet by a New Zealand politician sent within two months of the 2020 New Zealand election. With your knowledge of New Zealand politics, decide whether the author is affiliated with (A) the Labour Party or (B) the National Party. Reply solely with 'A' or 'B'. If the tweet is ambiguous, choose the more probable option.\n\nTweet: {text}",
                            'POLAND': "Here is a tweet from a Polish politician posted during the two months leading up to the 2020 Polish election. Using your familiarity with Polish politics, judge whether the author supports (A) Civic Coalition or (B) Law and Justice. Respond only with 'A' or 'B'. If the information is insufficient, provide your best guess.\n\nTweet: {text}",
                            'SWEDEN': "Below is a tweet written by a Swedish politician in the two months preceding the 2018 Swedish election. Based on your knowledge of Swedish politics, state whether the tweet comes from (A) Social Democrats or (B) Moderates. Output strictly 'A' or 'B'. If details are lacking, select the more likely choice.\n\nTweet: {text}",
                            'TURKEY': "Here is a tweet by a Turkish politician from the two months before the 2018 Turkish election. Using your understanding of Turkish politics, determine whether the author is from (A) CHP or (B) AKP. Return only 'A' or 'B'. If unsure, make the best possible guess.\n\nTweet: {text}",
                            'UK': "Below is a tweet from a UK politician dated within two months of the 2019 UK election. Drawing on your knowledge of UK politics, decide whether the author belongs to (A) Labour or (B) Conservative. Reply with ONLY 'A' or 'B'. If evidence is sparse, choose the more plausible answer.\n\nTweet: {text}",
                            'US': "Here is a tweet from a US politician posted in the two months preceding the 2020 US election. Using your understanding of US politics, identify whether the author is a member of (A) the Democrat Party or (B) the Republican Party. Respond strictly with 'A' or 'B'. If the tweet lacks clear signals, select the option that seems most likely.\n\nTweet: {text}"
                        }
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
                        "content": {
                            'AUSTRALIA': "Here is a tweet published by an Australian politician within the two months leading up to the 2019 Australian election. Using your familiarity with Australian politics, decide if the author is from the Australian Labor Party or the Liberal Party of Australia. Reply with only one of the following: 'Australian Labor Party' or 'Liberal Party of Australia'. If the tweet is ambiguous, still choose the party you think is likeliest.\n\nTweet: {text}",
                            'CANADA': "Below is a tweet posted by a Canadian politician during the two months before the 2021 Canadian election. Drawing on your knowledge of Canadian politics, determine whether the writer is Liberal or Conservative. Answer exclusively with either 'Liberal' or 'Conservative'. If the information is insufficient, give your best guess.\n\nTweet: {text}",
                            'DENMARK': "You are shown a tweet from a Danish politician sent in the two-month run-up to the 2019 Danish election. Based on your understanding of Danish politics, infer whether the tweeter belongs to The Social Democratic Party or The Liberal Party. Respond with exactly 'The Social Democratic Party' or 'The Liberal Party'. If details are scarce, select the party you consider most probable.\n\nTweet: {text}",
                            'GERMANY': "The following tweet was issued by a German politician in the two months preceding the 2021 German election. Use your political knowledge of Germany to judge whether the author is CDU or SPD. Return only 'CDU' or 'SPD'. When evidence is limited, provide the choice you deem more likely.\n\nTweet: {text}",
                            'NZ': "Here is a tweet from a New Zealand politician during the two months before the 2020 New Zealand election. Decide, using your knowledge of NZ politics, whether the poster is from the Labour Party or the National Party. Reply solely with 'Labour Party' or 'National Party'. If unsure, offer your most informed guess.\n\nTweet: {text}",
                            'POLAND': "Consider the tweet below, published by a Polish politician in the two months prior to the 2020 Polish election. From your understanding of Polish politics, assess whether the author represents Civic Coalition or Law and Justice. Output only 'Civic Coalition' or 'Law and Justice'. If the tweet lacks clear clues, state the party you find more plausible.\n\nTweet: {text}",
                            'SWEDEN': "A Swedish politician posted the following tweet during the two months before the 2018 Swedish election. Using your knowledge of Swedish politics, decide if the individual is affiliated with Social Democrats or Moderates. Respond with exactly 'Social Democrats' or 'Moderates'. If the content is inconclusive, still choose the party you think fits best.\n\nTweet: {text}",
                            'TURKEY': "Below is a tweet from a Turkish politician in the two months leading up to the 2018 Turkish election. With your awareness of Turkish politics, infer whether the author is from AKP or CHP. Answer only 'AKP' or 'CHP'. When information is sparse, provide the option you judge most likely.\n\nTweet: {text}",
                            'UK': "You are given a tweet written by a UK politician during the two months preceding the 2019 UK election. Apply your knowledge of UK politics to determine if the author is Conservative or Labour. Reply strictly with 'Conservative' or 'Labour'. If evidence is limited, choose the party you consider more probable.\n\nTweet: {text}",
                            'US': "The tweet below was sent by a US politician in the two months before the 2020 US election. Based on your understanding of US politics, decide whether the poster is Democrat or Republican. Respond only with 'Democrat' or 'Republican'. If the message is ambiguous, give your best-reasoned guess.\n\nTweet: {text}"
                            }
                    }
                ],
            },
        ]
