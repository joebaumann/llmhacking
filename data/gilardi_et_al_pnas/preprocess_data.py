
from data.gilardi_et_al_pnas.load_data_and_prompts import *

dir_path = 'all_data_processed_full'

data_loader = Gilardi2023_Data1Task1_Tweets_2020_2021()
df = data_loader.load_full_dataset(return_dataset_used_by_gilardi=True) # return full dataset used by gilardi inclduing duplicates, these will be filtered out later
data_loader.save_final_dataset_to_disk(df, processed_data_dir=dir_path) # ('relevance_tweets',)

data_loader = Gilardi2023_Data1Task2_Tweets_2020_2021()
df = data_loader.load_full_dataset(return_dataset_used_by_gilardi=True) # return full dataset used by gilardi inclduing duplicates, these will be filtered out later
data_loader.save_final_dataset_to_disk(df, processed_data_dir=dir_path) # ('framesI_tweets',)

data_loader = Gilardi2023_Data1Task3Tweets_2020_2021()
df = data_loader.load_full_dataset(return_dataset_used_by_gilardi=True) # return full dataset used by gilardi inclduing duplicates, these will be filtered out later
data_loader.save_final_dataset_to_disk(df, processed_data_dir=dir_path) # ('framesII_tweets',)

data_loader = Gilardi2023_Data1Task4Tweets_2020_2021()
df = data_loader.load_full_dataset(return_dataset_used_by_gilardi=True) # return full dataset used by gilardi inclduing duplicates, these will be filtered out later
data_loader.save_final_dataset_to_disk(df, processed_data_dir=dir_path) # ('stance_tweets',)

data_loader = Gilardi2023_Data1Task5Tweets_2020_2021()
df = data_loader.load_full_dataset(return_dataset_used_by_gilardi=True) # return full dataset used by gilardi inclduing duplicates, these will be filtered out later
data_loader.save_final_dataset_to_disk(df, processed_data_dir=dir_path) # ('topic_tweets',)

data_loader = Gilardi2023_Data2Task1_Tweets2023Relevance()
df = data_loader.load_full_dataset(return_dataset_used_by_gilardi=True) # return full dataset used by gilardi inclduing duplicates, these will be filtered out later
data_loader.save_final_dataset_to_disk(df, processed_data_dir=dir_path) # ('relevance_tweets23',)

data_loader = Gilardi2023_Data2Task2_Tweets2023Frame()
df = data_loader.load_full_dataset(return_dataset_used_by_gilardi=True) # return full dataset used by gilardi inclduing duplicates, these will be filtered out later
data_loader.save_final_dataset_to_disk(df, processed_data_dir=dir_path) # ('framesI_tweets23',)

data_loader = Gilardi2023_Data3Task1_CongressTweets_2017_2022_Relevance()
df = data_loader.load_full_dataset(return_dataset_used_by_gilardi=True) # return full dataset used by gilardi inclduing duplicates, these will be filtered out later
data_loader.save_final_dataset_to_disk(df, processed_data_dir=dir_path) # ('relevance_tweets17',)

data_loader = Gilardi2023_Data3Task2_CongressTweets_2017_2022PoliticalFrame()
df = data_loader.load_full_dataset(return_dataset_used_by_gilardi=True) # return full dataset used by gilardi inclduing duplicates, these will be filtered out later
data_loader.save_final_dataset_to_disk(df, processed_data_dir=dir_path) # ('framesII_tweets17',)

data_loader = Gilardi2023_Data4Task1_News_2020_2021()
df = data_loader.load_full_dataset(return_dataset_used_by_gilardi=True) # return full dataset used by gilardi inclduing duplicates, these will be filtered out later
data_loader.save_final_dataset_to_disk(df, processed_data_dir=dir_path) # ('relevance_news',)

data_loader = Gilardi2023_Data4Task2_News_2020_2021()
df = data_loader.load_full_dataset(return_dataset_used_by_gilardi=True) # return full dataset used by gilardi inclduing duplicates, these will be filtered out later
data_loader.save_final_dataset_to_disk(df, processed_data_dir=dir_path) # ('framesI_news',)
