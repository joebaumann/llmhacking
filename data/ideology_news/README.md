# Instructions to preprocess *ideology_news*

- Download data by cloning the repository:
```
cd data/ideology_news/data_raw
git clone git@github.com:ramybaly/Article-Bias-Prediction.git
```
- Then preprocess the data and save it as [all_data_processed_full/ideology_news.csv](all_data_processed_full/ideology_news.csv) with:
```
python -m data.ideology_news.preprocess_data
```