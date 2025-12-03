# Instructions to preprocess *emotion*

- Download the data as follows:
```
cd data/emotion/data_raw
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
```
- Then preprocess the data with:
```
python -m data.emotion.preprocess_data
```

## Notes:
- The [data/emotion/preprocess_data.py](data/emotion/preprocess_data.py) file is loaded from [https://github.com/google-research/google-research/blob/master/goemotions/data/sentiment_dict.json](https://github.com/google-research/google-research/blob/master/goemotions/data/sentiment_dict.json).
- The [data/emotion/ekman_mapping.json](data/emotion/ekman_mapping.json) file is loaded from [https://github.com/google-research/google-research/blob/master/goemotions/data/ekman_mapping.json](https://github.com/google-research/google-research/blob/master/goemotions/data/ekman_mapping.json).