# Instructions to preprocess *fakenews*

- Download data by cloning the following two repositories:
```
cd data/fakenews/data_raw
git clone git@github.com:KaiDMML/FakeNewsNet.git
git clone git@github.com:jiayingwu19/SheepDog.git
```
- Then preprocess the data and save it as [all_data_processed_full/fakenews.csv](all_data_processed_full/fakenews.csv) with:
```
python -m data.fakenews.preprocess_data
```