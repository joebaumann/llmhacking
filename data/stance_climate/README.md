# Instructions to preprocess *stance_climate*

- Download data by cloning the repository:
```
cd data/stance_climate/data_raw
git clone git@github.com:yiweiluo/GWStance.git
```
- Then preprocess the data and save it as [all_data_processed_full/stance_climate.csv](all_data_processed_full/stance_climate.csv) with:
```
python -m data.stance_climate.preprocess_data
```