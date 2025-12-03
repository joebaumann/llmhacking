# Instructions to preprocess *politeness_wiki* and *politeness_stack*

- Download from `http://www.cs.cornell.edu/~cristian/Politeness_files/Stanford_politeness_corpus.zip`
- Unzip into: [data_raw](data/data_raw)
- Download data processed by Gligoric et al. (2025) from [https://drive.google.com/drive/folders/1n4lRhggbpMTHB0eQB3QKtf6te2ngKc6D?usp=sharing](https://drive.google.com/drive/folders/1n4lRhggbpMTHB0eQB3QKtf6te2ngKc6D?usp=sharing). Unzip it and place the `politeness_dataset.csv` file in [data_raw](data/data_raw).
- Then preprocess the data and save it as [all_data_processed_full/politeness.csv](all_data_processed_full/politeness.csv) with:
```
python -m data.politeness.politeness_wiki.preprocess_data
python -m data.politeness.politeness_stack.preprocess_data
```