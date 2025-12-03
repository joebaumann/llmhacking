# Instructions to preprocess *factuality*

- Download the data (only the file `data.zip` is needed) from [https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I?usp=sharing](https://drive.google.com/drive/folders/1kFey69z8hGXScln01mVxrOhrqgM62X7I?usp=sharing) and unzip it into [data/factuality/data_raw](data/factuality/data_raw).
- Then preprocess the data and save it as [all_data_processed_full/factuality.csv](all_data_processed_full/factuality.csv) with:
```
python -m data.factuality.preprocess_data
```