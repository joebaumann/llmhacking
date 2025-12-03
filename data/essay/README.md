# Instructions to preprocess all *essay_\** tasks

- Download the data from [https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=5790](https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=5790) (free access granted after creating a UK Data Service account) and then unzip the data in [data/essay/data_raw](data/essay/data_raw).
- Then preprocess the data with:
```
python -m data.essay.preprocess_data
```
