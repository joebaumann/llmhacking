# Instructions to preprocess *hatespeech_implicit*, *hatespeech_explicit*, and *hatespeech_target*

- Download the data from [https://www.dropbox.com/s/24meryhqi1oo0xk/implicit-hate-corpus.zip?dl=0](https://www.dropbox.com/s/24meryhqi1oo0xk/implicit-hate-corpus.zip?dl=0) and unzip it into [data/hatespeech/data_raw](data/hatespeech/data_raw).
- Then preprocess the data with:
```
python -m data.hatespeech.preprocess_data
```