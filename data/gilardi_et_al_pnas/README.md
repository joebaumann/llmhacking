# Instructions to preprocess all datasets and tasks by Gilardi et al. (2023)

Download 2023 PNAS paper data from `https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PQYF6M`.
Place it in [data/gilardi_et_al_pnas/data_raw](data/gilardi_et_al_pnas/data_raw) and unzip.

- Then preprocess all 4 dataset and 11 tasks and save it in [all_data_processed_full](all_data_processed_full) with:
```
python -m data.gilardi_et_al_pnas.preprocess_data
```

## Prompts:

We also use some prompts from their 2024 paper. This data can be downloaded from `https://osf.io/adkun/files/osfstorage`.