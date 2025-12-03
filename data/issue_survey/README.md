# Instructions to preprocess *issue_survey*

- Download the sav files from [https://www.britishelectionstudy.com/data-object/british-election-study-combined-wave-1-26-internet-panel-open-ended-response-data/](https://www.britishelectionstudy.com/data-object/british-election-study-combined-wave-1-26-internet-panel-open-ended-response-data/) and [https://www.britishelectionstudy.com/data-object/british-election-study-combined-wave-1-29-internet-panel/](https://www.britishelectionstudy.com/data-object/british-election-study-combined-wave-1-29-internet-panel/) and save them in [data/issue_survey/data_raw](data/issue_survey/data_raw).
- Then preprocess the data with:
```
python -m data.issue_survey.preprocess_data
```