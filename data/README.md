
# Data Directory

The [data/all_data_processed/](data/all_data_processed/) directory contains all annotation tasks used in our LLM hacking study. Each task includes preprocessed text data with ground truth annotations from human experts (or crowdworkers).


### Data Structure

Each dataset is a CSV file with the following columns:

| Column | Description | Example Values |
|--------|-------------|----------------|
| `id` | Unique identifier (format: `{task}_{index}`) | `emotion_0`, `tone_142` |
| `ground_truth` | Expert/crowdworker annotation | `joy`, `Left`, `True`, `positive` |
| `text` | The text to be annotated | Full text of tweet, article, or document |
| **Metadata** | Task-specific additional columns | `author`, `date`, `subreddit`, `topic`, etc. |

Metadata columns are used to create data groupings for hypothesis testing.

### Loading Data Example

```python
from data.data_utils import map_dataset_name_to_class

# Load any task
data_loader = map_dataset_name_to_class("emotion")
dataset = data_loader.load_dataset()

print(dataset.columns)  # See all available columns
print(dataset['ground_truth'].value_counts())  # See label distribution
```

## Downloading Proprietary Data

For **7 tasks**, we cannot directly share the data due to licensing restrictions. You must register and download them yourself. Here's how to obtain the final preprocessed data we used for our experiments:
1. Essay tasks (`essay_domestic`, `essay_housewife`, `essay_living`, `essay_location`, `essay_narrative`, `essay_worksocial`):
    a. Download the data from [https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=5790](https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=5790) (free access granted after creating a UK Data Service account) and then unzip the data in [data/essay/data_raw](data/essay/data_raw).
    b. Then run the following script to get the preprocessed data: `python -m data.essay.load_proprietary_data_and_merge_with_redacted_data`.
2. Issue_survey task:
    a. Download the sav files from [https://www.britishelectionstudy.com/data-object/british-election-study-combined-wave-1-26-internet-panel-open-ended-response-data/](https://www.britishelectionstudy.com/data-object/british-election-study-combined-wave-1-26-internet-panel-open-ended-response-data/) and [https://www.britishelectionstudy.com/data-object/british-election-study-combined-wave-1-29-internet-panel/](https://www.britishelectionstudy.com/data-object/british-election-study-combined-wave-1-29-internet-panel/) and save them in [data/issue_survey/data_raw](data/issue_survey/data_raw).
    b. Then run the following script to get the preprocessed data: `python -m data.issue_survey.load_proprietary_data_and_merge_with_redacted_data`. This takes some minutes, as the data is very large...


## Replicating Original Data Preprocessing

If you want to recreate our preprocessing from scratch, each task directory contains:

- `README.md` - Instructions for downloading raw data
- `preprocess_data.py` or `preprocess.R` - Preprocessing scripts
Additionally, each task directory contains:
- `load_data_and_prompts.py` - Data loading and prompt definitions
- `config.yaml` - Task metadata information


### Preprocessing Pipeline

Our preprocessing consists of two stages:
1. Task-specific preprocessing:
    - Loads raw data from original sources
    - Applies task-specific transformations
    - Saves to `data/all_data_processed_full/{task}.csv`

2. Deduplication and downsampling:
    - `python -m data.prepare_final_deduplicated_data`
    - Running this script removes duplicate texts and applies stratified sampling for very large datasets (>10,000 samples), while preserving class distributions and metadata distributions
    - The final datasets are saved to `data/all_data_processed/{task}.csv`

## Task Directories

Detailed preprocessing instructions for each task:

- [`data/emotion`](emotion/) - Reddit emotion classification (26 classes)
- [`data/essay`](essay/) - Essay coding
    - contains 6 separate tasks: essay_domestic, essay_housewife, essay_living, essay_location, essay_narrative, essay_worksocial
- [`data/factuality`](factuality/) - Fact-checking LLM outputs
- [`data/fakenews`](fakenews/) - Fake news detection
- [`data/gilardi_et_al_pnas`](gilardi_et_al_pnas/) - Several political tasks
    - contains 11 separate tasks: framesI_news, framesI_tweets, framesI_tweets23, framesII_tweets, framesII_tweets17, relevance_news, relevance_tweets, relevance_tweets17, relevance_tweets23, stance_tweets, topic_tweets
- [`data/hatespeech`](hatespeech/) - Hate speech detection
    - contains 3 separate tasks: hatespeech_explicit, hatespeech_implicit, hatespeech_target
- [`data/humor`](humor/) - Humor detection
- [`data/ideology_news`](ideology_news/) - News article ideology
- [`data/ideology_tweets`](ideology_tweets/) - Tweet ideology
- [`data/issue_survey`](issue_survey/) - Political issue categorization
- [`data/manifestos`](manifestos/) - Political manifesto analysis
- [`data/manifestos_uk`](manifestos_uk/) - UK manifesto analysis
    - contains 3 separate tasks: manifestos_econ_ideology, manifestos_issue, manifestos_social_ideology
- [`data/misinfo`](misinfo/) - Misinformation detection
- [`data/politeness`](politeness/) - Politeness classification
    - contains 2 separate tasks: politeness_stack, politeness_wiki
- [`data/stance_climate`](stance_climate/) - Climate change stance
- [`data/tone`](tone/) - Political ad tone
- [`data/topic`](topic/) - Congressional bill topics
