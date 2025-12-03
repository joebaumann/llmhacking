# Large Language Model Hacking: Quantifying Hidden Risks in Text Annotation

[![arXiv](https://img.shields.io/badge/arXiv-2509.08825-b31b1b.svg)](https://arxiv.org/abs/2509.08825)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2509.08825-blue.svg)](https://doi.org/10.48550/arXiv.2509.08825)

This repository contains code and data for the paper **"Large Language Model Hacking: Quantifying the Hidden Risks of Using LLMs for Text Annotation"**. We investigate the reliability of scientific conclusions when using LLMs as annotators for social science research.

## What is LLM Hacking?

**LLM hacking** refers to systematic errors in downstream statistical analyses that occur when using LLM annotations instead of human annotations. These errors include:
- **Type I errors**: Finding statistical significance where none exists in ground truth
- **Type II errors**: Missing true statistical relationships
- **Type S errors**: Detecting effects in the opposite direction

Through our large-scale analysis of over **13 million annotations** across 37 tasks, we demonstrate that:
- Even highly accurate LLM annotators lead to incorrect research conclusions in **31-50% of cases**
- **Intentional LLM hacking** is strikingly feasible: false positives can be fabricated for 94.4% of null hypotheses, while true effects can be hidden in 98.1% of cases
- Effect directions can be reversed entirely in 68.3% of cases (Type S errors)

## Repository Structure
```
.
├── data/                       # Data processing and task definitions
│   ├── all_data_processed/    # Final preprocessed datasets
│   └── [task_name]/           # Individual task directories
└── src/                        # Source code
    ├── llm_data_annotator.py         # LLM annotation generation
    ├── evaluating_results.py         # Results evaluation
    └── llm_hacking_mitigations/     # Mitigation techniques (DSL, CDI)
```

## Setup

### Full Installation (with R support)

R is required for running regressions and the DSL (Design-based Supervised Learning) debiasing technique.
```bash
# Create conda environment with Python and R
conda create -n llmhacking python=3.12 R=4.3 -y
conda activate llmhacking

# Install R-Python interface
conda install -c conda-forge rpy2

# Install R packages
R --no-save -e "install.packages('devtools')"
R --no-save -e "library(devtools); install_github('naoki-egami/dsl', dependencies = TRUE)"

# Install Python dependencies
pip install -r requirements.txt
```

### Simple Setup (Python only)

If you only need to run LLM annotations without statistical analysis:
```bash
conda create -n llmhacking python=3.12 -y
conda activate llmhacking
pip install -r requirements.txt
```

## Annotation Tasks

We replicate **37 social science annotation tasks** from prior work, covering diverse domains.

See the [paper appendix](https://arxiv.org/abs/2509.08825) for detailed task descriptions.

## Data

### Loading Data
```python
from data.data_utils import map_dataset_name_to_class

# Load a specific task
task_name = "emotion"
data_loader = map_dataset_name_to_class(task_name)
dataset = data_loader.load_dataset()

# Display dataset structure
print(f"Dataset shape: {dataset.shape}")
print(f"Columns: {list(dataset.columns)}")
print(dataset.head())
```

### Data Structure

Each dataset contains the following core columns:

| Column | Description | Example |
|--------|-------------|---------|
| `id` | Unique identifier for each text | `emotion_42` |
| `ground_truth` | Human expert annotations | `joy`, `Left`, `True` |
| `text` | The text to be annotated | `"This is amazing!"` |

Additional metadata columns vary by task and may include:
- **Temporal**: `date`, `created_utc`, `Timestamp`
- **Source**: `author`, `subreddit`, `source`, `domain`
- **Content**: `topic`, `sentiment`, `title`

See [data/README.md](data/README.md) for more details.

## Prompts

Each task uses multiple prompt variants to test annotation robustness:
```python
from data.data_utils import map_dataset_name_to_class

# Load prompts for a task
data_loader = map_dataset_name_to_class("emotion")
prompts = data_loader.get_prompts()

# Inspect prompt structure
for prompt in prompts:
    print(f"Description: {prompt['description']}")
    print(f"Compatible mappings: {prompt['compatible_output_mapping']}")
    print(f"Prompt text: {prompt['prompt_text'][:100]}...")  # First 100 chars
    print("-" * 50)
```

### Formatting Prompts with Data
```python
# Format a prompt with actual text
datapoint = dataset.iloc[0]
formatted_prompt = data_loader.format_prompt(
    prompt_text=prompts[0]['prompt_text'],
    data=datapoint
)
print(formatted_prompt)
```

## Hypotheses and Groupings

Our analysis tests how LLM annotation errors affect research conclusions by testing realistic hypotheses (with ground truth annotations and then also with LLM-generated annotations):
```python
from data.data_utils import map_dataset_name_to_class

# Get data groupings for hypothesis testing
data_loader = map_dataset_name_to_class("emotion")
dataset = data_loader.load_dataset()

# Generate all groupings (default + task-specific)
groupings, n_default, n_specific = data_loader.get_groups(dataset)

print(f"Total groupings: {len(groupings)}")
print(f"Default groupings: {n_default}")
print(f"Task-specific groupings: {n_specific}")

# Example: Split by keyword
grouping_name, split_function, split_args = groupings[0]
group1, group2, g1_name, g2_name = split_function(dataset, **split_args)
print(f"\n{grouping_name}:")
print(f"  {g1_name}: {len(group1)} samples")
print(f"  {g2_name}: {len(group2)} samples")
```

### Running Statistical Tests
```python
from src.statistical_downstream_analysis import log_regression_R_glm

# Run regression comparing two groups
conclusions = log_regression_R_glm(
    group1=group1,
    group2=group2,
    gt_classes=dataset['ground_truth'].unique(),
    using_ground_truth_annotations=True
)

for result in conclusions:
    print(f"Class: {result['class_name']}")
    print(f"Conclusion: {result['conclusion']}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Effect size: {result['effect_size']:.4f}")
```

## Full Evaluation Pipeline

### 1. Generate LLM Annotations
```bash
# Annotate a single task with one model
python -m src.llm_data_annotator \
    --task_names "emotion" \
    --models "meta-llama/Llama-3.1-8B-Instruct" \
    --seed 42 \
    --results_folder "results_FINAL"
```

### 2. Evaluate Annotation Quality
```bash
# Evaluate LLM annotations against ground truth
python -m src.evaluating_results \
    --num-cpus 4 \
    --task_names "emotion" \
    --models "meta-llama/Llama-3.1-8B-Instruct" \
    --results_folder "results_FINAL"
```

## Citation
```bibtex
@article{baumann2025llmhacking,
  title={Large Language Model Hacking: Quantifying the Hidden Risks of Using LLMs for Text Annotation},
  author={Baumann, Joachim and R{\"o}ttger, Paul and Urman, Aleksandra and Wendsj{\"o}, Albert and Plaza-del-Arco, Flor Miriam and Gruber, Johannes B and Hovy, Dirk},
  journal={arXiv preprint arXiv:2509.08825},
  year={2025}
}
```

## Contact

For questions or issues, please contact: joachimbaumann1@gmail.com
