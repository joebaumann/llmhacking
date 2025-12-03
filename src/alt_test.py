# code taken from https://github.com/nitaytech/AltTest
import pdb
import json
import os
import numpy as np
from scipy.stats import ttest_1samp
from typing import Any, Optional, List, Tuple, Dict, Union, Callable

# %%
def open_json(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def by_procedure(p_values: List[float], q: float) -> List[int]:
    p_values = np.array(p_values, dtype=float)
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    # Compute the harmonic sum H_m = 1 + 1/2 + ... + 1/m
    H_m = np.sum(1.0 / np.arange(1, m + 1))
    # Compute the BY thresholds for each rank i
    by_thresholds = (np.arange(1, m + 1) / m) * (q / H_m)
    max_i = -1
    for i in range(m):
        if sorted_pvals[i] <= by_thresholds[i]:
            max_i = i
    if max_i == -1:
        return []
    rejected_sorted_indices = sorted_indices[:max_i + 1]
    return list(rejected_sorted_indices)


def accuracy(pred: Any, annotations: List[Any]) -> float:
    return float(np.mean([pred == ann for ann in annotations]))


def neg_rmse(pred: Union[int, float], annotations: List[Union[int, float]]) -> float:
    return -1 * float(np.sqrt(np.mean([(pred - ann) ** 2 for ann in annotations])))


def sim(pred: str, annotations: List[str], similarity_func: Callable) -> float:
    return float(np.mean([similarity_func(pred, ann) for ann in annotations]))


def ttest(indicators, epsilon: float) -> float:
    return ttest_1samp(indicators, epsilon, alternative='less').pvalue


def alt_test(llm_annotations: Dict[Union[int, str], Any],
             humans_annotations: Dict[Union[int, str], Dict[Union[int, str], Any]],
             scoring_function: Union[str, Callable] = 'accuracy',
             epsilon: float = 0.2,
             q_fdr: float = 0.05,
             min_humans_per_instance: int = 2,
             min_instances_per_human: int = 30):
    # prepare alignment scoring function
    if isinstance(scoring_function, str):
        if scoring_function == 'accuracy':
            scoring_function = accuracy
        elif scoring_function == 'neg_rmse':
            scoring_function = neg_rmse
        else:
            raise ValueError("Unknown scoring function")
    else:
        scoring_function = scoring_function

    # prepare sets - i_set has humans as keys, h_set has instances as keys
    i_set, h_set = {}, {}
    for h, anns in humans_annotations.items():
        i_set[h] = list(anns.keys())
        for i, ann in anns.items():
            if i not in h_set:
                h_set[i] = []
            h_set[i].append(h)

    # remove instances with less than min_humans_per_instance
    instances_to_keep = {i for i in h_set if len(h_set[i]) >= min_humans_per_instance and i in llm_annotations}
    if len(instances_to_keep) < len(h_set):
        print(f"Dropped {len(h_set) - len(instances_to_keep)} instances with less than {min_humans_per_instance} annotators.")
    i_set = {h: [i for i in i_set[h] if i in instances_to_keep] for h in i_set}
    h_set = {i: h_set[i] for i in h_set if i in instances_to_keep}

    p_values, advantage_probs, humans = [], [], []
    for excluded_h in humans_annotations:
        llm_indicators = []
        excluded_indicators = []
        instances = [i for i in i_set[excluded_h] if i in llm_annotations]
        if len(instances) < min_instances_per_human:
            print(f"Skipping annotator {excluded_h} with only {len(instances)} instances < {min_instances_per_human}.")
            continue

        for i in instances:
            human_ann = humans_annotations[excluded_h][i]
            llm_ann = llm_annotations[i]
            remaining_anns = [humans_annotations[h][i] for h in h_set[i] if h != excluded_h]
            human_score = scoring_function(human_ann, remaining_anns)
            llm_score = scoring_function(llm_ann, remaining_anns)
            llm_indicators.append(1 if llm_score >= human_score else 0)
            excluded_indicators.append(1 if human_score >= llm_score else 0)

        diff_indicators = [exc_ind - llm_ind for exc_ind, llm_ind in zip(excluded_indicators, llm_indicators)]
        p_values.append(ttest(diff_indicators, epsilon))
        advantage_probs.append(float(np.mean(llm_indicators)))
        humans.append(excluded_h)

    rejected_indices = by_procedure(p_values, q_fdr)
    advantage_prob = float(np.mean(advantage_probs))
    winning_rate = len(rejected_indices) / len(humans)
    return winning_rate, advantage_prob

def format_llm_annotations(datasets_llm_annotated, gt_ids, model, prompt):

    # - **`llm_annotations`**:
    #   A dictionary where the keys represent instances (instance ids) and the values represent LLM predictions.
    #   **Example:**
    #   ```python
    #   {'instance1': 'A', 'instance2': 'B', 'instance3': 'A'}
    #   ```

    # prepare llm annotations . use id column for dict keys. ensure each prediction is of type str and lowercase.

    llm_data = datasets_llm_annotated[
        (datasets_llm_annotated['model'] == model) &
        (datasets_llm_annotated['prompt_description'] == prompt) &
        # (datasets_llm_annotated['temperature'] == temperature) &
        # (datasets_llm_annotated['seed'] == seed) &
        (datasets_llm_annotated['id'].isin(gt_ids))
    ]

    llm_data = llm_data.sort_values('response_mapped', key=lambda x: x.str.lower() == 'na').drop_duplicates(subset='id', keep='first')
    
    llm_ids = set(llm_data['id'])
    llm_data_is_complete = (gt_ids == llm_ids) and (len(gt_ids) == len(llm_data))
    
    if not llm_data_is_complete:
        print(f'Skipping incomplete LLM data: model={model}, prompt={prompt}')
        pdb.set_trace()

    # Create dictionary with id as keys and lowercase response_mapped as values
    llm_annotations = {
        row['id']: row['response_mapped'].lower() 
        for _, row in llm_data.iterrows()
    }
    return llm_annotations

def run_alt_test(task, data_loader, dataset, model, datasets_llm_annotated, gt_ids, prompt, metric='accuracy', epsilon=0.1):
    humans_annotations = data_loader.load_human_annotations_for_alt_test(dataset)
    if humans_annotations is not None:
        llm_annotations = format_llm_annotations(datasets_llm_annotated, gt_ids, model, prompt)
        wr, ap = alt_test(llm_annotations, humans_annotations, metric, epsilon=epsilon)
        print(f"task:{task} model:{model} prompt:{prompt} [{'PASSED' if wr >= 0.5 else 'FAILED'}]:\tWinning Rate={wr:.2f}\tAdvantage Probability={ap:.2f}")
        return wr, ap
    else:
        return None, None