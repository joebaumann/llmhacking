from sklearn.metrics import accuracy_score, f1_score as sklearn_f1_score
import numpy as np
import pandas as pd


def get_annotations_from_dfs(df_ground_truth, df_llm_annotated):
    ground_truth_annotations = df_ground_truth['ground_truth'].astype(str).str.strip().str.lower()
    llm_annotations = df_llm_annotated['response_mapped'].astype(str).str.strip().str.lower()
    return ground_truth_annotations, llm_annotations

# aggregate metrics 
def accuracy(ground_truth_annotations, llm_annotations):
    return accuracy_score(ground_truth_annotations, llm_annotations)

def f1_score(ground_truth_annotations, llm_annotations):
    return sklearn_f1_score(ground_truth_annotations, llm_annotations, average='macro')

def f1_score_weighted(ground_truth_annotations, llm_annotations):
    return sklearn_f1_score(ground_truth_annotations, llm_annotations, average='weighted')

def fraction_of_answers_that_did_not_follow_instructions(llm_annotations):
    return (llm_annotations == 'na').mean()

# metrics for downstream analysis conclusions
def is_correct_conclusion(combined_downstream_analysis):
    return combined_downstream_analysis['conclusion_gt'] == combined_downstream_analysis['conclusion_llm']

def is_type_I_error(combined_downstream_analysis):
    # Type I: LLM finds significance when ground truth doesn't
    gt_not_sig = combined_downstream_analysis['conclusion_gt'] == 'no_difference'
    llm_sig = 'larger' in combined_downstream_analysis['conclusion_llm']
    return int(gt_not_sig and llm_sig)

def is_type_II_error(combined_downstream_analysis):
    # Type II: LLM doesn't find significance when ground truth does
    gt_sig = 'larger' in combined_downstream_analysis['conclusion_gt']
    llm_not_sig = combined_downstream_analysis['conclusion_llm'] == 'no_difference'
    return int(gt_sig and llm_not_sig)

def is_type_S_error(combined_downstream_analysis):
    # Type S: Both find significance but in opposite directions
    gt_sig = 'larger' in combined_downstream_analysis['conclusion_gt']
    llm_sig = 'larger' in combined_downstream_analysis['conclusion_llm']
    opposite_direction = combined_downstream_analysis['conclusion_gt'] != combined_downstream_analysis['conclusion_llm']
    return int(gt_sig and llm_sig and opposite_direction)

def downstream_conclusion_effect_error(combined_downstream_analysis):
    # Actual difference (not absolute)
    return combined_downstream_analysis['effect_size_llm'] - combined_downstream_analysis['effect_size_gt']

def downstream_conclusion_effect_error_absolute(combined_downstream_analysis):
    # Absolute difference
    return abs(combined_downstream_analysis['effect_size_llm'] - combined_downstream_analysis['effect_size_gt'])

def annotation_reliability_abs_diff(list_of_performance_metrics):
    return max(list_of_performance_metrics) - min(list_of_performance_metrics)

def annotation_reliability_variance(list_of_performance_metrics):
    return np.var(list_of_performance_metrics)

def annotation_reliability_stdev(list_of_performance_metrics):
    return np.std(list_of_performance_metrics)
