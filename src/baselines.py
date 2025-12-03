# run as: python -m src.baselines
import pdb
from tqdm import tqdm
import pandas as pd
from pathlib import Path

import argparse
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

import numpy as np
from data.data_utils import map_dataset_name_to_class, all_tasks
from src.metrics import *
from utils import get_experiment_start_date
from src.helpers import load_confidence_results
from src.statistical_downstream_analysis import all_statistical_tests
from src.helpers import load_llm_annotated_data, binarize_labels
from src.llm_hacking_mitigations.cdi_utils import train_tree, tree_predict, inv_hessian_col_imputed
from src.evaluating_results import assess_llm_hacking
from src.statistical_downstream_analysis_and_hacking_mitigation_helpers import *
from src.metrics import f1_score_weighted
from src.generate_results_tables_and_plots import empirical_llm_hacking_risk, empirical_type_I_risk, empirical_type_II_risk, empirical_type_S_risk

parser = argparse.ArgumentParser()
parser.add_argument('--task_names', type=str, help='Comma-separated list of tasks to run (e.g., "emotion,humor")')
args = parser.parse_args()
if args.task_names:
    all_tasks = args.task_names.split(',')


def get_conclusion(group1, group2, class_name, col_name, significance_treshold=0.05):
    try:
        if set(group1[col_name].unique()) == {0,1}:
            # Calculate counts and proportions, use sum cause labels are already binarized
            group1_positive = group1[col_name].sum()
            group2_positive = group2[col_name].sum()
        else:
            # Calculate counts and proportions
            group1_positive = (group1[col_name].astype(str).str.strip().str.lower() == class_name.lower()).sum()
            group2_positive = (group2[col_name].astype(str).str.strip().str.lower() == class_name.lower()).sum()
    except Exception as e:
        print(e)
        pdb.set_trace()
    group1_total = len(group1)
    group2_total = len(group2)
    
    group_1_proportion = group1_positive / group1_total if group1_total > 0 else 0
    group_2_proportion = group2_positive / group2_total if group2_total > 0 else 0
    
    # Handle zero proportion cases
    if group_1_proportion == 0 or group_2_proportion == 0:
        if col_name == 'ground_truth':
            # Skip this evaluation for ground truth
            conclusion = None
            p_value = None
            z_stat = None
        else:
            # Make comparable with ground truth conclusion
            if group_1_proportion == 0 and group_2_proportion == 0:
                conclusion = "no_difference"
            elif group_1_proportion == 0:
                conclusion = "group2_larger"
            else:  # group_2_proportion == 0
                conclusion = "group1_larger"
            
            p_value = None
            z_stat = None
    else:
        # Run proportions z-test
        count = [group1_positive, group2_positive]
        nobs = [group1_total, group2_total]
        z_stat, p_value = call_r_regression(count, nobs)
            
        # Determine conclusion
        if p_value < significance_treshold:
            if group_1_proportion > group_2_proportion:
                conclusion = "group1_larger"
            else:
                conclusion = "group2_larger"
        else:
            conclusion = "no_difference"
    
    return conclusion, p_value, z_stat


def run_baselines(filepath_for_baseline_data, nr_of_bootstrap_samples, nr_of_bootstrap_samples_bl3):
    assert nr_of_bootstrap_samples > 0 and nr_of_bootstrap_samples_bl3 > 0 and nr_of_bootstrap_samples >= nr_of_bootstrap_samples_bl3, 'Nr of bootstrap samples not supported.'
    # baseline 1: random conclusions
    # baseline 2: random labels
    # baseline 3: random errors (random labels with a given weighted f1 score)
    timestamp = get_experiment_start_date()
    all_bl1_llm_hacking_risks = []
    all_bl1_type_I_risks = []
    all_bl1_type_II_risks = []
    all_bl1_type_S_risks = []
    all_bl2_llm_hacking_risks = []
    all_bl2_type_I_risks = []
    all_bl2_type_II_risks = []
    all_bl2_type_S_risks = []
    all_bl1_results = []
    all_bl2_results = []
    all_bl3_results = []
    for task_name in tqdm(all_tasks, desc='Processing tasks', leave=True, total=len(all_tasks)):
        bl1_results = []
        bl2_results = []
        bl3_results = []
        tn = task_name.strip()
        data_loader = map_dataset_name_to_class(tn)
        dataset = data_loader.load_dataset()
        dataset = dataset.reset_index(drop=True)  # Reset index to 0, 1, 2, ..., len(dataset)-1
        # Set the global groupings variable
        groupings, nr_of_default_groupings, nr_of_dataset_specific_groups = data_loader.get_groups(dataset)
        # load all llm annotated data
        
        for seed in tqdm(range(nr_of_bootstrap_samples), total=nr_of_bootstrap_samples, desc='Bootstrapping...'):
            if seed <= nr_of_bootstrap_samples_bl3:
                run_bl3 = True
            else:
                run_bl3 = False
            rng = np.random.default_rng(seed)
            gt_classes = list(dataset['ground_truth'].unique())

            # Generate random labels for baseline 2 on the entire dataset
            dataset['random_labels'] = rng.choice(gt_classes, size=len(dataset))

            cols_to_keep = []
            if run_bl3:
                # Generate random labels with specific F1 scores for baseline 3
                all_f1_performance_cols = []
                for fraction_of_labels_to_flip in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
                    col_name = f'random_labels_{fraction_of_labels_to_flip}'
                    cols_to_keep.append(col_name)
                    
                    # Start with ground truth and flip fraction_of_labels_to_flip proportion of labels
                    dataset[col_name] = dataset['ground_truth'].copy()
                    if fraction_of_labels_to_flip == 0:
                        pass
                    else:
                        if fraction_of_labels_to_flip == 1:
                            indices_to_flip = range(len(dataset))
                        else:
                            n_to_flip = int(len(dataset) * fraction_of_labels_to_flip)
                            indices_to_flip = rng.choice(len(dataset), size=n_to_flip, replace=False)
                        # For each index to flip, randomly select a different class
                        for idx in indices_to_flip:
                            try:
                                current_class = dataset.loc[idx, col_name]
                            except Exception as e:
                                print('aaa')
                                print(e)
                                pdb.set_trace()
                            other_classes = [c for c in gt_classes if c != current_class]
                            if other_classes:  # Only flip if there are other classes available
                                dataset.loc[idx, col_name] = rng.choice(other_classes)
                            else:
                                print("Why no other classes available?")
                                pdb.set_trace()
                    f1_weighted_performance = f1_score_weighted(dataset['ground_truth'], dataset[col_name])
                    all_f1_performance_cols.append((col_name, f1_weighted_performance, fraction_of_labels_to_flip))

            cols_to_keep.extend(['ground_truth', 'response_mapped', 'random_labels'])
            cols_to_keep = [c for c in cols_to_keep if c in dataset.columns]
            
            for grouping_name, grouping_function, grouping_args in groupings:
                group1, group2, group1_name, group2_name = grouping_function(dataset, **grouping_args)
                group1 = group1[cols_to_keep]
                group2 = group2[cols_to_keep]
                if group1.empty or group2.empty:
                    # pdb.set_trace()
                    continue
                group1 = group1.copy()  # Make copies to avoid modifying original data
                group2 = group2.copy()
                group1.loc[:, 'group'] = 0
                group2.loc[:, 'group'] = 1

                # Handle binarization
                if len(gt_classes) > 2:
                    binarized_classes = gt_classes
                else:
                    binarized_classes = [gt_classes[0]]
                
                for class_name in binarized_classes:

                    conclusion_gt, p_value_gt, z_stat_gt = get_conclusion(group1, group2, class_name, col_name='ground_truth')
                    if conclusion_gt is not None:
                        # baseline 1 - random conclusions
                        conclusion_bl1 = rng.choice(['group1_larger', 'group2_larger', 'no_difference', 'no_difference'])
                        bl1_results.append({
                            'task': tn,
                            'conclusion_gt': conclusion_gt,
                            'conclusion_llm': conclusion_bl1,
                            'llm_hacking': assess_llm_hacking({'conclusion_gt':conclusion_gt, 'conclusion_llm':conclusion_bl1}),
                        })
                        
                        # baseline 2 - random labels
                        conclusion_bl2, p_value_bl2, z_stat_bl2 = get_conclusion(group1, group2, class_name, col_name='random_labels')
                        bl2_results.append({
                            'task': tn,
                            'conclusion_gt': conclusion_gt,
                            'p_value_gt': p_value_gt,
                            'z_stat_gt': z_stat_gt,
                            'conclusion_llm': conclusion_bl2,
                            'p_value_llm': p_value_bl2,
                            'z_stat_llm': z_stat_bl2,
                            'llm_hacking': assess_llm_hacking({'conclusion_gt':conclusion_gt, 'conclusion_llm':conclusion_bl2}),
                        })
                        
                        if run_bl3:
                            # baseline 3 - random errors with specific F1 scores
                            for f1_weighted_performance_col_name, f1_score, fraction_of_labels_to_flip in all_f1_performance_cols:
                                conclusion_bl3, p_value_bl3, z_stat_bl3 = get_conclusion(group1, group2, class_name, col_name=f1_weighted_performance_col_name)
                                bl3_results.append({
                                    'task': tn,
                                    'conclusion_gt': conclusion_gt,
                                    'p_value_gt': p_value_gt,
                                    'z_stat_gt': z_stat_gt,
                                    'conclusion_llm': conclusion_bl3,
                                    'p_value_llm': p_value_bl3,
                                    'z_stat_llm': z_stat_bl3,
                                    'f1_score_weighted': f1_score,
                                    'llm_hacking': assess_llm_hacking({'conclusion_gt':conclusion_gt, 'conclusion_llm':conclusion_bl3}),
                                    'fraction_of_labels_to_flip': fraction_of_labels_to_flip,
                                })

        bl1_results_df = pd.DataFrame(bl1_results)
        bl1_type_I_risks = empirical_type_I_risk(df=bl1_results_df)
        bl1_type_II_risks = empirical_type_II_risk(df=bl1_results_df)
        bl1_type_S_risks = empirical_type_S_risk(df=bl1_results_df)
        bl1_llm_hacking_risk = empirical_llm_hacking_risk(task_type_i_risk=bl1_type_I_risks, task_type_ii_risk=bl1_type_II_risks, task_type_s_risk=bl1_type_S_risks)
        all_bl1_llm_hacking_risks.append(bl1_llm_hacking_risk)
        all_bl1_type_I_risks.append(bl1_type_I_risks)
        all_bl1_type_II_risks.append(bl1_type_II_risks)
        all_bl1_type_S_risks.append(bl1_type_S_risks)
        bl2_results_df = pd.DataFrame(bl2_results)
        bl2_type_I_risks = empirical_type_I_risk(df=bl2_results_df)
        bl2_type_II_risks = empirical_type_II_risk(df=bl2_results_df)
        bl2_type_S_risks = empirical_type_S_risk(df=bl2_results_df)
        bl2_llm_hacking_risk = empirical_llm_hacking_risk(task_type_i_risk=bl2_type_I_risks, task_type_ii_risk=bl2_type_II_risks, task_type_s_risk=bl2_type_S_risks)
        all_bl2_llm_hacking_risks.append(bl2_llm_hacking_risk)
        all_bl2_type_I_risks.append(bl2_type_I_risks)
        all_bl2_type_II_risks.append(bl2_type_II_risks)
        all_bl2_type_S_risks.append(bl2_type_S_risks)

        all_bl1_results.extend(bl1_results)
        all_bl2_results.extend(bl2_results)
        all_bl3_results.extend(bl3_results)

        # Save risk metrics for this task to disk
        task_risk_data = [
            {'task': task_name, 'baseline': 'baseline1_random_conclusions', 'error_type': 'llm_hacking_risk', 'value': bl1_llm_hacking_risk},
            {'task': task_name, 'baseline': 'baseline1_random_conclusions', 'error_type': 'type_I_risk', 'value': bl1_type_I_risks},
            {'task': task_name, 'baseline': 'baseline1_random_conclusions', 'error_type': 'type_II_risk', 'value': bl1_type_II_risks},
            {'task': task_name, 'baseline': 'baseline1_random_conclusions', 'error_type': 'type_S_risk', 'value': bl1_type_S_risks},
            {'task': task_name, 'baseline': 'baseline2_random_labels', 'error_type': 'llm_hacking_risk', 'value': bl2_llm_hacking_risk},
            {'task': task_name, 'baseline': 'baseline2_random_labels', 'error_type': 'type_I_risk', 'value': bl2_type_I_risks},
            {'task': task_name, 'baseline': 'baseline2_random_labels', 'error_type': 'type_II_risk', 'value': bl2_type_II_risks},
            {'task': task_name, 'baseline': 'baseline2_random_labels', 'error_type': 'type_S_risk', 'value': bl2_type_S_risks}
        ]
        task_risk_df = pd.DataFrame(task_risk_data)
        task_risk_df.to_csv(f'{filepath_for_baseline_data}/risk_metrics_summary.csv', 
                           mode='a', header=not os.path.exists(f'{filepath_for_baseline_data}/risk_metrics_summary.csv'), index=False)

        bl3_results_df = pd.DataFrame(bl3_results)
        bl3_results_df.to_csv(f'{filepath_for_baseline_data}/bl3_summary.csv', 
                           mode='a', header=not os.path.exists(f'{filepath_for_baseline_data}/risk_metrics_summary.csv'), index=False)
        
    # Convert to DataFrames and save to CSV
    bl1_results_df = pd.DataFrame(all_bl1_results)
    print(f'Average baseline 1 LLM hacking risk across all {len(all_bl1_llm_hacking_risks)} tasks: {np.mean(all_bl1_llm_hacking_risks)}')
    print(f'Average baseline 1 TYPE I risk across all {len(all_bl1_type_I_risks)} tasks: {np.mean(all_bl1_type_I_risks)}')
    print(f'Average baseline 1 TYPE II risk across all {len(all_bl1_type_II_risks)} tasks: {np.mean(all_bl1_type_II_risks)}')
    print(f'Average baseline 1 TYPE S risk across all {len(all_bl1_type_S_risks)} tasks: {np.mean(all_bl1_type_S_risks)}')
    bl1_results_df.to_csv(f'{filepath_for_baseline_data}/baseline1_random_conclusions_{timestamp}.csv', index=False)
    
    bl2_results_df = pd.DataFrame(all_bl2_results)
    print(f'Average baseline 2 LLM hacking risk across all {len(all_bl2_llm_hacking_risks)} tasks: {np.mean(all_bl2_llm_hacking_risks)}')
    print(f'Average baseline 2 TYPE I risk across all {len(all_bl2_type_I_risks)} tasks: {np.mean(all_bl2_type_I_risks)}')
    print(f'Average baseline 2 TYPE II risk across all {len(all_bl2_type_II_risks)} tasks: {np.mean(all_bl2_type_II_risks)}')
    print(f'Average baseline 2 TYPE S risk across all {len(all_bl2_type_S_risks)} tasks: {np.mean(all_bl2_type_S_risks)}')
    bl2_results_df.to_csv(f'{filepath_for_baseline_data}/baseline2_random_labels_{timestamp}.csv', index=False)
    
    all_bl3_results_df = pd.DataFrame(all_bl3_results)
    all_bl3_results_df.to_csv(f'{filepath_for_baseline_data}/baseline3_random_errors_{timestamp}.csv', index=False)


def main():
    filepath = f'results_FINAL/baseline_data'
    filepath_for_baseline_data = Path(filepath)
    filepath_for_baseline_data.mkdir(parents=True, exist_ok=True)

    run_baselines(filepath_for_baseline_data, nr_of_bootstrap_samples=1000, nr_of_bootstrap_samples_bl3=100)


if __name__ == "__main__":
     main()
