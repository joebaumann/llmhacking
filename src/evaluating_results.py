import pdb
import sys
import psutil
from tqdm import tqdm
import pandas as pd
from pathlib import Path

import concurrent.futures
import argparse
import json
from datetime import datetime
import os
import shutil

import time
import random
import ast
import re
import numpy as np
from data.data_utils import map_dataset_name_to_class, all_tasks
from src.metrics import *
from utils import get_experiment_start_date
from src.helpers import load_confidence_results
from src.statistical_downstream_analysis import all_statistical_tests, pooled_logistic_regression
from src.helpers import load_llm_annotated_data, binarize_labels
from src.llm_hacking_mitigations.cdi_utils import train_tree, tree_predict, inv_hessian_col_imputed

sys.stdout.reconfigure(line_buffering=True)

FRACTION_NA_REMOVAL_THRESHOLD = 0.01
GROUPINGS = None # using global groupings variable since they contain non-pickleable functions and can thus not be passed to parallel execution

if __name__ == "__main__":
    only_check_number_of_missing_combos = False #True
    do_not_save_results_if_only_checking_number_of_missing_combos = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-cpus', type=int, default=1, help='Number of CPUs to use for parallel processing')
    parser.add_argument('--task_names', type=str, help='Comma-separated list of tasks to run (e.g., "emotion,humor")')
    parser.add_argument('--models', type=str, default=None, help='Comma-separated list of models to use (e.g., "Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct")')
    parser.add_argument('--prompt_version_index', type=int, help='Only process one prompt version if index is given.', default=None)
    parser.add_argument('--results_folder', type=str, default=None, help='Path to the folder where the results should be saved')
    parser.add_argument('--only_run_pooled_logistic_regression', action='store_true')
    args = parser.parse_args()
    NUM_CPUS = args.num_cpus
    print('NUM_CPUS:', NUM_CPUS)

    total_number_of_combos_by_task = {}
    total_number_of_completed_combos_by_task = {}
    if only_check_number_of_missing_combos:
        models_list = [
            'gpt-4o-mini-2024-07-18',
            'gpt-4o-2024-08-06',
            'meta-llama/Llama-3.2-1B-Instruct',
            'meta-llama/Llama-3.2-3B-Instruct',
            'meta-llama/Llama-3.1-8B-Instruct',
            'Qwen/Qwen3-1.7B',
            'Qwen/Qwen3-4B',
            'Qwen/Qwen3-8B',
            'Qwen/Qwen2.5-1.5B-Instruct',
            'Qwen/Qwen2.5-3B-Instruct',
            'Qwen/Qwen2.5-7B-Instruct',
            'meta-llama/Llama-3.1-70B-Instruct',
            'Qwen/Qwen2.5-72B-Instruct',
            'Qwen/Qwen2.5-32B-Instruct',
            'Qwen/Qwen3-32B',
            'google/gemma-3-27b-it',
            'google/gemma-3-4b-it',
            'google/gemma-3-1b-it',
        ]
        print(f"Running all {len(all_tasks)} tasks!")
        print(f"Running all {len(models_list)} models!")
    else:
        print(f"Running tasks: {args.task_names}")
        print(f"Models: {args.models}")

        # Process models argument if provided
        models_list = None
        if args.models:
            models_list = [model.strip() for model in args.models.split(',') if model.strip() != '']

        print(f"  Models: {models_list}")
        print(f"  Results folder: {args.results_folder}")

        if args.task_names:
            all_tasks = args.task_names.split(',')

    # Keep trying until we successfully create a new folder
    while True:
        timestamp = get_experiment_start_date()
        if only_check_number_of_missing_combos:
            eval_results_outpath = Path('src', args.results_folder, f'{timestamp}_CHECKCOMBOSONLY')
        else:
            eval_results_outpath = Path('src', args.results_folder, timestamp)
        
        try:
            # Try to create directory - this will raise FileExistsError if it exists
            eval_results_outpath.mkdir(parents=True, exist_ok=False)
            # If we get here, the folder was successfully created
            break
        except FileExistsError:
            # Folder already exists, wait and try again
            wait_time = random.uniform(5, 20)
            print(f"Folder {eval_results_outpath} already exists. Waiting {wait_time:.1f} seconds before retrying...")
            time.sleep(wait_time)

    print(f"Successfully created new folder: {eval_results_outpath}")


    # Add temp directory for intermediate results
    temp_results_path = eval_results_outpath / 'temp_results'
    temp_results_path.mkdir(parents=True, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays and other numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.complex):
            return {'real': obj.real, 'imag': obj.imag}
        return super(NumpyEncoder, self).default(obj)


def load_checkpoint(task_name):
    """Load checkpoint files from all temp subdirectories to see which combinations have been processed.
    Also returns all relevant task directories for reuse in aggregation."""
    completed_combos = set()
    relevant_task_dirs = []
    
    # Get the base evaluation results directory (parent of current temp_results_path)
    base_eval_dir = eval_results_outpath.parent
    
    # Iterate through all subdirectories in the base evaluation results directory
    for subdir in base_eval_dir.iterdir():
        if subdir.is_dir():
            # Check if this subdirectory has a temp_results folder
            temp_dir = subdir / 'temp_results'
            if temp_dir.exists() and temp_dir.is_dir():
                # Check if there's a task-specific directory
                task_temp_path = temp_dir / task_name
                if task_temp_path.exists() and task_temp_path.is_dir():
                    relevant_task_dirs.append(task_temp_path)
                
                # Look for the checkpoint file for this task
                checkpoint_file = temp_dir / f'{task_name}_checkpoint.json'
                if checkpoint_file.exists():
                    try:
                        with open(checkpoint_file, 'r') as f:
                            combos = json.load(f)
                            # Add all combinations from this checkpoint file
                            completed_combos.update(tuple(combo) for combo in combos)
                        print(f"Loaded {len(combos)} combinations from {checkpoint_file}")
                    except (json.JSONDecodeError, FileNotFoundError) as e:
                        print(f"Warning: Could not load checkpoint file {checkpoint_file}: {e}")
                        continue
    
    print(f"Total unique combinations loaded for {task_name}: {len(completed_combos)}")
    print(f"Found {len(relevant_task_dirs)} relevant task directories")
    return completed_combos, relevant_task_dirs

def save_checkpoint(task_name, completed_combos):
    """Save checkpoint file with completed combinations."""
    checkpoint_file = temp_results_path / f'{task_name}_checkpoint.json'
    with open(checkpoint_file, 'w') as f:
        json.dump([list(combo) for combo in completed_combos], f)

def get_combo_key(model_name, prompt, temperature, seed):
    """Create a unique key for a model/param combination."""
    return (model_name, prompt, float(temperature), int(seed))

def log_progress(message):
    """Log progress to file with timestamp."""
    log_file = eval_results_outpath / 'progress.log'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a') as f:
        f.write(f'[{timestamp}] {message}\n')

def save_csv_to_disk(df, filename, dir_name=None):
    if dir_name is None:
        dir_name = eval_results_outpath
    filepath = Path(dir_name, filename)
    # Create directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Convert to DataFrame if it's a list of dicts
    if isinstance(df, list):
        df = pd.DataFrame(df)
    # save df to disk as csv
    df.to_csv(filepath, index=False)


def aggregate_all_results(task_name, relevant_task_dirs):
    """Aggregate all results from relevant task directories."""
    performance_results = []
    llm_downstream_analysis_results = []
    
    print(f"Aggregating results for {task_name} from {len(relevant_task_dirs)} directories...")
    
    for task_temp_path in relevant_task_dirs:
        print(f"Loading results from {task_temp_path}")
        
        # Load performance results
        performance_files = list(task_temp_path.glob('performance_*.json'))
        for f in performance_files:
            try:
                with open(f, 'r') as file:
                    result = json.load(file)
                    performance_results.append(result)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not load performance file {f}: {e}")
                continue
        
        # Load comparison results
        comparison_files = list(task_temp_path.glob('comparison_*.json'))
        for f in comparison_files:
            try:
                with open(f, 'r') as file:
                    results = json.load(file)
                    # Handle both single results and lists of results
                    if isinstance(results, list):
                        llm_downstream_analysis_results.extend(results)
                    else:
                        llm_downstream_analysis_results.append(results)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not load comparison file {f}: {e}")
                continue
        
        print(f"  - Loaded {len(performance_files)} performance files")
        print(f"  - Loaded {len(comparison_files)} comparison files")
    
    print(f"Total aggregated results for {task_name}:")
    print(f"  - Performance results: {len(performance_results)}")
    print(f"  - Downstream analysis results: {len(llm_downstream_analysis_results)}")
    
    return performance_results, llm_downstream_analysis_results


def cleanup_temp_files(task_name, relevant_task_dirs):
    """Clean up temp files for this task from all relevant directories."""
    print(f"Cleaning up temp files for {task_name} from {len(relevant_task_dirs)} directories...")
    
    for task_temp_path in relevant_task_dirs:
        if task_temp_path.exists():
            try:
                shutil.rmtree(task_temp_path)
                print(f"Cleaned up {task_temp_path}")
            except Exception as e:
                print(f"Warning: Could not clean up {task_temp_path}: {e}")



def compare_annotations_with_ground_truth(dataset_ground_truth, llm_data):
    # annotation performance
    ground_truth_annotations, llm_annotations = get_annotations_from_dfs(dataset_ground_truth, llm_data)
    return {
        'accuracy': accuracy(ground_truth_annotations, llm_annotations),
        'f1_score': f1_score(ground_truth_annotations, llm_annotations),
        'f1_score_weighted': f1_score_weighted(ground_truth_annotations, llm_annotations),
        'fraction_na': fraction_of_answers_that_did_not_follow_instructions(llm_annotations),
    }


def get_stat_info(stat_test_name):
    stat_function = all_statistical_tests[stat_test_name]['stat_function']
    stat_function_params = all_statistical_tests[stat_test_name].get('stat_function_params', {})
    name_of_corresp_stat_test_used_for_ground_truth = all_statistical_tests[stat_test_name].get('name_of_corresp_stat_test_used_for_ground_truth', stat_test_name)
    return stat_function, stat_function_params, name_of_corresp_stat_test_used_for_ground_truth


def sample_by_probability(group, k, rng, prob_column='sampling_probability'):
    """Sample k items using probability-based selection with guaranteed k selections"""
    if len(group) <= k:
        return group.index.tolist()
    
    # Use numpy's choice with probabilities (more efficient)
    probabilities = group[prob_column].values
    # Normalize probabilities to sum to 1
    probabilities = probabilities / probabilities.sum()
    
    selected_indices = rng.choice(
        group.index.tolist(), 
        size=k, 
        replace=False, 
        p=probabilities
    )
    return selected_indices.tolist()



def get_samples(nr_of_known_gt, bs_sample_idx, group1, group2, low_confidence_sampling_global=True, tau=0.1, seed=42):
    # Create deep copies to avoid modifying original data
    group1 = group1.copy(deep=True)
    group2 = group2.copy(deep=True)
    
    # verbalized_llm_confidence is a probability between 0 and 1
    # set sampling probability to be inverse confidence proba for DSL
    base_proba = nr_of_known_gt / (len(group1) + len(group2))

    # Mix adaptive sampling with uniform sampling
    def compute_sampling_prob(confidence, tau, base_proba):
        if confidence is None or pd.isna(confidence):
            return base_proba
        inverse_conf = 1 - confidence
        mixed_proba = tau * base_proba + (1 - tau) * inverse_conf
        return np.clip(mixed_proba, 0.01, 0.99)  # More reasonable bounds

    group1['sampling_probability'] = group1['verbalized_llm_confidence'].apply(
        lambda x: compute_sampling_prob(x, tau, base_proba)
    )
    group2['sampling_probability'] = group2['verbalized_llm_confidence'].apply(
        lambda x: compute_sampling_prob(x, tau, base_proba)
    )

    # initialize rng with bs_sample_idx for seed
    rng = np.random.RandomState(seed+bs_sample_idx)
    
    # Calculate proportional sizes ensuring minimum 2 per group
    n1, n2 = len(group1), len(group2)
    n_total = n1 + n2
    k1 = max(2, int(np.round(nr_of_known_gt * n1 / n_total)))
    k2 = max(2, nr_of_known_gt - k1)
    # Adjust if we exceeded total
    if k1 + k2 > nr_of_known_gt:
        if k1 > k2:
            k1 = nr_of_known_gt - k2
        else:
            k2 = nr_of_known_gt - k1
    
    # Random sampling
    idx1_random = rng.choice(n1, k1, replace=False)
    idx2_random = rng.choice(n2, k2, replace=False)
    group1_random_sample = group1.iloc[idx1_random].copy(deep=True)
    group2_random_sample = group2.iloc[idx2_random].copy(deep=True)
    assert (len(group1_random_sample) + len(group2_random_sample)) == nr_of_known_gt

    accuracy_on_random_sample = (
        (group1_random_sample['response_mapped'].str.lower() == group1_random_sample['ground_truth'].str.lower()).sum() + 
        (group2_random_sample['response_mapped'].str.lower() == group2_random_sample['ground_truth'].str.lower()).sum()
    ) / (len(group1_random_sample) + len(group2_random_sample))

    # for both df samples, set response_mapped to ground_truth. this represents the situation in which we rely on small sample of known grund truth instead of many llm annotations
    group1_random_sample['response_mapped'] = group1_random_sample['ground_truth']
    group2_random_sample['response_mapped'] = group2_random_sample['ground_truth']

    # Set ground_truth_known for random sampling
    group1_random_gt = group1.copy(deep=True)
    group2_random_gt = group2.copy(deep=True)
    group1_random_gt['ground_truth_known'] = None
    group2_random_gt['ground_truth_known'] = None
    # ground_truth_known represents a subsample of known ground truth labels, used for dsl
    group1_random_gt.iloc[idx1_random, group1_random_gt.columns.get_loc('ground_truth_known')] = group1_random_gt.iloc[idx1_random]['ground_truth'].values
    group2_random_gt.iloc[idx2_random, group2_random_gt.columns.get_loc('ground_truth_known')] = group2_random_gt.iloc[idx2_random]['ground_truth'].values

    # set response_mapped to ground truth for known samples and keep llm annotations for all others
    group1_random_sample_and_llm_annotations = group1_random_gt.copy(deep=True)
    group2_random_sample_and_llm_annotations = group2_random_gt.copy(deep=True)
    group1_random_sample_and_llm_annotations['response_mapped'] = group1_random_sample_and_llm_annotations['ground_truth_known'].fillna(group1_random_sample_and_llm_annotations['response_mapped'])
    group2_random_sample_and_llm_annotations['response_mapped'] = group2_random_sample_and_llm_annotations['ground_truth_known'].fillna(group2_random_sample_and_llm_annotations['response_mapped'])

    # low_confidence sampling based on confidence
    if low_confidence_sampling_global:
        # Global approach: combine groups and take lowest confidence
        combined = pd.concat([group1, group2], ignore_index=False)

        selected_idx = sample_by_probability(combined, nr_of_known_gt, rng)

        # Split back into groups
        g1_selected = [idx for idx in selected_idx if idx in group1.index]
        g2_selected = [idx for idx in selected_idx if idx in group2.index]
        
        # Create low_confidence samples
        group1_low_confidence_sample = combined.loc[g1_selected].copy(deep=True)
        group2_low_confidence_sample = combined.loc[g2_selected].copy(deep=True)
        
    else:
        # Per-group approach: sample by lower confidence within each group using sampling_probability
        # Sample from each group
        g1_low_confidence_idx = sample_by_probability(group1, k1, rng)
        g2_low_confidence_idx = sample_by_probability(group2, k2, rng)
        group1_low_confidence_sample = group1.loc[g1_low_confidence_idx]
        group2_low_confidence_sample = group2.loc[g2_low_confidence_idx]
        
    
    assert (len(group1_low_confidence_sample) + len(group2_low_confidence_sample)) == nr_of_known_gt

    # for both df samples, set response_mapped to ground_truth. this represents the situation in which we rely on small sample of known grund truth instead of many llm annotations
    group1_low_confidence_sample['response_mapped'] = group1_low_confidence_sample['ground_truth']
    group2_low_confidence_sample['response_mapped'] = group2_low_confidence_sample['ground_truth']

    # Set ground_truth_known for low_confidence sampling
    group1_low_confidence_gt = group1.copy(deep=True)
    group2_low_confidence_gt = group2.copy(deep=True)
    group1_low_confidence_gt['ground_truth_known'] = None
    group2_low_confidence_gt['ground_truth_known'] = None
    
    # ground_truth_known represents a subsample of known ground truth labels
    if low_confidence_sampling_global:
        group1_low_confidence_gt.loc[g1_selected, 'ground_truth_known'] = group1_low_confidence_gt.loc[g1_selected, 'ground_truth']
        group2_low_confidence_gt.loc[g2_selected, 'ground_truth_known'] = group2_low_confidence_gt.loc[g2_selected, 'ground_truth']
    else:
        group1_low_confidence_gt.loc[g1_low_confidence_idx, 'ground_truth_known'] = group1_low_confidence_gt.loc[g1_low_confidence_idx, 'ground_truth']
        group2_low_confidence_gt.loc[g2_low_confidence_idx, 'ground_truth_known'] = group2_low_confidence_gt.loc[g2_low_confidence_idx, 'ground_truth']
    
    # set response_mapped to ground truth for known samples and keep llm annotations for all others
    group1_low_confidence_sample_and_llm_annotations = group1_low_confidence_gt.copy(deep=True)
    group2_low_confidence_sample_and_llm_annotations = group2_low_confidence_gt.copy(deep=True)
    group1_low_confidence_sample_and_llm_annotations['response_mapped'] = group1_low_confidence_sample_and_llm_annotations['ground_truth_known'].fillna(group1_low_confidence_sample_and_llm_annotations['response_mapped'])
    group2_low_confidence_sample_and_llm_annotations['response_mapped'] = group2_low_confidence_sample_and_llm_annotations['ground_truth_known'].fillna(group2_low_confidence_sample_and_llm_annotations['response_mapped'])

    return group1_random_sample, group2_random_sample, group1_random_sample_and_llm_annotations, group2_random_sample_and_llm_annotations, group1_random_gt, group2_random_gt, group1_low_confidence_sample, group2_low_confidence_sample, group1_low_confidence_sample_and_llm_annotations, group2_low_confidence_sample_and_llm_annotations, group1_low_confidence_gt, group2_low_confidence_gt, accuracy_on_random_sample


def get_active_samples(nr_of_known_gt, bs_sample_idx, group1, group2, previous_sample_idx, gt_classes, tau=0.1, seed=42):
    rng = np.random.RandomState(seed+bs_sample_idx)
    
    # Initialize return dicts
    group1_active_sample_BY_GT_CLASS = {}
    group2_active_sample_BY_GT_CLASS = {}
    group1_active_sample_and_llm_annotations_BY_GT_CLASS = {}
    group2_active_sample_and_llm_annotations_BY_GT_CLASS = {}
    group1_active_gt_BY_GT_CLASS = {}
    group2_active_gt_BY_GT_CLASS = {}

    # previous_sample_idx are the idx of the previous sampling round
    nr_of_new_samples_this_round = nr_of_known_gt - len(previous_sample_idx)

    if len(gt_classes) > 2:
        binarized_classes = gt_classes
    else:
        binarized_classes = [gt_classes[0]]
    
    for class_name in binarized_classes:
        # Create combined df
        df = pd.concat([group1.copy(deep=True), group2.copy(deep=True)])
        df = binarize_labels(df, class_name)
        
        # Store original group indices for mapping
        group1_indices = df[df['group'] == 0].index.tolist()
        group2_indices = df[df['group'] == 1].index.tolist()
        
        # Initialize weights
        df['weights_active'] = 0.0

        # Create design matrix (group first, intercept second)
        n = len(df)
        X = np.column_stack((df['group'].values, np.ones(n))).astype(float)

        # get nr_of_new_samples_this_round new samples
        if len(previous_sample_idx) == 0:
            # Burn-in: random sampling
            n1, n2 = len(group1_indices), len(group2_indices)
            n_total = n1 + n2
            k1 = max(1, int(np.round(nr_of_new_samples_this_round * n1 / n_total)))
            k2 = nr_of_new_samples_this_round - k1
            
            new_idx1 = rng.choice(group1_indices, min(k1, n1), replace=False).tolist()
            new_idx2 = rng.choice(group2_indices, min(k2, n2), replace=False).tolist()
            sampled_indices = new_idx1 + new_idx2
            
            # NEW CODE TO HAVE AT LEAST ONE EXAMPLE OF BOTH CLASSES !!FOR EACH GROUP!!
            # Check if each group has both classes
            for group_idx, group_name in [(new_idx1, 'group0'), (new_idx2, 'group1')]:
                if len(group_idx) > 0:  # Only check if we sampled from this group
                    sampled_gt = df.loc[group_idx, 'ground_truth'].values
                    unique_classes = np.unique(sampled_gt)
                    
                    if len(unique_classes) == 1:
                        # This group only has one class, need to swap one element to get the other class
                        missing_class = 1 - unique_classes[0]
                        
                        # Find indices from the SAME GROUP with the missing class that weren't sampled
                        group_val = 0 if group_name == 'group0' else 1
                        available_missing = df[(df['ground_truth'] == missing_class) & 
                                              (df['group'] == group_val) &
                                              (~df.index.isin(sampled_indices))].index.tolist()
                        
                        if len(available_missing) > 0:
                            # Replace a random sample in this group
                            swap_idx = rng.choice(len(group_idx))
                            group_idx[swap_idx] = rng.choice(available_missing)
            
            # Update sampled_indices after potential swaps
            sampled_indices = new_idx1 + new_idx2

            # Burn-in samples have weight 1 (prob = 1)
            df.loc[sampled_indices, 'weights_active'] = 1.0
            
        else:
            print('test1')
            # Active sampling using CDI approach
            # First, copy weights from previous rounds
            if 'weights_active' in group1.columns:
                df.loc[group1_indices, 'weights_active'] = group1['weights_active'].values
            if 'weights_active' in group2.columns:
                df.loc[group2_indices, 'weights_active'] = group2['weights_active'].values
            
            # Train on all previously sampled data
            labeled_mask = df['weights_active'] > 0
            prev_data = df[labeled_mask]
            # prev_data = df.loc[previous_sample_idx]
            
            # Train confidence → error model
            # retrain for every value in nr_of_known_ground_truth_samples_for_validation
            confidence_prev = prev_data['verbalized_llm_confidence'].values.reshape(-1, 1)
            errors_prev = ((prev_data['ground_truth'] - prev_data['response_mapped']) ** 2).values
            tree = train_tree(confidence_prev, errors_prev)
            
            # Calculate uncertainties with inverse Hessian weighting
            confidence_all = df['verbalized_llm_confidence'].values.reshape(-1, 1)
            
            # Get inverse Hessian column for optimal sampling
            Yhat_all = df['response_mapped'].values

            try:
                h = inv_hessian_col_imputed(X, Yhat_all)
            except Exception as e:
                print(f"inv_hessian_col_imputed exception: --{e}--")
                try:
                    # in case values in Yhat_all are the same, we prevent hessian function from crashing by adding a little noise (flipping a low confidence output)
                    # flip one output so that there are always two different values in Yhat_all_noisy
                    Yhat_all_noisy = Yhat_all.copy()
                    
                    # Check if all values are the same
                    if len(np.unique(Yhat_all_noisy)) == 1:
                        # Find indices of two samples with lowest confidence
                        confidence_flat = confidence_all.flatten()
                        sorted_indices = np.argsort(confidence_flat)
                        
                        # Flip the least confident predictions to ensure both 0 and 1 are present
                        Yhat_all_noisy[sorted_indices[0]] = 1 - Yhat_all_noisy[sorted_indices[0]]
                    
                    h = inv_hessian_col_imputed(X, Yhat_all_noisy)
                except Exception as e:
                    print(f"inv_hessian_col_imputed exception2: --{e}--")
                    # Fallback: create a reasonable default h vector
                    h = np.ones(X.shape[1]) / np.sqrt(X.shape[1])
            
            # Calculate weighted uncertainties
            uncertainties = np.sqrt(tree_predict(tree, confidence_all)) * np.abs(X.dot(h))
            avg_uncertainty = np.mean(uncertainties)

            # Get remaining indices - using positional indices
            remaining_mask = df['weights_active'] == 0
            remaining_positions = np.where(remaining_mask)[0]  # Get positional indices
            
            if len(remaining_positions) > 0:
                # Calculate sampling probabilities using positional indices
                # Higher uncertainty → higher sampling probability
                remaining_uncertainties = uncertainties[remaining_positions]
                base_prob = nr_of_new_samples_this_round / len(remaining_positions)
                sampling_probs = remaining_uncertainties / avg_uncertainty * base_prob

                # Mix with uniform sampling (tau=0.1 means 10% uniform, 90% adaptive)
                # This ensures some exploration and prevents extreme probabilities
                sampling_probs = (1 - tau) * sampling_probs + tau * base_prob
                sampling_probs = np.clip(sampling_probs, 0, 1)
                
                # Sample points using positional indices
                sampled_mask = rng.random(len(remaining_positions)) < sampling_probs
                sampled_positions = remaining_positions[sampled_mask]
                
                # Convert back to DataFrame indices
                sampled_indices = df.iloc[sampled_positions].index.tolist()
                
                # Ensure we get exactly the right number
                if len(sampled_indices) < nr_of_new_samples_this_round:
                    unsampled_positions = remaining_positions[~sampled_mask]
                    additional_positions = rng.choice(unsampled_positions, 
                                                    nr_of_new_samples_this_round - len(sampled_indices), 
                                                    replace=False)
                    additional_indices = df.iloc[additional_positions].index.tolist()
                    sampled_indices.extend(additional_indices)
                elif len(sampled_indices) > nr_of_new_samples_this_round:
                    sampled_indices = rng.choice(sampled_indices, 
                                               nr_of_new_samples_this_round, 
                                               replace=False).tolist()
                

                # Set weights for newly sampled points
                for idx in sampled_indices:
                    # Find the positional index of this DataFrame index
                    pos_in_df = df.index.get_loc(idx)
                    # Find where this positional index appears in remaining_positions
                    pos_in_remaining_array = np.where(remaining_positions == pos_in_df)[0]
                    
                    if len(pos_in_remaining_array) > 0:
                        prob_idx = pos_in_remaining_array[0]
                        df.loc[idx, 'weights_active'] = 1.0 / sampling_probs[prob_idx]
                    else:
                        # Fallback for edge cases
                        df.loc[idx, 'weights_active'] = 1.0
                        pdb.set_trace()

            else:
                pdb.set_trace()
                sampled_indices = []
        
        # Extract the data with weights
        group1_with_weights = df.loc[group1_indices].copy()
        group2_with_weights = df.loc[group2_indices].copy()
        
        # MITIGATION 7
        group1_active_sample = group1_with_weights[group1_with_weights['weights_active'] > 0].copy(deep=True)
        group2_active_sample = group2_with_weights[group2_with_weights['weights_active'] > 0].copy(deep=True)
        group1_active_sample['response_mapped'] = group1_active_sample['ground_truth']
        group2_active_sample['response_mapped'] = group2_active_sample['ground_truth']
        
        # MITIGATION 9
        group1_active_gt = group1_with_weights.copy(deep=True)
        group2_active_gt = group2_with_weights.copy(deep=True)
        
        # MITIGATION 8
        # group1_active_gt['ground_truth_known'] = None
        # group2_active_gt['ground_truth_known'] = None
        group1_active_sample_and_llm_annotations = group1_active_gt.copy()
        group2_active_sample_and_llm_annotations = group2_active_gt.copy()
        
        # Set ground_truth_known for sampled points
        sampled_idx1 = group1_active_sample_and_llm_annotations[group1_active_sample_and_llm_annotations['weights_active'] > 0].index
        sampled_idx2 = group2_active_sample_and_llm_annotations[group2_active_sample_and_llm_annotations['weights_active'] > 0].index
        
        group1_active_sample_and_llm_annotations.loc[sampled_idx1, 'response_mapped'] = group1_active_sample_and_llm_annotations.loc[sampled_idx1, 'ground_truth']
        group2_active_sample_and_llm_annotations.loc[sampled_idx2, 'response_mapped'] = group2_active_sample_and_llm_annotations.loc[sampled_idx2, 'ground_truth']
        
        # Store results
        group1_active_sample_BY_GT_CLASS[class_name] = group1_active_sample
        group2_active_sample_BY_GT_CLASS[class_name] = group2_active_sample
        group1_active_sample_and_llm_annotations_BY_GT_CLASS[class_name] = group1_active_sample_and_llm_annotations
        group2_active_sample_and_llm_annotations_BY_GT_CLASS[class_name] = group2_active_sample_and_llm_annotations
        group1_active_gt_BY_GT_CLASS[class_name] = group1_active_gt
        group2_active_gt_BY_GT_CLASS[class_name] = group2_active_gt

        del df
    
    all_sample_idx_for_next_round = previous_sample_idx + sampled_indices
    
    return group1_active_sample_BY_GT_CLASS, group2_active_sample_BY_GT_CLASS, group1_active_sample_and_llm_annotations_BY_GT_CLASS, group2_active_sample_and_llm_annotations_BY_GT_CLASS, group1_active_gt_BY_GT_CLASS, group2_active_gt_BY_GT_CLASS, all_sample_idx_for_next_round, group1_with_weights, group2_with_weights


def process_single_grouping(args_tuple, nr_of_known_ground_truth_samples_for_validation, max_percentage_of_known_gt, nr_of_bootstrap_samples):
    """Process a single grouping with pre-computed groups - for parallel execution."""
    grouping_name, group1, group2, group1_name, group2_name, gt_classes, fraction_na, using_ground_truth_annotations = args_tuple
    
    downstream_conclusions_updated = []
    
    # Groups are already computed, just need to set group indicators
    n_total = len(group1) + len(group2)
    group1 = group1.copy()  # Make copies to avoid modifying original data
    group2 = group2.copy()
    group1.loc[:, 'group'] = 0
    group2.loc[:, 'group'] = 1

    base_conclusion_dict = {
        'grouping_name': grouping_name,
        'group1_name': group1_name,
        'group2_name': group2_name,
    }
    if fraction_na is not None:
        base_conclusion_dict.update({'fraction_na': fraction_na})

    # downstream conclusions without any mitigation techniques for ground truth and for LLM annotations
    stat_test_name = 'log_regression_R_glm'
    stat_function, stat_function_params, name_of_corresp_stat_test_used_for_ground_truth = get_stat_info(stat_test_name)
    downstream_conclusions = stat_function(
        group1,
        group2,
        gt_classes,
        using_ground_truth_annotations=using_ground_truth_annotations,
        **stat_function_params
    )
    # Update each conclusion with metadata
    for conclusion in downstream_conclusions:
        full_conclusion_dict = base_conclusion_dict.copy()
        full_conclusion_dict.update({
            'stat_test_name': stat_test_name,
            'name_of_corresp_stat_test_used_for_ground_truth': name_of_corresp_stat_test_used_for_ground_truth,
            'stat_function_params': str(stat_function_params),
            **conclusion
        })
        downstream_conclusions_updated.append(full_conclusion_dict)
    
    # Add mitigation techniques processing if not using ground truth
    if not using_ground_truth_annotations:
        previous_sample_idx_dict = {}
        is_burn_in_sample = True
        for nr_of_known_gt in nr_of_known_ground_truth_samples_for_validation:
            print("\nnr_of_known_gt:", nr_of_known_gt, '\n')
            # only include tasks for which at least a max_percentage_of_known_gt fraction of datapoints are unknown
            if (n_total*max_percentage_of_known_gt) < nr_of_known_gt:
            # or len(group1) < nr_of_known_gt or len(group2) < nr_of_known_gt:
                # pdb.set_trace()
                continue
            for bs_sample_idx in range(nr_of_bootstrap_samples):

                print(f"\nbs_sample_idx: {bs_sample_idx}\n")

                group1_random_sample, group2_random_sample, group1_random_sample_and_llm_annotations, group2_random_sample_and_llm_annotations, group1_random_gt, group2_random_gt, group1_low_confidence_sample, group2_low_confidence_sample, group1_low_confidence_sample_and_llm_annotations, group2_low_confidence_sample_and_llm_annotations, group1_low_confidence_gt, group2_low_confidence_gt, accuracy_on_random_sample = get_samples(nr_of_known_gt, bs_sample_idx, group1, group2)

                all_mitigation_techniques = {
                    ##############################################
                    # MITIGATION 1: use only sample of known ground truth labels instead of LLM annotations
                    ##############################################
                    'ground_truth_sample_only': {
                        'stat_test_name': 'log_regression_R_glm',
                        'g1': group1_random_sample,
                        'g2': group2_random_sample,
                    },

                    ##############################################
                    # MITIGATION 2: use llm annotations PLUS sample of known ground truth labels
                    ##############################################
                    'ground_truth_sample_plus_annotations': {
                        'stat_test_name': 'log_regression_R_glm',
                        'g1': group1_random_sample_and_llm_annotations,
                        'g2': group2_random_sample_and_llm_annotations,
                    },

                    ##############################################
                    # MITIGATION 3: use llm annotations PLUS a sample of known ground truth labels to correct estimates with DSL
                    ##############################################
                    'dsl': {
                        'stat_test_name': 'log_regression_R_glm_dsl',
                        'g1': group1_random_gt,
                        'g2': group2_random_gt,
                    },

                    ##############################################
                    # MITIGATION 4: use only low_confidence (i.e., driven by low confidence) sample of known ground truth labels for which the LLM is not confident instead of LLM annotations
                    ##############################################
                    'ground_truth_sample_only_low_confidence': {
                        'stat_test_name': 'log_regression_R_glm',
                        'g1': group1_low_confidence_sample,
                        'g2': group2_low_confidence_sample,
                    },

                    ##############################################
                    # MITIGATION 5: use llm annotations PLUS low_confidence (i.e., driven by low confidence) sample of known ground truth labels for which the LLM is not confident
                    ##############################################
                    'ground_truth_sample_low_confidence_plus_annotations': {
                        'stat_test_name': 'log_regression_R_glm',
                        'g1': group1_low_confidence_sample_and_llm_annotations,
                        'g2': group2_low_confidence_sample_and_llm_annotations,
                    },

                    ##############################################
                    # MITIGATION 6: use llm annotations PLUS low_confidence (i.e., driven by low confidence) sample of known ground truth labels for which the LLM is not confident to correct estimates with DSL
                    ##############################################
                    'dsl_low_confidence': {
                        'stat_test_name': 'log_regression_R_glm_dsl_low_confidence',
                        'g1': group1_low_confidence_gt,
                        'g2': group2_low_confidence_gt,
                    },
                }

                if nr_of_bootstrap_samples > 1:
                    previous_sample_idx, group1_with_weights, group2_with_weights = previous_sample_idx_dict.get(bs_sample_idx, ([], group1, group2))
                elif is_burn_in_sample:
                    previous_sample_idx, group1_with_weights, group2_with_weights = [], group1, group2
                else:
                    previous_sample_idx, group1_with_weights, group2_with_weights = all_sample_idx_for_next_round, group1_with_weights, group2_with_weights
                group1_active_sample_BY_GT_CLASS, group2_active_sample_BY_GT_CLASS, group1_active_sample_and_llm_annotations_BY_GT_CLASS, group2_active_sample_and_llm_annotations_BY_GT_CLASS, group1_active_gt_BY_GT_CLASS, group2_active_gt_BY_GT_CLASS, all_sample_idx_for_next_round, group1_with_weights, group2_with_weights = get_active_samples(nr_of_known_gt, bs_sample_idx, group1_with_weights, group2_with_weights, previous_sample_idx, gt_classes)

                is_burn_in_sample = False

                # update the previous_sample_idx_dict for the next round of active sampling
                if nr_of_bootstrap_samples > 1:
                    previous_sample_idx_dict[bs_sample_idx] = (all_sample_idx_for_next_round, group1_with_weights, group2_with_weights)

                all_mitigation_techniques.update({

                    ##############################################
                    # MITIGATION 7: use only active samples of known ground truth labels
                    ##############################################
                    'ground_truth_sample_only_active_new': {
                        'stat_test_name': 'log_regression_R_glm',
                        'g1': group1_active_sample_BY_GT_CLASS,
                        'g2': group2_active_sample_BY_GT_CLASS,
                    },

                    ##############################################
                    # MITIGATION 8: use llm annotations PLUS active (i.e., driven by low confidence) sample of known ground truth labels
                    ##############################################
                    'ground_truth_sample_active_plus_annotations_new': {
                        'stat_test_name': 'log_regression_R_glm',
                        'g1': group1_active_sample_and_llm_annotations_BY_GT_CLASS,
                        'g2': group2_active_sample_and_llm_annotations_BY_GT_CLASS,
                    },

                    ##############################################
                    # MITIGATION 9: use llm annotations PLUS active (i.e., driven by low confidence) sample of known ground truth labels and correct estimates with CDI
                    ##############################################
                    'cdi_active_new': {
                        'stat_test_name': 'log_regression_R_glm_cdi_active',
                        'g1': group1_active_gt_BY_GT_CLASS,
                        'g2': group2_active_gt_BY_GT_CLASS,
                    },
                })

                for mitigation_technique, params in all_mitigation_techniques.items():
                    stat_test_name = params['stat_test_name']
                    g1 = params['g1']
                    g2 = params['g2']
                    stat_function, stat_function_params, name_of_corresp_stat_test_used_for_ground_truth = get_stat_info(stat_test_name)
                    # if mitigation_technique == 'cdi_active':
                    #     pdb.set_trace()
                    downstream_conclusions = stat_function(
                        g1,
                        g2,
                        gt_classes,
                        using_ground_truth_annotations=using_ground_truth_annotations,
                        **stat_function_params
                    )
                    # Update each conclusion with metadata
                    for conclusion in downstream_conclusions:
                        full_conclusion_dict = base_conclusion_dict.copy()
                        full_conclusion_dict.update({
                            'nr_of_known_gt': nr_of_known_gt,
                            'bs_sample_idx': bs_sample_idx,
                            'mitigation_technique': mitigation_technique,
                            'stat_test_name': stat_test_name,
                            'name_of_corresp_stat_test_used_for_ground_truth': name_of_corresp_stat_test_used_for_ground_truth,
                            'stat_function_params': str(stat_function_params),
                            'accuracy_on_random_sample': accuracy_on_random_sample,
                            **conclusion
                        })
                        downstream_conclusions_updated.append(full_conclusion_dict)
                    print(f'\n~~~~mitigation_technique: {mitigation_technique}~~~')
                print(f"done bs_sample_idx: {bs_sample_idx}")

        # all_downstream_conclusions.extend(downstream_conclusions_updated)
        print('done')
    
    return downstream_conclusions_updated

def statistical_downstream_analysis(dataset, gt_classes, fraction_na=None, using_ground_truth_annotations=False, max_percentage_of_known_gt=0.7, nr_of_known_ground_truth_samples_for_validation=[25, 50, 100, 250, 500, 1000], nr_of_bootstrap_samples=1):
    global GROUPINGS  # Access the global groupings
    
    all_downstream_conclusions = []
    tqdm_desc = 'GROUND TRUTH ANNOTATION' if using_ground_truth_annotations else 'LLM ANNOTATION'

    # nr_of_known_ground_truth_samples_for_validation must be ascending
    nr_of_known_ground_truth_samples_for_validation = sorted(nr_of_known_ground_truth_samples_for_validation)
    cols_to_keep = ['id', 'ground_truth', 'ground_truth_llm', 'response_mapped', 'response', 'output_mapping_description', 'model', 'params', 'prompt_description', 'seed', 'temperature', 'timestamp', 'source_path', 'verbalized_llm_confidence']
    cols_to_keep = [c for c in cols_to_keep if c in dataset.columns]
    
    # Pre-execute all grouping functions to get the actual groups
    print("\nPre-executing grouping functions...\n")
    executed_groupings = []
    for grouping_name, grouping_function, grouping_args in GROUPINGS:
        try:
            group1, group2, group1_name, group2_name = grouping_function(dataset, **grouping_args)
            group1 = group1[cols_to_keep]
            group2 = group2[cols_to_keep]
            # Only include groupings that have sufficient data
            if len(group1) >= 2 and len(group2) >= 2:
                executed_groupings.append((grouping_name, group1, group2, group1_name, group2_name))
            else:
                print(f"Skipping grouping '{grouping_name}' due to insufficient data: group1={len(group1)}, group2={len(group2)}")
        except Exception as e:
            print(f"Error executing grouping '{grouping_name}': {e}")
            pdb.set_trace()
    
    print(f"Successfully executed {len(executed_groupings)} out of {len(GROUPINGS)} groupings")
    
    if NUM_CPUS == 1 or using_ground_truth_annotations:
        # Serial execution (original behavior)
        for grouping_name, group1, group2, group1_name, group2_name in tqdm(executed_groupings, desc=f'Processing [{tqdm_desc}] groupings', leave=True, total=len(executed_groupings)):
            args_tuple = (grouping_name, group1, group2, group1_name, group2_name, gt_classes, fraction_na, using_ground_truth_annotations)
            results = process_single_grouping(args_tuple, nr_of_known_ground_truth_samples_for_validation, max_percentage_of_known_gt, nr_of_bootstrap_samples)
            all_downstream_conclusions.extend(results)
    else:
        # Parallel execution - now much simpler since we only pass DataFrames and strings
        grouping_args_list = [
            (grouping_name, group1, group2, group1_name, group2_name, gt_classes, fraction_na, using_ground_truth_annotations)
            for grouping_name, group1, group2, group1_name, group2_name in executed_groupings
        ]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
            # Use same approach for both ground truth and LLM annotations
            futures = [executor.submit(process_single_grouping, args, nr_of_known_ground_truth_samples_for_validation, max_percentage_of_known_gt, nr_of_bootstrap_samples) for args in grouping_args_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f'Processing [{tqdm_desc}] groupings in {NUM_CPUS} parallel cpus'):
                try:
                    results = future.result()
                    all_downstream_conclusions.extend(results)
                except Exception as e:
                    print(f"Error in parallel execution: {e}")
                    # Stop all execution as requested
                    executor.shutdown(wait=False)
                    raise
    
    return all_downstream_conclusions

def check_annotation_reliability(performance_results):
    df = pd.DataFrame(performance_results)
    df = df[df['fraction_na'] <= FRACTION_NA_REMOVAL_THRESHOLD]

    results = []

    metrics_to_check_for_reliability = ['accuracy', 'f1_score', 'f1_score_weighted']
    
    # Overall reliability (across all models)
    for metric in metrics_to_check_for_reliability:
        values = df[metric].tolist()
        if len(values) > 1:
            results.append({
                'reliability_level': 'overall',
                'model_name': None,
                'metric': metric,
                'abs_diff': annotation_reliability_abs_diff(values),
                'variance': annotation_reliability_variance(values),
                'stdev': annotation_reliability_stdev(values),
            })
    
    # Within model reliability (prompt variation at temperature=0)
    for model in df['model_name'].unique():
        model_df = df[(df['model_name'] == model) & (df['temperature'] == 0)]
        for metric in metrics_to_check_for_reliability:
            values = model_df[metric].tolist()
            if len(values) > 1:
                results.append({
                    'reliability_level': 'prompt_variation',
                    'model_name': model,
                    'metric': metric,
                    'abs_diff': annotation_reliability_abs_diff(values),
                    'variance': annotation_reliability_variance(values),
                    'stdev': annotation_reliability_stdev(values),
                })
    
    # Sampling variation (within model and prompt across temperatures)
    for model in df['model_name'].unique():
        for prompt in df['prompt'].unique():
            subset = df[(df['model_name'] == model) & (df['prompt'] == prompt)]
            for metric in metrics_to_check_for_reliability:
                values = subset[metric].tolist()
                if len(values) > 1:
                    results.append({
                        'reliability_level': 'sampling_variation',
                        'model_name': model,
                        'prompt': prompt,
                        'metric': metric,
                        'abs_diff': annotation_reliability_abs_diff(values),
                        'variance': annotation_reliability_variance(values),
                        'stdev': annotation_reliability_stdev(values),
                    })
    
    return results


def check_downstream_analysis_reliability(list_of_downstream_conclusions):
    # Calculate reliability metrics for downstream conclusions
    results = []

    df = pd.DataFrame(list_of_downstream_conclusions)
    if 'fraction_na' in df.columns:
        df = df[df['fraction_na'] <= FRACTION_NA_REMOVAL_THRESHOLD]
    if 'stat_test_name_llm' in df.columns:
        df.rename(columns={"stat_test_name_llm": "stat_test_name"}, inplace=True)
    if 'grouping_name_llm' in df.columns:
        df.rename(columns={"grouping_name_llm": "grouping_name"}, inplace=True)
    if 'p_value_llm' in df.columns:
        df.rename(columns={"p_value_llm": "p_value"}, inplace=True)
    if 'effect_size_llm' in df.columns:
        df.rename(columns={"effect_size_llm": "effect_size"}, inplace=True)
    if 'test_statistic_llm' in df.columns:
        df.rename(columns={"test_statistic_llm": "test_statistic"}, inplace=True)

    for stat_test_name in df['stat_test_name'].unique():
        df_subset = df[df['stat_test_name'] == stat_test_name]
        if stat_test_name in ['log_regression_R_glm', 'log_regression_R_glm_dsl',' log_regression_R_glm_dsl_low_confidence']:
            # Group by unique downstream analysis parameters
            grouped = df_subset.groupby(
                ['stat_test_name', 'grouping_name', 'group1_name', 'group2_name', 'class_name']
            )
            for name, group in grouped:
                p_values = group['p_value'].dropna().tolist()
                effect_sizes = group['effect_size'].dropna().tolist()
                test_statistics = group['test_statistic'].dropna().tolist()
                if len(p_values) > 1:
                    results.append({
                        'stat_test_name': name[0],
                        'grouping_name': name[1],
                        'group1_name': name[2],
                        'group2_name': name[3],
                        'class_name': name[4],
                        'p_value_abs_diff': annotation_reliability_abs_diff(p_values),
                        'p_value_variance': annotation_reliability_variance(p_values),
                        'p_value_stdev': annotation_reliability_stdev(p_values),
                        'effect_size_abs_diff': annotation_reliability_abs_diff(effect_sizes),
                        'effect_size_variance': annotation_reliability_variance(effect_sizes),
                        'effect_size_stdev': annotation_reliability_stdev(effect_sizes),
                        'test_statistic_abs_diff': annotation_reliability_abs_diff(test_statistics),
                        'test_statistic_variance': annotation_reliability_variance(test_statistics),
                        'test_statistic_stdev': annotation_reliability_stdev(test_statistics),
                    })
            return results
        else:
            print(f'downstream_conclusion_reliability functionality for stat_test_name={stat_test_name} NOT YET IMPLEMENTED!')
            pdb.set_trace()


def assess_llm_hacking(row):
    # Calculate errors
    correct_conclusion = is_correct_conclusion(row)
    type_I_error = is_type_I_error(row)
    type_II_error = is_type_II_error(row)
    type_S_error = is_type_S_error(row)
    assert sum([correct_conclusion, type_I_error, type_II_error, type_S_error]) == 1, 'sum is not 1'
    if correct_conclusion:
        return 'correct'
    elif type_I_error:
        return 'type_I_error'
    elif type_II_error:
        return 'type_II_error'
    elif type_S_error:
        return 'type_S_error'
    else:
        pdb.set_trace()
        

def compare_downstream_analysis_with_ground_truth(ground_truth_downstream_analysis_results, llm_downstream_results):
    # Convert to DataFrames for easier matching
    gt_df = pd.DataFrame(ground_truth_downstream_analysis_results)
    llm_df = pd.DataFrame(llm_downstream_results)
    
    # Define matching columns
    match_cols = ['name_of_corresp_stat_test_used_for_ground_truth', 'stat_function_params', 'grouping_name', 'group1_name', 'group2_name', 'class_name']

    for col in match_cols:
        gt_df[col] = gt_df[col].str.lower()
        llm_df[col] = llm_df[col].str.lower()
    
    # Merge on matching columns
    merged = gt_df.merge(llm_df, on=match_cols, suffixes=('_gt', '_llm'), how='inner')
    
    results = []
    for _, row in merged.iterrows():
        # For DSL/CDI, take the effect direction from the corrected test statistic
        # (test_statistic_llm) rather than the raw LLM proportions. Only the direction of a
        # significant conclusion is affected.
        est = str(row.get('stat_test_name_llm', ''))
        if (('dsl' in est or 'cdi' in est)
                and row.get('conclusion_llm') in ('group1_larger', 'group2_larger')
                and pd.notna(row.get('test_statistic_llm'))):
            row['conclusion_llm'] = 'group1_larger' if row['test_statistic_llm'] > 0 else 'group2_larger'
        # result = {col: row[col] for col in match_cols}
        result = row.to_dict()
        result['llm_hacking'] = assess_llm_hacking(row)

        # Calculate effect size errors
        if pd.notna(row['effect_size_gt']) and pd.notna(row['effect_size_llm']):
            result['effect_size_error'] = downstream_conclusion_effect_error(row)
            result['effect_size_error_absolute'] = downstream_conclusion_effect_error_absolute(row)
        else:
            result['effect_size_error'] = None
            result['effect_size_error_absolute'] = None
        
        # Add proportion errors if they exist
        if 'group_1_proportion_gt' in row:
            result['group_1_proportion_error'] = row['group_1_proportion_llm'] - row['group_1_proportion_gt']
            result['group_2_proportion_error'] = row['group_2_proportion_llm'] - row['group_2_proportion_gt']
        
        results.append(result)
    
    return results


def ground_truth_downstream_analysis(task_name, data_loader, dataset):
    global GROUPINGS  # Access the global groupings
    
    gt_classes = list(dataset['ground_truth'].unique())
    ground_truth_downstream_analysis_results = statistical_downstream_analysis(
        dataset, gt_classes, using_ground_truth_annotations=True
    )
    
    # save to disk
    save_csv_to_disk(ground_truth_downstream_analysis_results, f'{task_name}/ground_truth_downstream_analysis.csv')
    return ground_truth_downstream_analysis_results



def process_single_model_param_combo(args_tuple):
    """Process a single model/param combination."""
    task_name, model_name, prompt, temperature, seed, dataset_ground_truth, datasets_llm_annotated, ground_truth_downstream_analysis_results, gt_classes = args_tuple
    
    gt_ids = set(dataset_ground_truth['id'])
    llm_data = datasets_llm_annotated[
        (datasets_llm_annotated['model'] == model_name) &
        (datasets_llm_annotated['prompt_description'] == prompt) &
        (datasets_llm_annotated['temperature'] == temperature) &
        (datasets_llm_annotated['seed'] == seed) &
        (datasets_llm_annotated['id'].isin(gt_ids))
    ]

    llm_data = llm_data.sort_values('response_mapped', key=lambda x: x.str.lower() == 'na').drop_duplicates(subset='id', keep='first')
    
    llm_ids = set(llm_data['id'])
    llm_data_is_complete = (gt_ids == llm_ids) and (len(gt_ids) == len(dataset_ground_truth) == len(llm_data))
    
    if not llm_data_is_complete:
        print(f'Skipping incomplete LLM data: model={model_name}, prompt={prompt}, temp={temperature}, seed={seed}')
        return None, None
    
    llm_data_merged = dataset_ground_truth.merge(llm_data, on='id', suffixes=('', '_llm'))
    
    performance_values = compare_annotations_with_ground_truth(dataset_ground_truth, llm_data_merged)
    performance_result = {
        'model_name': model_name,
        'prompt': prompt,
        'temperature': temperature,
        'seed': seed,
        **performance_values
    }
    
    fraction_na = performance_values['fraction_na']
    llm_downstream_results = statistical_downstream_analysis(llm_data_merged, gt_classes, fraction_na=fraction_na)
    
    comparison_results = compare_downstream_analysis_with_ground_truth(
        ground_truth_downstream_analysis_results, llm_downstream_results
    )
    
    comparison_results_updated = []
    for result in comparison_results:
        full_comparison_dict = {
            'model_name': model_name,
            'prompt': prompt,
            'temperature': temperature,
            'seed': seed,
        }
        full_comparison_dict.update(result)
        comparison_results_updated.append(full_comparison_dict)
    
    return performance_result, comparison_results_updated

def evaluate_llm_annotations(task_name, data_loader, dataset_ground_truth, datasets_llm_annotated, ground_truth_downstream_analysis_results, prompt_descriptions):
    global total_number_of_combos_by_task
    global total_number_of_completed_combos_by_task
    gt_classes = [gt_c.strip().lower() for gt_c in list(dataset_ground_truth['ground_truth'].unique())]

    all_output_mappings = data_loader.get_all_output_mappings()
    mapping_classes = set(all_output_mappings[list(all_output_mappings.keys())[0]].get('params', {})['mapping_options'].values())
    mapping_classes = {m_c.strip().lower() for m_c in mapping_classes}
    all_mapping_classes_not_in_gt = mapping_classes - set(gt_classes)
    
    llm_classes = set([c for c in list(datasets_llm_annotated['response_mapped'].unique()) if c != 'na'])
    llm_classes = {llm_c.strip().lower() for llm_c in llm_classes}
    all_llm_classes_not_in_gt = llm_classes - set(gt_classes)

    # Load checkpoint
    completed_combos, relevant_task_dirs = load_checkpoint(task_name)
    
    unique_model_and_param_data = datasets_llm_annotated[
        ['model', 'prompt_description', 'temperature', 'seed']
    ].drop_duplicates().values.tolist()
    
    # Filter out already completed combinations
    remaining_combos = [
        combo for combo in unique_model_and_param_data
        if get_combo_key(*combo) not in completed_combos
        and combo[0] in models_list
    ]

    if prompt_descriptions is not None:
        # check if prompt_descriptions is in remaining_combos
        if any(combo[1] == prompt_descriptions for combo in remaining_combos):
            remaining_combos = [combo for combo in remaining_combos if combo[1] == prompt_descriptions]


    if only_check_number_of_missing_combos:
        all_models = [combo[0] for combo in unique_model_and_param_data]
        assert set(all_models) == set(models_list), f'Not checking all models!\n\nmodels_list: {models_list}\n\nall_models: {all_models}'
        # save info of how many 
        if task_name not in total_number_of_combos_by_task:
            total_number_of_combos_by_task[task_name] = 0
        if task_name not in total_number_of_completed_combos_by_task:
            total_number_of_completed_combos_by_task[task_name] = 0
        
        total_number_of_combos_by_task[task_name] += len(unique_model_and_param_data)
        total_number_of_completed_combos_by_task[task_name] += len(completed_combos)

    
    if len(remaining_combos) == 0:
        print(f"All combinations already processed for {task_name}")
    elif only_check_number_of_missing_combos:
        print(f"Remaining combinations for {task_name}: {len(remaining_combos)} out of {len(unique_model_and_param_data)} total")
        print(f"Total combinations processed: {total_number_of_completed_combos_by_task[task_name]} out of {total_number_of_combos_by_task[task_name]} total")
    else:
        print(f"Processing {len(remaining_combos)} remaining combinations out of {len(unique_model_and_param_data)} total")
        
        # Create task-specific temp directory
        task_temp_path = temp_results_path / task_name
        task_temp_path.mkdir(parents=True, exist_ok=True)

        global GROUPINGS
        NUM_CPUS_FOR_unique_model_and_param = NUM_CPUS // len(GROUPINGS)
        NUM_CPUS_FOR_unique_model_and_param = 1
        
        # Process remaining combinations
        if NUM_CPUS_FOR_unique_model_and_param < 2:
            # Serial execution
            for combo in tqdm(remaining_combos, desc='Processing unique_model_and_param combos', leave=True):
                model_name, prompt, temperature, seed = combo
                args_tuple = (task_name, model_name, prompt, temperature, seed, dataset_ground_truth, 
                            datasets_llm_annotated, ground_truth_downstream_analysis_results, gt_classes)
                performance_result, comparison_results_updated = process_single_model_param_combo(args_tuple)
                
                if performance_result is not None:
                    # Save intermediate results
                    combo_key = get_combo_key(model_name, prompt, temperature, seed)
                    combo_hash = hash(combo_key) % 1000000
                    
                    with open(task_temp_path / f'performance_{combo_hash}.json', 'w') as f:
                        json.dump(performance_result, f, cls=NumpyEncoder)
                    with open(task_temp_path / f'comparison_{combo_hash}.json', 'w') as f:
                        json.dump(comparison_results_updated, f, cls=NumpyEncoder)

                    # Update checkpoint
                    completed_combos.add(combo_key)
                    save_checkpoint(task_name, completed_combos)

        else:
            args_list = [
                (task_name, model_name, prompt, temperature, seed, dataset_ground_truth, 
                 datasets_llm_annotated, ground_truth_downstream_analysis_results, gt_classes)
                for model_name, prompt, temperature, seed in remaining_combos
            ]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPUS_FOR_unique_model_and_param) as executor:
                futures = {executor.submit(process_single_model_param_combo, args): args for args in args_list}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                                 desc='Processing unique_model_and_param combos'):
                    try:
                        performance_result, comparison_results_updated = future.result()
                        if performance_result is not None:
                            args = futures[future]
                            model_name, prompt, temperature, seed = args[1:5]
                            combo_key = get_combo_key(model_name, prompt, temperature, seed)
                            combo_hash = hash(combo_key) % 1000000
                            
                            with open(task_temp_path / f'performance_{combo_hash}.json', 'w') as f:
                                json.dump(performance_result, f, cls=NumpyEncoder)
                            with open(task_temp_path / f'comparison_{combo_hash}.json', 'w') as f:
                                json.dump(comparison_results_updated, f, cls=NumpyEncoder)
                            
                            completed_combos.add(combo_key)
                            save_checkpoint(task_name, completed_combos)
                    except Exception as e:
                        # Log which specific combination failed
                        args = futures[future]
                        model_name, prompt, temperature, seed = args[1:5]
                        print(f"Error processing combination {model_name}, {prompt}, {temperature}, {seed}: {e}")
                        executor.shutdown(wait=False)
                        raise
        
        # Update relevant_task_dirs to include the newly created directory
        if task_temp_path not in relevant_task_dirs:
            relevant_task_dirs.append(task_temp_path)

    if len(completed_combos) == len(unique_model_and_param_data) and (not (only_check_number_of_missing_combos and do_not_save_results_if_only_checking_number_of_missing_combos)):
        # Aggregate all results from all relevant directories (including newly processed ones)
        performance_results, llm_downstream_analysis_results = aggregate_all_results(task_name, relevant_task_dirs)

        # Save final aggregated results
        save_csv_to_disk(performance_results, f'{task_name}/performance_results.csv')
        save_csv_to_disk(llm_downstream_analysis_results, f'{task_name}/llm_downstream_analysis_results.csv')
        
        # Calculate reliability metrics
        performance_reliability_results = check_annotation_reliability(performance_results)
        save_csv_to_disk(performance_reliability_results, f'{task_name}/performance_reliability_results.csv')

        # Use the already loaded llm_downstream_analysis_results instead of loading files again
        downstream_analysis_reliability_results = check_downstream_analysis_reliability(llm_downstream_analysis_results)
        save_csv_to_disk(downstream_analysis_reliability_results, f'{task_name}/downstream_analysis_reliability_results.csv')

    else:
        print(f'Only completed {len(completed_combos)} of {len(unique_model_and_param_data)} combos!')

def load_all_pooled_results(results_path, task_name=None):
    # Load all pooled regression results
    print("Loading pooled regression results...")
    all_pooled_results = []
    results_path = Path(results_path)
    
    if task_name is None:
        fp_glob = '*/*/all_pooled_regression_results*.csv'
    else:
        fp_glob = f'*/{task_name}/all_pooled_regression_results*.csv'

    # Find all CSV files in subdirectories
    for csv_file in results_path.glob(fp_glob):
        try:
            df = pd.read_csv(csv_file)
            all_pooled_results.append(df)
            print(f"  Loaded {len(df)} results from {csv_file.parent.name}")
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")
    
    if not all_pooled_results:
        print("No pooled regression results found!")
        pdb.set_trace()
    
    # Combine all results
    combined_df = pd.concat(all_pooled_results, ignore_index=True)
    print(f"\nTotal pooled regression results: {len(combined_df)}")
    return combined_df


def run_pooled_regressions(task_name, data_loader, dataset_ground_truth, datasets_llm_annotated, prompt_descriptions):
    global total_number_of_combos_by_task
    global total_number_of_completed_combos_by_task

    gt_classes = [gt_c.strip().lower() for gt_c in list(dataset_ground_truth['ground_truth'].unique())]

    all_output_mappings = data_loader.get_all_output_mappings()
    mapping_classes = set(all_output_mappings[list(all_output_mappings.keys())[0]].get('params', {})['mapping_options'].values())
    mapping_classes = {m_c.strip().lower() for m_c in mapping_classes}
    all_mapping_classes_not_in_gt = mapping_classes - set(gt_classes)
    
    llm_classes = set([c for c in list(datasets_llm_annotated['response_mapped'].unique()) if c != 'na'])
    llm_classes = {llm_c.strip().lower() for llm_c in llm_classes}
    all_llm_classes_not_in_gt = llm_classes - set(gt_classes)

    unique_model_and_param_data = datasets_llm_annotated[
        ['model', 'prompt_description', 'temperature', 'seed']
    ].drop_duplicates().values.tolist()
    
    all_pooled_reg_results = load_all_pooled_results(eval_results_outpath.parent, task_name)
    
    completed_combos = all_pooled_reg_results[
        ['model_name', 'prompt'] #, 'temperature', 'seed']
    ].drop_duplicates().values.tolist()
    
    unique_model_and_param_data = datasets_llm_annotated[
        ['model', 'prompt_description', 'temperature', 'seed']
    ].drop_duplicates().values.tolist()
    
    # Filter out already completed combinations
    remaining_combos = [
        combo for combo in unique_model_and_param_data
        if combo[:2] not in completed_combos
    ]

    if len(remaining_combos) == 0:
        print(f'Task {task_name} is done! No remaining_combos! :)')
        sys.exit()

    if prompt_descriptions is not None:
        # check if prompt_descriptions is in remaining_combos
        if any(combo[1] == prompt_descriptions for combo in remaining_combos):
            remaining_combos = [combo for combo in remaining_combos if combo[1] == prompt_descriptions]


    print(f"Processing {len(remaining_combos)} remaining combinations out of {len(unique_model_and_param_data)} total")
    
    # Create task-specific temp directory
    task_temp_path = temp_results_path / task_name
    task_temp_path.mkdir(parents=True, exist_ok=True)

    global GROUPINGS
    
    for combo in tqdm(remaining_combos, desc='Processing unique_model_and_param combos', leave=True):

        all_pooled_regression_results = []

        model_name, prompt, temperature, seed = combo

        gt_ids = set(dataset_ground_truth['id'])
        llm_data = datasets_llm_annotated[
            (datasets_llm_annotated['model'] == model_name) &
            (datasets_llm_annotated['prompt_description'] == prompt) &
            (datasets_llm_annotated['temperature'] == temperature) &
            (datasets_llm_annotated['seed'] == seed) &
            (datasets_llm_annotated['id'].isin(gt_ids))
        ]

        llm_data = llm_data.sort_values('response_mapped', key=lambda x: x.str.lower() == 'na').drop_duplicates(subset='id', keep='first')
        
        llm_ids = set(llm_data['id'])
        llm_data_is_complete = (gt_ids == llm_ids) and (len(gt_ids) == len(dataset_ground_truth) == len(llm_data))
        
        if not llm_data_is_complete:
            print(f'Skipping incomplete LLM data: model={model_name}, prompt={prompt}, temp={temperature}, seed={seed}')
            return None, None
        
        dataset = dataset_ground_truth.merge(llm_data, on='id', suffixes=('', '_llm'))

        cols_to_keep = ['id', 'ground_truth', 'ground_truth_llm', 'response_mapped', 'response', 'output_mapping_description', 'model', 'params', 'prompt_description', 'seed', 'temperature', 'timestamp', 'source_path', 'verbalized_llm_confidence']
        cols_to_keep = [c for c in cols_to_keep if c in dataset.columns]
        
        # Pre-execute all grouping functions to get the actual groups
        for grouping_name, grouping_function, grouping_args in GROUPINGS:
            group1, group2, group1_name, group2_name = grouping_function(dataset, **grouping_args)
            group1 = group1[cols_to_keep].copy()
            group2 = group2[cols_to_keep].copy()
            # Only include groupings that have sufficient data
            if len(group1) >= 2 and len(group2) >= 2:
                group1.loc[:, 'group'] = 0
                group2.loc[:, 'group'] = 1
                base_conclusion_dict = {
                    'task_name': task_name,
                    'model_name': model_name,
                    'prompt': prompt,
                    'grouping_name': grouping_name,
                    'group1_name': group1_name,
                    'group2_name': group2_name,
                }

                pooled_regression_result = pooled_logistic_regression(group1, group2, gt_classes, assess_llm_hacking)

                for result in pooled_regression_result:
                    res_dict = base_conclusion_dict.copy()
                    res_dict.update(**result)
                    all_pooled_regression_results.append(res_dict)

        # Keep trying until we successfully create a new folder
        while True:
            ts = get_experiment_start_date()
            fn = f'{task_name}/all_pooled_regression_results_{ts}.csv'

            if not os.path.exists(fn):
                break
            time.sleep(2)

        save_csv_to_disk(all_pooled_regression_results, fn)


def extract_probability_from_text(text):
    """Extract probability from text using regex patterns."""
    if pd.isna(text) or text is None:
        return None
    
    text = str(text).strip()
    
    # Pattern 1: Only contains a probability (0.0 to 1.0)
    only_prob_pattern = r'^0*\.?\d*\.?\d+$'
    if re.match(only_prob_pattern, text):
        try:
            prob = float(text)
            if 0 <= prob <= 1:
                return prob
            elif 1 < prob <= 100:
                return prob / 100
        except ValueError:
            pass
    
    # Pattern 2: Starts with a probability
    start_prob_pattern = r'^(0*\.?\d*\.?\d+)'
    match = re.match(start_prob_pattern, text)
    if match:
        try:
            prob = float(match.group(1))
            if 0 <= prob <= 1:
                return prob
            elif 1 < prob <= 100:
                return prob / 100
        except ValueError:
            pass
    
    # Pattern 3: Contains a probability anywhere (decimal between 0 and 1)
    decimal_pattern = r'0\.\d+'
    matches = re.findall(decimal_pattern, text)
    for match in matches:
        try:
            prob = float(match)
            if 0 <= prob <= 1:
                return prob
        except ValueError:
            continue
    
    # Pattern 4: Contains a number between 0 and 100
    number_pattern = r'\b(\d{1,3}(?:\.\d+)?)\b'
    matches = re.findall(number_pattern, text)
    for match in matches:
        try:
            num = float(match)
            if 0 <= num <= 1:
                return num
            elif 1 < num <= 100:
                return num / 100
        except ValueError:
            continue
    
    return None


def get_verbalized_llm_confidence(datasets_llm_annotated, task_name, path_dir="llm-hacking-private/results_llm_confidence"):
    """
    Load verbalized confidence scores and match them to annotations.
    """
    if datasets_llm_annotated.empty:
        return datasets_llm_annotated
    
    # Initialize confidence column
    
    # Get unique models
    unique_models = datasets_llm_annotated['model'].unique()
    
    confidence_results = []
    
    # Load confidence results for each model
    for model in unique_models:
        try:
            confidence_df = load_confidence_results(
                results_base_path=path_dir,
                task_name=task_name,
                model=model
            )
            
            if not confidence_df.empty:
                confidence_df['model'] = model
                confidence_results.append(confidence_df)
                
        except Exception as e:
            print(f"Could not load confidence results for {model}: {e}")
            pdb.set_trace()
            continue
    
    if not confidence_results:
        print(f"No confidence results found for task {task_name}")
        return datasets_llm_annotated
    
    # Combine all confidence results
    all_confidence_df = pd.concat(confidence_results, ignore_index=True)

    
    # Extract numerical probabilities
    all_confidence_df['verbalized_llm_confidence'] = all_confidence_df['confidence'].apply(
        extract_probability_from_text
    ).clip(0, 1)

    # ensure response_mapped is encoded as string in both dfs
    all_confidence_df['response_mapped'] = all_confidence_df['response_mapped'].astype(str)
    datasets_llm_annotated['response_mapped'] = datasets_llm_annotated['response_mapped'].astype(str)
    
    result = datasets_llm_annotated.merge(
        all_confidence_df[['id', 'response_mapped', 'model', 'verbalized_llm_confidence', 'confidence']],
        on=['id', 'response_mapped', 'model'],
        how='left',
        suffixes=('', '_conf')
    )
    
    # Handle duplicates if any (take mean, max, or first)
    if 'verbalized_llm_confidence_conf' in result.columns:
        print('duplicates')
        pdb.set_trace()
        # This means there were duplicates - handle them
        result = result.groupby(['id', 'response_mapped', 'model']).agg({
            'verbalized_llm_confidence': 'first',  # or 'mean' if you want average
            'confidence': 'first',  # Keep confidence column too
            # Include all other columns
            **{col: 'first' for col in result.columns 
               if col not in ['id', 'response_mapped', 'model', 'verbalized_llm_confidence', 'confidence']}
        }).reset_index()
    
    # Print summary statistics
    total_annotations = len(result)
    with_confidence = result['verbalized_llm_confidence'].notna().sum()
    
    print(f"Loaded verbalized confidence for {with_confidence}/{total_annotations} "
          f"annotations ({with_confidence/total_annotations*100:.1f}%)")
    
    if with_confidence/total_annotations < 0.5:
        print('verbalized_llm_confidence is nan for more than 50% of data!!!')
        pdb.set_trace()
    
    # Analyze the confidence values where verbalized_llm_confidence is nan
    na_confidence = result[result['verbalized_llm_confidence'].isna()]
    if not na_confidence.empty:
        print(f"Analysis of {len(na_confidence)} rows where verbalized_llm_confidence is NaN:")
        print(na_confidence['confidence'].value_counts(dropna=False))
    
    if with_confidence > 0:
        confidence_stats = result['verbalized_llm_confidence'].describe()
        print(f"Confidence statistics:\n{confidence_stats}")
    
    return result



def main():
    global GROUPINGS
    global total_number_of_combos_by_task
    global total_number_of_completed_combos_by_task
    
    log_progress(f'Starting evaluation with {NUM_CPUS} CPUs')
    
    # Main execution loop
    for task_name in tqdm(all_tasks, desc='Processing tasks', leave=True, total=len(all_tasks)):
        tn = task_name.strip()
        if tn == '':
            continue
        log_progress(f'Starting task: {tn}')
        
        data_loader = map_dataset_name_to_class(tn)
        dataset = data_loader.load_dataset()
        if args.prompt_version_index is not None:
            prompt_descriptions = [p['description'] for p in data_loader.get_prompts()]
            if args.prompt_version_index < len(prompt_descriptions):
                prompt_descriptions = prompt_descriptions[args.prompt_version_index]
            else:
                continue
        else:
            prompt_descriptions = None
        
        # Set the global groupings variable
        GROUPINGS, nr_of_default_groupings, nr_of_dataset_specific_groups = data_loader.get_groups(dataset)

        if args.only_run_pooled_logistic_regression:
            datasets_llm_annotated = load_llm_annotated_data(tn)
            run_pooled_regressions(
                tn, data_loader, dataset, datasets_llm_annotated, prompt_descriptions
            )
            print(f'Done with pooled regressions for task {task_name}')
        else:
            ground_truth_downstream_analysis_results = ground_truth_downstream_analysis(
                tn, data_loader, dataset
            )
            print('Finished ground truth downstream analysis')
            log_progress(f'Finished ground truth downstream analysis for {tn}')
            
            # load all llm annotated data
            datasets_llm_annotated = load_llm_annotated_data(tn)
            datasets_llm_annotated = get_verbalized_llm_confidence(datasets_llm_annotated, tn)
            
            if not datasets_llm_annotated.empty:
                evaluate_llm_annotations(
                    tn, data_loader, dataset, datasets_llm_annotated, ground_truth_downstream_analysis_results, prompt_descriptions
                )
            else:
                print(f'datasets_llm_annotated is empty')
                pdb.set_trace()

            if only_check_number_of_missing_combos:
                continue

            print(f'Finished task {tn} eval. Results saved to {Path(eval_results_outpath, tn)}')
            log_progress(f'Finished task {tn}')
    
    if args.only_run_pooled_logistic_regression:
        print('done with all pooled regressions')
        sys.exit()
        pdb.set_trace()
    
    if only_check_number_of_missing_combos:
        print('Done checking all (completed) combos for all tasks!')
        print(f'\n\n~~~total_number_of_combos_by_task: {total_number_of_combos_by_task}')
        print(f'\n\n~~~total_number_of_completed_combos_by_task: {total_number_of_completed_combos_by_task}')
        print(f'\n\n~~~SUM OF ALL {len(total_number_of_combos_by_task)} TASKS IN: total_number_of_combos_by_task: {sum(total_number_of_combos_by_task.values())}')
        print(f'\n\n~~~SUM OF ALL {len(total_number_of_completed_combos_by_task)} TASKS IN: total_number_of_completed_combos_by_task: {sum(total_number_of_completed_combos_by_task.values())}')
        # percentage completed
        print(f'\n\n~~~PERCENTAGE OF COMPLETED COMBOS: {sum(total_number_of_completed_combos_by_task.values()) / sum(total_number_of_combos_by_task.values()) * 100}%\n\n')
        # check all completed tasks
        print(f'Incomplete_tasks:')
        incomplete_tasks = []
        for task_name in all_tasks:
            if total_number_of_combos_by_task[task_name] != total_number_of_completed_combos_by_task[task_name]:
                incomplete_tasks.append(task_name)
                print(f'{task_name}: {total_number_of_combos_by_task[task_name] - total_number_of_completed_combos_by_task[task_name]}/{total_number_of_combos_by_task[task_name]} ({(total_number_of_combos_by_task[task_name] - total_number_of_completed_combos_by_task[task_name]) / total_number_of_combos_by_task[task_name] * 100:.2f}%)')
        print(f'{len(incomplete_tasks)} out of totally {len(all_tasks)} are incomplete!')
        pdb.set_trace()


    print(f'\n~~Finished eval for all tasks :):). Results saved to {eval_results_outpath}')
    log_progress('Evaluation completed for all tasks')
    
    # Clean up temp directory if empty
    if temp_results_path.exists() and not any(temp_results_path.iterdir()):
        temp_results_path.rmdir()

if __name__ == "__main__":
    main()