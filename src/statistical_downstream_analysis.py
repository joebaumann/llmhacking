from statsmodels.stats.proportion import proportions_ztest
from src.statistical_downstream_analysis_and_hacking_mitigation_helpers import *
import numpy as np
import pandas as pd
import pdb
from src.helpers import binarize_labels


def run_log_regressions(group1_df_or_dict, group2_df_or_dict, gt_classes, using_ground_truth_annotations, implementation_version, significance_threshold=0.05, **kwargs):
    conclusions = []
    additional_results = {}
    
    # Handle binarization
    if len(gt_classes) > 2:
        binarized_classes = gt_classes
    else:
        binarized_classes = [gt_classes[0]]
    
    for class_name in binarized_classes:
        # Get annotations column
        if using_ground_truth_annotations:
            col_name = 'ground_truth'
        else:
            col_name = 'response_mapped'
        if isinstance(group1_df_or_dict, dict):
            # groups were constructed with active sampling and thus differ by class_name. get groups df for this class_name from dict
            group1 = group1_df_or_dict[class_name]
            group2 = group2_df_or_dict[class_name]
        else:
            # groups do not differ by class_name and are provided as df directly
            group1 = group1_df_or_dict
            group2 = group2_df_or_dict
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
            continue
        group1_total = len(group1)
        group2_total = len(group2)
        
        group_1_proportion = group1_positive / group1_total if group1_total > 0 else 0
        group_2_proportion = group2_positive / group2_total if group2_total > 0 else 0
        group_proportion_difference = group_1_proportion - group_2_proportion
        
        # Handle zero proportion cases
        if group_1_proportion == 0 or group_2_proportion == 0:
            if using_ground_truth_annotations:
                # Skip this evaluation for ground truth
                continue
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
                effect_size = group_proportion_difference
        else:
            # Run proportions z-test
            count = [group1_positive, group2_positive]
            nobs = [group1_total, group2_total]
            if implementation_version == 'log_regression_py':
                z_stat, p_value = proportions_ztest(count, nobs)
            elif implementation_version == 'log_regression_R_glm':
                z_stat, p_value = call_r_regression(count, nobs)
            elif implementation_version == 'log_regression_R_glm_dsl':
                data = pd.concat([group1, group2])
                data = binarize_labels(data, class_name)
                z_stat, p_value, additional_results = call_r_regression_dsl(data)
            elif implementation_version == 'log_regression_R_glm_dsl_low_confidence':
                data = pd.concat([group1, group2])
                data = binarize_labels(data, class_name)
                z_stat, p_value, additional_results = call_r_regression_dsl(data, sample_prob='sampling_probability')
            elif implementation_version == 'log_regression_R_glm_cdi_active':
                data = pd.concat([group1, group2])
                z_stat, p_value, additional_results = get_cdi_estimator(data)
                
            # Determine conclusion
            if z_stat is None or p_value is None:
                estimator_correction_method = 'CDI' if implementation_version == 'log_regression_R_glm_cdi_active' else 'DSL'
                print(f'{estimator_correction_method} estimation failed! continuing with next class')
                continue
            elif p_value < significance_threshold:
                # Effect direction is taken from the (bias-corrected) test statistic, not from
                # the raw LLM proportions. For DSL/CDI the corrected estimate can point the
                # opposite way to the naive response_mapped proportions, and it is the corrected
                # sign that is valid. All estimators share the convention z_stat > 0 <=> group1
                # larger, so this is identical to comparing proportions for the uncorrected tests.
                if z_stat > 0:
                    conclusion = "group1_larger"
                else:
                    conclusion = "group2_larger"
            else:
                conclusion = "no_difference"

            effect_size = group_proportion_difference

        conclusions.append({
            # dict keys required by all statistical tests
            'conclusion': conclusion,
            'p_value': p_value,
            'effect_size': effect_size,
            'test_statistic': z_stat,
            # optional dict keys
            'class_name': class_name,
            'group_1_proportion': group_1_proportion,
            'group_2_proportion': group_2_proportion,
            'group_proportion_difference': group_proportion_difference,
            'group_proportion_difference_abs': abs(group_proportion_difference),
            **additional_results
        })
    
    return conclusions



def log_regression_ground_truth_bootstrap(group1, group2, class_name, significance_threshold=0.05):

    result = {}

    col_name = 'ground_truth'

    if isinstance(group1, dict):
        pdb.set_trace()
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
        return None
    group1_total = len(group1)
    group2_total = len(group2)
    
    group_1_proportion = group1_positive / group1_total if group1_total > 0 else 0
    group_2_proportion = group2_positive / group2_total if group2_total > 0 else 0
    group_proportion_difference = group_1_proportion - group_2_proportion
    
    # Handle zero proportion cases
    if group_1_proportion == 0 or group_2_proportion == 0:
        # Skip this evaluation for ground truth
        return None
    else:
        # Run proportions z-test
        count = [group1_positive, group2_positive]
        nobs = [group1_total, group2_total]
        z_stat, p_value = call_r_regression(count, nobs)

        # Determine conclusion
        if p_value < significance_threshold:
            if group_1_proportion > group_2_proportion:
                conclusion = "group1_larger"
            else:
                conclusion = "group2_larger"
        else:
            conclusion = "no_difference"
        
        effect_size = group_proportion_difference
    
    return {
        # dict keys required by all statistical tests
        f'conclusion': conclusion,
        f'p_value': p_value,
        f'effect_size': effect_size,
        f'test_statistic': z_stat,
        # optional dict keys
        f'class_name': class_name,
        f'group_1_proportion': group_1_proportion,
        f'group_2_proportion': group_2_proportion,
        f'group_proportion_difference': group_proportion_difference,
        f'group_proportion_difference_abs': abs(group_proportion_difference),
    }



def log_regression_annotation_ensembple_mitigation(group1, group2, class_name, col_names_and_suffices=[('ground_truth', '_gt'),('response_mapped', '_llm')], significance_threshold=0.05):

    result = {}
    
    for col_name, suffix in col_names_and_suffices:
        if isinstance(group1, dict):
            pdb.set_trace()
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
            return None
        group1_total = len(group1)
        group2_total = len(group2)
        
        group_1_proportion = group1_positive / group1_total if group1_total > 0 else 0
        group_2_proportion = group2_positive / group2_total if group2_total > 0 else 0
        group_proportion_difference = group_1_proportion - group_2_proportion
        
        # Handle zero proportion cases
        if group_1_proportion == 0 or group_2_proportion == 0:
            if col_name in ['ground_truth', 'ground_truth_llm_majority_vote']:
                # Skip this evaluation for ground truth
                return None
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
                effect_size = group_proportion_difference
        else:
            # Run proportions z-test
            count = [group1_positive, group2_positive]
            nobs = [group1_total, group2_total]
            z_stat, p_value = call_r_regression(count, nobs)

            # Determine conclusion
            if p_value < significance_threshold:
                if group_1_proportion > group_2_proportion:
                    conclusion = "group1_larger"
                else:
                    conclusion = "group2_larger"
            else:
                conclusion = "no_difference"
            
            effect_size = group_proportion_difference
        
        result.update({
            # dict keys required by all statistical tests
            f'conclusion{suffix}': conclusion,
            f'p_value{suffix}': p_value,
            f'effect_size{suffix}': effect_size,
            f'test_statistic{suffix}': z_stat,
            # optional dict keys
            f'class_name{suffix}': class_name,
            f'group_1_proportion{suffix}': group_1_proportion,
            f'group_2_proportion{suffix}': group_2_proportion,
            f'group_proportion_difference{suffix}': group_proportion_difference,
            f'group_proportion_difference_abs{suffix}': abs(group_proportion_difference),
        })
    return result


def pooled_logistic_regression(group1, group2, gt_classes, assess_llm_hacking, significance_threshold=0.05):
    """
    Updated pooled logistic regression with complete implementation
    """
    conclusions = []
    
    # Handle binarization
    if len(gt_classes) > 2:
        binarized_classes = gt_classes
    else:
        binarized_classes = [gt_classes[0]]
    
    for class_name in binarized_classes:
        # Get basic proportions first (for descriptive stats)
        col_name_gt = 'ground_truth'
        col_name_llm = 'response_mapped'
        try:
            if set(group1[col_name_gt].unique()) == {0,1}:
                # Calculate counts and proportions, use sum cause labels are already binarized
                group1_positive_gt = group1[col_name_gt].sum()
                group2_positive_gt = group2[col_name_gt].sum()
            else:
                # Calculate counts and proportions
                group1_positive_gt = (group1[col_name_gt].astype(str).str.strip().str.lower() == class_name.lower()).sum()
                group2_positive_gt = (group2[col_name_gt].astype(str).str.strip().str.lower() == class_name.lower()).sum()
            if set(group1[col_name_llm].unique()) == {0,1}:
                # Calculate counts and proportions, use sum cause labels are already binarized
                group1_positive_llm = group1[col_name_llm].sum()
                group2_positive_llm = group2[col_name_llm].sum()
            else:
                # Calculate counts and proportions
                group1_positive_llm = (group1[col_name_llm].astype(str).str.strip().str.lower() == class_name.lower()).sum()
                group2_positive_llm = (group2[col_name_llm].astype(str).str.strip().str.lower() == class_name.lower()).sum()
        except Exception as e:
            print(e)
            pdb.set_trace()
            continue
        group1_total = len(group1)
        group2_total = len(group2)
        
        group_1_proportion_gt = group1_positive_gt / group1_total if group1_total > 0 else 0
        group_2_proportion_gt = group2_positive_gt / group2_total if group2_total > 0 else 0
        group_proportion_difference_gt = group_1_proportion_gt - group_2_proportion_gt

        group_1_proportion_llm = group1_positive_llm / group1_total if group1_total > 0 else 0
        group_2_proportion_llm = group2_positive_llm / group2_total if group2_total > 0 else 0
        group_proportion_difference_llm = group_1_proportion_llm - group_2_proportion_llm
        
        # Handle zero proportion cases
        if group_1_proportion_gt == 0 or group_2_proportion_gt == 0 or group_1_proportion_llm == 0 or group_2_proportion_llm == 0:
            continue
        else:
            # Run proportions z-tests
            count_gt = [group1_positive_gt, group2_positive_gt]
            count_llm = [group1_positive_llm, group2_positive_llm]
            nobs = [group1_total, group2_total]
            z_stat_gt, p_value_gt = proportions_ztest(count_gt, nobs)
            z_stat_llm, p_value_llm = proportions_ztest(count_llm, nobs)

            # Determine conclusion
            if p_value_gt < significance_threshold:
                if group_1_proportion_gt > group_2_proportion_gt:
                    conclusion_gt = "group1_larger"
                else:
                    conclusion_gt = "group2_larger"
            else:
                conclusion_gt = "no_difference"
            if p_value_llm < significance_threshold:
                if group_1_proportion_llm > group_2_proportion_llm:
                    conclusion_llm = "group1_larger"
                else:
                    conclusion_llm = "group2_larger"
            else:
                conclusion_llm = "no_difference"
            
            llm_hacking = assess_llm_hacking({
                'conclusion_gt': conclusion_gt,
                'conclusion_llm': conclusion_llm,
            })

        # Call pooled regression
        pooled_results = call_r_pooled_logistic(group1, group2, class_name)
        
        conclusions.append({
            'class_name': class_name,
            'group_1_proportion_gt': group_1_proportion_gt,
            'group_2_proportion_gt': group_2_proportion_gt,
            'group_1_proportion_llm': group_1_proportion_llm,
            'group_2_proportion_llm': group_2_proportion_llm,
            'group_proportion_difference_gt': group_proportion_difference_gt,
            'group_proportion_difference_abs_gt': abs(group_proportion_difference_gt),
            'group_proportion_difference_llm': group_proportion_difference_llm,
            'group_proportion_difference_abs_llm': abs(group_proportion_difference_llm),
            'z_stat_gt': z_stat_gt,
            'p_value_gt': p_value_gt,
            'z_stat_llm': z_stat_llm,
            'p_value_llm': p_value_llm,
            'llm_hacking': llm_hacking,
            
            # all results from pooled logistic regression!
            **pooled_results
        })
    
    return conclusions


def log_regression_R_glm(group1, group2, gt_classes, using_ground_truth_annotations, **kwargs):
    implementation_version = 'log_regression_R_glm'
    return run_log_regressions(group1, group2, gt_classes, using_ground_truth_annotations, implementation_version, **kwargs)

def log_regression_R_glm_dsl(group1, group2, gt_classes, using_ground_truth_annotations, **kwargs):
    implementation_version = 'log_regression_R_glm_dsl'
    return run_log_regressions(group1, group2, gt_classes, using_ground_truth_annotations, implementation_version, **kwargs)

def log_regression_R_glm_dsl_low_confidence(group1, group2, gt_classes, using_ground_truth_annotations, **kwargs):
    implementation_version = 'log_regression_R_glm_dsl_low_confidence'
    return run_log_regressions(group1, group2, gt_classes, using_ground_truth_annotations, implementation_version, **kwargs)

def log_regression_R_glm_cdi_active(group1, group2, gt_classes, using_ground_truth_annotations, **kwargs):
    implementation_version = 'log_regression_R_glm_cdi_active'
    return run_log_regressions(group1, group2, gt_classes, using_ground_truth_annotations, implementation_version, **kwargs)




all_statistical_tests = {
    ### STATISTICAL ANALYSES WITHOUT ERROR CORRECTION ###
    'log_regression_R_glm': {
        'stat_function': log_regression_R_glm,
        'stat_function_params': {},
    },
    ### STATISTICAL ANALYSES WITH ERROR CORRECTION ###
    # applying Design-based Supervised Learning (DSL) by Egami et al.
    'log_regression_R_glm_dsl': {
        'stat_function': log_regression_R_glm_dsl,
        'name_of_corresp_stat_test_used_for_ground_truth': 'log_regression_R_glm',
        'stat_function_params': {},
    },
    # dsl with low_confidence samples
    'log_regression_R_glm_dsl_low_confidence': {
        'stat_function': log_regression_R_glm_dsl_low_confidence,
        'name_of_corresp_stat_test_used_for_ground_truth': 'log_regression_R_glm',
        'stat_function_params': {},
    },
    # applying Confidence-Driven Inference by Gligoric et al.
    'log_regression_R_glm_cdi_active': {
        'stat_function': log_regression_R_glm_cdi_active,
        'name_of_corresp_stat_test_used_for_ground_truth': 'log_regression_R_glm',
        'stat_function_params': {},
    },
}