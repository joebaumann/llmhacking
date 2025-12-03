import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
import pdb
import pandas as pd
from scipy.stats import norm, bernoulli

from src.llm_hacking_mitigations.cdi_utils import *

# Enable pandas-R dataframe conversion
pandas2ri.activate()

# Load R scripts
r.source('src/statistical_tests.R')
r.source('src/llm_hacking_mitigations/dsl.R')

# Load the dsl package
r('library(dsl)')


def call_r_regression(count, nobs):
    try:
        # Convert Python lists to R vectors
        r_count = robjects.IntVector(count)
        r_nobs = robjects.IntVector(nobs)

        # Call the R function
        result = r.regression_glm(r_count, r_nobs)
        # print(r.str(result))

        # Extract results
        z_stat = result.rx2('z_stat')[0]
        p_value = result.rx2('p_value')[0]
    except Exception as e:
        print(f"Error in call_r_regression: {e}")
        pdb.set_trace()

    return z_stat, p_value


def call_r_regression_dsl(data, sample_prob=None):
    try:
        # Convert pandas DataFrame to R dataframe
        data_clean = data.infer_objects()
        r_data = pandas2ri.py2rpy(data_clean)

        # Call the R DSL function
        if sample_prob is not None:
            result = r.regression_glm_dsl(
                r_data, sample_prob=sample_prob)
        else:
            result = r.regression_glm_dsl(r_data)

        # print(r.str(result))

        # Check if DSL failed by looking for error indicator or None values
        if result is None:
            print("DSL returned None - failed")
            return None, None, {}
        
        # Extract results
        try:
            z_stat = result.rx2('z_stat')[0]
            p_value = result.rx2('p_value')[0]
        except Exception as e:
            print(f"DSL returned invalid z_stat or p_value. Exception: {e}")
            return None, None, {}
        
        # Check for invalid values
        if z_stat is None or p_value is None or pd.isna(z_stat) or pd.isna(p_value):
            print("DSL returned invalid z_stat or p_value")
            return None, None, {}
        
        # Extract and convert DSL result components to Python-friendly formats
        dsl_full_result = result.rx2('dsl_full_result')
        
        # Extract coefficients and standard errors safely
        coefficients = dsl_full_result.rx2('coefficients')
        standard_errors = dsl_full_result.rx2('standard_errors')
        rmse_matrix = dsl_full_result.rx2('RMSE')
        
        # Convert to Python lists/arrays
        coef_values = list(coefficients)
        se_values = list(standard_errors)
        
        # Get coefficient names if they exist, otherwise use indices
        try:
            coef_names = list(coefficients.names)
        except (AttributeError, TypeError):
            # Fallback: use generic names based on the logistic regression structure
            coef_names = ['(Intercept)', 'group_binary'] if len(coef_values) == 2 else [f'coef_{i}' for i in range(len(coef_values))]
        
        additional_results = {
            'group_1_proportion_R': result.rx2('group_1_proportion_R')[0],
            'group_2_proportion_R': result.rx2('group_2_proportion_R')[0],
            'group_proportion_difference_R': result.rx2('group_proportion_difference_R')[0],
            'cross_fit': result.rx2('cross_fit')[0],
            'sample_split': result.rx2('sample_split')[0],
            # Convert DSL result components to Python-friendly formats
            'dsl_coefficients': dict(zip(coef_names, coef_values)),
            'dsl_standard_errors': dict(zip(coef_names, se_values)),
            'dsl_rmse': float(rmse_matrix.mean()) if hasattr(rmse_matrix, 'mean') else float(rmse_matrix[0]),
        }
        
        return z_stat, p_value, additional_results
            
    except Exception as e:
        print(f"Error in call_r_regression_dsl: {e}")
        pdb.set_trace()


def get_cdi_estimator(data, alpha=0.05):
    try:
        # Prepare data
        n = len(data)
        data = data.reset_index(drop=True)
        
        # Get LLM annotations and confidence
        Yhat = data['response_mapped'].values.astype(int)
        
        # Create design matrix for logistic regression (group indicator)
        X = np.column_stack((data['group'].values, np.ones(n))).astype(float)
        
        # Get ground truth for known samples
        Y = data['ground_truth'].values.astype(int)

        # get weights_active from data
        weights_active = data['weights_active'].values

        pointest_found = False
        all_lam_init = iter([1, 0.9, 0.7, 0.4, 0.1, 0])
        lam_init = next(all_lam_init)
        lam_init_suffix = ''
        while not pointest_found and lam_init >= 0:
            try:
                # First get initial estimate with λ=1 (full trust in LLM annotations)
                pointest_init = active_logistic_pointestimate(X, Y, Yhat, weights_active, lam=lam_init)
                
                # Optimize λ to minimize variance (this is the "power tuning")
                lam = opt_logistic_tuning(pointest_init, X, Y, Yhat, weights_active)

                if lam < 0 or lam > 1:
                    print(f'clipping lam={lam}')
                    lam_clip_suffix = f' lam_unclipped={lam}'
                    lam = np.clip(lam, 0, 1)
                else:
                    lam_clip_suffix = ''
                
                # Get final estimate with optimized λ
                pointest = active_logistic_pointestimate(X, Y, Yhat, weights_active, lam=lam)
                
                # Calculate covariance matrix for the estimates
                Sigmahat = logistic_cov(pointest, X, Y, Yhat, weights_active, lam=lam)
                pointest_found = True
            except Exception as e:
                print(f'CDI estimator failed with lam_init={lam_init} and Exception: {e}')
                lam_init = next(all_lam_init, None)
                if lam_init is None:
                    print('All lambda initialization values exhausted')
                    break
                lam_init_suffix = f' lam_init={lam_init}'
                print(f'Try with new lam_init={lam_init}')
        
        if not pointest_found:
            try:
                lam=0
                pointest = active_logistic_pointestimate(X, Y, Yhat, weights_active, lam=lam)
                Sigmahat = logistic_cov(pointest, X, Y, Yhat, weights_active, lam=lam)
                if Sigmahat[0, 0] < 0:
                    return None, None, {}
                method_used = 'CDI with lam=0 fallback'
                print(method_used)
            except Exception as e:
                print(f'CDI estimator failed FULLY EVEN WITH lam=0 and Exception: {e}')
                import traceback
                traceback.print_exc()
                print('bummer..')
                # pdb.set_trace()
                return None, None, {}
        else:
            method_used = f'CDI{lam_clip_suffix}{lam_init_suffix}'
        
        # Extract coefficient for group effect (first coefficient)
        group_coef = pointest[0]
        # get standard error
        if Sigmahat[0, 0] < 0:
            if lam == 0:
                return None, None, {}
            print(Sigmahat, Sigmahat[0, 0], Sigmahat[1, 1])
            print('negative variance: fallback to lam=0')
            try:
                lam=0
                pointest = active_logistic_pointestimate(X, Y, Yhat, weights_active, lam=lam)
                Sigmahat = logistic_cov(pointest, X, Y, Yhat, weights_active, lam=lam)
                group_coef = pointest[0]
                method_used += ' - negative variance: fallback to lam=0'
                if Sigmahat[0, 0] < 0:
                    print(f"Warning: Variance still negative after lam=0 fallback: {Sigmahat[0, 0]}")
                    # This shouldn't happen with lam=0
                    return None, None, {}
            except Exception as e:
                print(f'CDI estimator failed FULLY EVEN WITH lam=0 and Exception2: {e}')
                import traceback
                traceback.print_exc()
                print('bummer2..')
                # pdb.set_trace()
                return None, None, {}

        group_se = np.sqrt(Sigmahat[0, 0] / n)
        
        # Calculate z-statistic and p-value
        z_stat = -group_coef / group_se  # Negate to match GLM convention
        p_value = 2 * norm.cdf(-abs(z_stat))  # Two-sided test
        
        # Extract confidence interval for first coefficient (device effect)
        # I.e., in OUR setup with binary group indicator, this is be the group effect
        l = pointest - norm.ppf(1-alpha/2)*np.sqrt(np.diag(Sigmahat)/n)  # Lower bound
        u = pointest + norm.ppf(1-alpha/2)*np.sqrt(np.diag(Sigmahat)/n)  # Upper bound

        true_pointest = logistic(X,Y)
        true_effect = true_pointest[0]  # first coordinate is effect of device
        # Check if true effect is covered by confidence interval
        true_coverage = (true_effect >= l[0])*(true_effect <= u[0]) # Does interval contain truth?
        true_Sigma = logistic_cov(true_pointest, X, Y, Yhat, np.ones(n), lam=0)
        true_z_stat = -true_effect / np.sqrt(true_Sigma[0, 0] / n)
        true_p_value = 2 * norm.cdf(-abs(true_z_stat))

        # Only use labeled points for calculations
        labeled_mask = weights_active > 0
        # ESS calculation
        Y_labeled = Y[labeled_mask]
        ess = float(np.sum(Y_labeled**2)/(np.var(Y_labeled) + np.mean(Y_labeled)**2)*n) if len(Y_labeled) > 0 else 0
        
        additional_results = {
            'cdi_coefficients': {'group': pointest[0], 'intercept': pointest[1]},
            'cdi_standard_errors': {'group': group_se, 'intercept': np.sqrt(Sigmahat[1, 1] / n)},
            'cdi_lambda': float(lam),
            'cdi_n_labeled': int(np.sum(labeled_mask)),
            'lb': l[0],
            'ub': u[0],
            'interval width': u[0]-l[0],
            'Sigmahat': Sigmahat,
            'Sigmahat2': Sigmahat[0, 0],
            'Sigmahat3': np.diag(Sigmahat),
            'n': n,
            'true_pointest': true_pointest,
            'true_effect': true_effect,
            'true_coverage': true_coverage,
            'true_Sigma': true_Sigma,
            'true_z_stat': true_z_stat,
            'true_p_value': true_p_value,
            'cdi_effective_sample_size': ess,
            'method_used': method_used,
        }

        print('cdi done')
        return z_stat, p_value, additional_results
        
    except Exception as e:
        print(f"Error in get_cdi_estimator: {e}")
        import traceback
        traceback.print_exc()
        pdb.set_trace()
        return None, None, {}



def call_r_pooled_logistic(group1, group2, class_name):
    """
    Call R pooled logistic regression function
    """
    try:
        # Prepare data for stacking
        group1_data = group1.copy()
        group2_data = group2.copy()
        
        # Add group indicator (x)
        group1_data['x'] = 0
        group2_data['x'] = 1
        
        # Combine groups
        combined = pd.concat([group1_data, group2_data], ignore_index=True)
        
        # Stack data: create two rows per observation (GT and LLM)
        stacked_data = []
        
        for idx, row in combined.iterrows():
            # Ground truth row
            if len(set(combined['ground_truth'].unique())) == 2 and 0 in combined['ground_truth'].unique():
                # Already binarized
                y_gt = row['ground_truth']
            else:
                # Binarize for specific class
                y_gt = 1 if str(row['ground_truth']).strip().lower() == class_name.lower() else 0
            
            stacked_data.append({
                'y': y_gt,
                'x': row['x'],
                's': 0,  # Ground truth indicator
                'obs_id': row['id']
            })
            
            # LLM row
            if len(set(combined['response_mapped'].unique())) == 2 and 0 in combined['response_mapped'].unique():
                # Already binarized
                y_llm = row['response_mapped']
            else:
                # Binarize for specific class
                y_llm = 1 if str(row['response_mapped']).strip().lower() == class_name.lower() else 0
            
            stacked_data.append({
                'y': y_llm,
                'x': row['x'],
                's': 1,  # LLM indicator
                'obs_id': row['id']
            })
        
        stacked_df = pd.DataFrame(stacked_data)
        
        # Convert to R dataframe
        r_data = pandas2ri.py2rpy(stacked_df)
        
        # Call R function
        result = r.pooled_logistic_regression(r_data)
        
        # Extract results
        output = {}
        for k in [
            'beta_gt',
            'beta_llm',
            'delta',
            'delta_se',
            'delta_p_value',
            'delta_ci_lower',
            'delta_ci_upper',
            'se_beta_gt',
            'se_beta_llm',
            'effects_differ_significantly',
            'z_stat_delta',
            ]:
            try:
                if k == 'effects_differ_significantly':
                    val = bool(result.rx2(k)[0])
                else:
                    val = result.rx2(k)[0]
            except Exception as e:
                print(f'Exceptionnnn: {e}')
                val = None
            output[k] = val

        return output
        
    except Exception as e:
        print(f"Error in pooled logistic regression: {e}")
        import traceback
        traceback.print_exc()
        pdb.set_trace()
        return None


def call_r_empirical_bayes(model_data, merge_cols):
    """
    Call R function to compute empirical Bayes probabilities of effect existence.
    
    Args:
        model_data: DataFrame with hypothesis test results
        
    Returns:
        DataFrame with local FDR and probability of effect existence
    """
    # Prepare data - select relevant columns for R
    cols_to_keep = list(set(merge_cols + ['task', 'grouping_name', 'group1_name', 'group2_name', 'class_name', 'p_value_gt', 'test_statistic_gt','group_1_proportion_gt', 'group_2_proportion_gt']))
    data_for_r = model_data[cols_to_keep].copy()
    data_for_r = data_for_r.drop_duplicates(subset=merge_cols, keep='first')
    nr_of_z_values = len(data_for_r)
    
    # Convert to R dataframe
    r_data = pandas2ri.py2rpy(data_for_r)
    
    # Call R function
    result = r.compute_empirical_bayes_probabilities(r_data)
    
    # Convert back to pandas DataFrame
    result_df = pandas2ri.rpy2py(result)

    # Ensure class_name is string type in result as well
    result_df['class_name'] = result_df['class_name'].astype(str)

    return result_df

