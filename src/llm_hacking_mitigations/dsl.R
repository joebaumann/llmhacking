# Generic function that executes dsl and returns all results
execute_dsl <- function(data, cross_fit = 5, sample_split = 10, ...) {
  # Ensure the dataframe has the required columns
  required_cols <- c("ground_truth_known", "response_mapped", "group")
  if (!all(required_cols %in% colnames(data))) {
    stop("Data must contain columns: ground_truth_known, response_mapped, group")
  }
  
  # Create binary group indicator (assuming two groups)
  unique_groups <- unique(data$group)
  if (length(unique_groups) != 2) {
    stop("Currently only supports two groups")
  }
  
  # Create binary group variable (0 for first group, 1 for second)
  # This matches the GLM convention where group effect represents group 1 vs group 0
  data$group_binary <- as.numeric(data$group == unique_groups[2])
  
  # Check for minimum labeled data
  labeled_count <- sum(!is.na(data$ground_truth_known))
  if (labeled_count < 4) {
    stop("Insufficient labeled data: need at least 4 labeled observations, found ", labeled_count)
  }
  
  # Define parameter values to try (start with defaults, then alternatives)
  cross_fit_values <- c(cross_fit, 4, 3, 2, 6, 7, 8, 9, 10)
  sample_split_values <- c(sample_split, 8, 6, 4, 3, 2, 12, 15)
  
  last_error <- NULL
  
  # Try all combinations
  for (cf in cross_fit_values) {
    for (ss in sample_split_values) {
      tryCatch({
        # Attempt to run DSL
        dsl_result <- dsl(
          model = "logit",
          formula = ground_truth_known ~ group_binary,
          predicted_var = "ground_truth_known",
          prediction = "response_mapped",
          data = data,
          cross_fit = cf,
          sample_split = ss,
          ...  # Pass any additional arguments to dsl()
        )
        # print(summary(dsl_result))
        
        # If successful, return result with info about parameters used
        if (cf != cross_fit || ss != sample_split) {
          warning("DSL succeeded with alternative parameters: cross_fit=", cf, 
                  ", sample_split=", ss, 
                  " (requested: cross_fit=", cross_fit, ", sample_split=", sample_split, ")")
        }
        
        # Return successful result with parameters
        # print("dsl_result:")
        # summary(dsl_result)
        return(list(
          error = FALSE,
          dsl_result = dsl_result,
          cross_fit = cf,
          sample_split = ss
        ))
        
      }, error = function(e) {
        last_error <<- e
        # Continue to next combination
      })
    }
  }
  
  # If we get here, all attempts failed
  return(list(
    error = TRUE, 
    message = paste("DSL failed with all parameter combinations. Last error: ", last_error$message, 
                   ". Try collecting more labeled data or use traditional statistical methods.")
  ))
}

# Function that calculates p_value and z_stat using GLM with DSL
regression_glm_dsl <- function(data, two.sided = TRUE, ...) {
  # Execute DSL with any additional parameters
  execute_result <- execute_dsl(data, ...)
  
  # Check if DSL execution failed
  if (execute_result$error) {
    warning("DSL execution failed: ", execute_result$message)
    return(list(
      z_stat = NULL,
      p_value = NULL,
      group_1_proportion_R = NA,
      group_2_proportion_R = NA,
      group_proportion_difference_R = NA,
      cross_fit = NA,
      sample_split = NA,
      dsl_full_result = NULL,
      error = TRUE,
      error_message = execute_result$message
    ))
  }
  
  # Extract successful results
  dsl_result <- execute_result$dsl_result
  cf <- execute_result$cross_fit
  ss <- execute_result$sample_split
  
  # Extract coefficient for group_binary (excluding intercept)
  coef_group <- dsl_result$coefficients["group_binary"]
  se_group <- dsl_result$standard_errors["group_binary"]
  
  # Calculate z-statistic
  # Note: DSL uses group_binary where 1 = second group, 0 = first group
  # To match prop.test convention (positive when p1 > p2), we need to negate
  z_stat <- -coef_group / se_group
  
  # IMPORTANT: DSL reports one-sided p-values, so we need to adjust
  if (two.sided) {
    # Calculate two-sided p-value
    p_value <- 2 * (1 - pnorm(abs(coef_group / se_group)))
  } else {
    # Calculate one-sided p-value
    p_value <- 1 - pnorm(abs(coef_group / se_group))
  }
  
  # Get group proportions from the model
  # For logistic regression: p = exp(β0 + β1*x) / (1 + exp(β0 + β1*x))
  intercept <- dsl_result$coefficients["(Intercept)"]
  
  # Group 0 proportion (first group, group_binary = 0)
  logit_group0 <- intercept
  group_0_proportion_R <- exp(logit_group0) / (1 + exp(logit_group0))
  
  # Group 1 proportion (second group, group_binary = 1)
  logit_group1 <- intercept + coef_group
  group_1_proportion_R <- exp(logit_group1) / (1 + exp(logit_group1))
  
  # Return results with groups correctly labeled
  return(list(
    z_stat = as.numeric(z_stat),
    p_value = as.numeric(p_value),
    group_1_proportion_R = as.numeric(group_0_proportion_R),
    group_2_proportion_R = as.numeric(group_1_proportion_R),
    group_proportion_difference_R = as.numeric(group_0_proportion_R - group_1_proportion_R),
    cross_fit = cf,
    sample_split = ss,
    dsl_full_result = dsl_result
  ))
}
