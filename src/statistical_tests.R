# Implementation using GLM (Generalized Linear Model)
regression_glm <- function(count, nobs, two.sided = TRUE) {
  # Create data in aggregated form
  successes <- count
  failures <- nobs - count
  group <- factor(c(0, 1)) # Group indicator

  # Fit GLM with binomial family
  glm_model <- glm(cbind(successes, failures) ~ group, family = binomial(link = "logit"))

  # Extract coefficient summary
  coef_summary <- summary(glm_model)$coefficients

  # The z-statistic for group effect is in the coefficient table
  z_stat_glm <- coef_summary["group1", "z value"]

  if (two.sided) {
    p_value <- coef_summary["group1", "Pr(>|z|)"]
  } else {
    # Calculate one-sided p-value
    p_value <- pnorm(-abs(z_stat_glm))
  }

  # Match prop.test sign convention:
  # prop.test: positive when p1 > p2
  # GLM: positive when group1 (second group) > group0 (first group)
  # So we need to negate the GLM z-stat to match prop.test
  z_stat <- -z_stat_glm

  return(list(z_stat = z_stat, p_value = p_value))
}


# Pooled logistic regression for testing LLM vs GT differences
pooled_logistic_regression <- function(data) {
  # Ensure proper data types
  data$y <- as.numeric(data$y)
  data$x <- as.numeric(data$x)
  data$s <- as.numeric(data$s)
  data$obs_id <- as.factor(data$obs_id)

  # Fit the pooled model with interaction
  # y ~ α + βx + γs + δ(x×s)
  model <- glm(y ~ x + s + x:s,
    data = data,
    family = binomial(link = "logit")
  )

  # Get robust clustered standard errors
  library(sandwich)
  library(lmtest)

  # Calculate clustered variance-covariance matrix
  vcov_cluster <- vcovCL(model, cluster = data$obs_id)

  # Get coefficients with clustered SEs
  coef_test <- coeftest(model, vcov = vcov_cluster)

  # Extract key statistics
  alpha <- coef_test["(Intercept)", "Estimate"]
  beta <- coef_test["x", "Estimate"] # GT effect
  gamma <- coef_test["s", "Estimate"] # Source effect
  delta <- coef_test["x:s", "Estimate"] # Interaction (difference in effects)

  # Calculate beta_LLM = beta_GT + delta
  beta_llm <- beta + delta

  # Standard errors
  se_beta <- coef_test["x", "Std. Error"]
  se_delta <- coef_test["x:s", "Std. Error"]

  # P-values
  p_beta <- coef_test["x", "Pr(>|z|)"]
  p_delta <- coef_test["x:s", "Pr(>|z|)"]

  # Calculate confidence interval for difference (delta)
  ci_delta_lower <- delta - 1.96 * se_delta
  ci_delta_upper <- delta + 1.96 * se_delta

  # Calculate SE for beta_LLM using delta method
  # Var(beta_LLM) = Var(beta) + Var(delta) + 2*Cov(beta, delta)
  cov_matrix <- vcov_cluster
  var_beta_llm <- cov_matrix["x", "x"] +
    cov_matrix["x:s", "x:s"] +
    2 * cov_matrix["x", "x:s"]
  se_beta_llm <- sqrt(var_beta_llm)

  # Test if effects are significantly different (H0: delta = 0)
  z_stat_delta <- delta / se_delta
  effects_differ <- p_delta < 0.05

  return(list(
    # Main effects
    beta_gt = beta,
    beta_llm = beta_llm,

    # Difference (interaction)
    delta = delta,
    delta_se = se_delta,
    delta_p_value = p_delta,
    delta_ci_lower = ci_delta_lower,
    delta_ci_upper = ci_delta_upper,

    # Standard errors
    se_beta_gt = se_beta,
    se_beta_llm = se_beta_llm,

    # Test results
    effects_differ_significantly = effects_differ,
    z_stat_delta = z_stat_delta,

    # Model coefficients for reference
    alpha = alpha,
    gamma = gamma,

    # Full coefficient table
    coef_table = as.data.frame(coef_test)
  ))
}

# Empirical Bayes using locfdr package
compute_empirical_bayes_probabilities <- function(data) {
  # Install and load required package if not already installed
  if (!require("locfdr")) {
    install.packages("locfdr")
    library(locfdr)
  }

  z_scores_original <- data$test_statistic_gt
  n_total <- length(z_scores_original)

  # Identify extreme z-scores
  extreme_mask <- abs(z_scores_original) >= 10
  n_extreme <- sum(extreme_mask)

  # Print fraction of extreme z scores
  fraction_extreme <- n_extreme / n_total
  cat(sprintf("Extreme z-scores (|z| >= 10): %d / %d (%.2f%%)\n", n_extreme, n_total, fraction_extreme * 100))

  # Filter to non-extreme z-scores for locfdr
  z_scores_filtered <- z_scores_original[!extreme_mask]

  # Run locfdr to estimate local FDR
  # nulltype = 1 means empirical null (estimated from data)
  tryCatch(
    {
      locfdr_result <- locfdr(z_scores_filtered, nulltype = 1, plot = 4, df = 15)

      # Extract local FDR values (probability of null hypothesis being true)
      local_fdr_filtered <- locfdr_result$fdr

      # Probability of effect existing = 1 - P(null is true)
      prob_effect_exists_filtered <- 1 - local_fdr_filtered

      # Initialize full-length vectors
      local_fdr <- numeric(n_total)
      prob_effect_exists <- numeric(n_total)

      # Assign values for non-extreme z-scores
      local_fdr[!extreme_mask] <- local_fdr_filtered
      prob_effect_exists[!extreme_mask] <- prob_effect_exists_filtered

      # For extreme z-scores, assign very high confidence
      local_fdr[extreme_mask] <- 0.0001 # Very low FDR
      prob_effect_exists[extreme_mask] <- 1 # Very high probability

      cat(sprintf("Successfully computed local FDR for %d observations\n", n_total))
    },
    error = function(e) {
      # If locfdr fails, fall back to a simpler approach
      warning(paste("locfdr failed:", e$message, "."))
    }
  )

  # Return results as a data frame
  result <- data.frame(
    task = data$task,
    grouping_name = data$grouping_name,
    group1_name = data$group1_name,
    group2_name = data$group2_name,
    class_name = data$class_name,
    z_scores = z_scores_original,
    local_fdr = local_fdr,
    prob_effect_exists = prob_effect_exists
  )

  return(result)
}
