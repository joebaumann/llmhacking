library(dataverse)
#Sys.setenv(DATAVERSE_SERVER = "dataverse.harvard.edu")

data <- get_dataframe_by_name(
  filename = "application2-carlson-montgomery-2017.tab",
  dataset  = "10.7910/DVN/DZZ0OM",
  server   = "dataverse.harvard.edu",
  original = TRUE,
  .f       = read.csv
)
data <- data[c("text","tone")]
colnames(data)[2] <- "ground_truth"

# get value_counts for ground_truth column
table(data$ground_truth, useNA = "always")

# Apply ground truth mapping of CARLSON & MONTGOMERY (2017)
# ad tone is determined by expert coders who categorized ads as either promoting
# a single candidate, contrasting two candidates, or attacking a candidate. If 
# the ad is contrasting, it is further categorized as either being more aimed at 
# promoting than attacking, more attacking than promoting, or equally attacking 
# and promoting. The result is a 5-point scale of negativity ranging from 
# 1 (positive) to 5 (attack).
data$ground_truth[data$ground_truth %in% c(1, 2)] <- "positive"
data$ground_truth[data$ground_truth == 3] <- "neutral"
data$ground_truth[data$ground_truth %in% c(4, 5)] <- "negative"

table(data$ground_truth)

write.csv(data, '../all_data_processed_full/tone.csv', row.names = F)
