library(dataverse)
library(tidyverse)
library(irr)
library(jsonlite)

# Prevent scientific notation for large numbers
options(scipen = 999)

calculate_krippendorff <- function(data) {
  # Handle duplicates by taking the first annotation from each worker for each sentence
  data <- data %>%
    group_by(coderid, sentenceid) %>%
    slice(1) %>%
    ungroup()
  
  # Convert data to wide format for Krippendorff's alpha
  # Each row is an annotator, each column is an item
  wide_data <- data %>%
    select(sentenceid, coderid, code) %>%
    pivot_wider(names_from = sentenceid, values_from = code, values_fill = NA)
  
  # Remove the coderid column and convert to matrix
  alpha_matrix <- as.matrix(wide_data[, -1])
  
  # Calculate Krippendorff's alpha
  alpha <- kripp.alpha(alpha_matrix, method = "nominal")
  
  return(alpha$value)
}


# Alternative function for use with pick()
create_annotation_string_from_pick <- function(picked_data) {
  # Create a JSON object mapping coderid to code
  annotation_dict <- setNames(as.list(picked_data$code), picked_data$coderid)
  return(toJSON(annotation_dict, auto_unbox = TRUE))
}

data <- get_dataframe_by_name(
  filename = "application3-benoit-sentence-estimates.tab",
  dataset  = "10.7910/DVN/DZZ0OM",
  server   = "dataverse.harvard.edu",
  original = TRUE,
  .f       = read.csv
)
# Ensure coderid is treated as character to avoid scientific notation
data$coderid <- as.character(data$coderid)
data <- data[data$scale=="Economic"|data$scale=="Social",]
data <- data[!data$manifestoid%in%c("99","screener"),]
data <- data[data$source=="Crowd",]

# Calculate total number of unique annotators and annotations per datapoint
total_annotators <- length(unique(data$coderid))
cat("Total number of unique annotators:", total_annotators, "\n")

# For manifestos_issue task, we need to calculate alpha on the scale dimension
# First, let's see how many annotations per sentence for the issue classification
issue_annotations_per_sentence <- data %>%
  group_by(sentenceid) %>%
  summarise(n_annotations = n(), .groups = 'drop') %>%
  pull(n_annotations)
avg_annotations_issue <- mean(issue_annotations_per_sentence)
cat("Average annotations per sentence (issue):", round(avg_annotations_issue, 2), "\n")

# Calculate Krippendorff's alpha on full dataset for issue classification (Economic vs Social)
# Create a dataset for issue classification
data_issue_alpha <- data %>%
  mutate(code = ifelse(scale == "Economic", 0, 1)) %>%
  select(sentenceid, coderid, code)

# Handle duplicates by taking the first annotation from each worker for each sentence
data_issue_alpha <- data_issue_alpha %>%
  group_by(coderid, sentenceid) %>%
  slice(1) %>%
  ungroup()

# # Reshape for Krippendorff calculation
# issue_wide <- data_issue_alpha %>%
#   pivot_wider(names_from = sentenceid, values_from = code, values_fill = NA)
# issue_matrix <- as.matrix(issue_wide[, -1])
# krippendorff_alpha_issue_full_dataset <- kripp.alpha(issue_matrix, method = "nominal")$value
# cat("Krippendorff's Alpha on full dataset (manifestos_issue):", round(krippendorff_alpha_issue_full_dataset, 4), "\n")

# Continue with the filtered data for ground truth
# Keep original annotations with annotator ids
data_issues <- data %>% group_by(sentenceid,scale) %>% 
  summarise(n = n(), .groups = 'drop') %>% 
  group_by(sentenceid) %>%
  summarise(
    total_annotations = sum(n),
    economic_annotations = sum(n[scale == "Economic"]),
    economic_proportion = (economic_annotations / total_annotations),
    .groups = 'drop'
  ) %>%
  filter(economic_proportion>0.8|economic_proportion<0.2) %>%
  mutate(ground_truth = ifelse(economic_proportion>0.8,"Economy","Social"))

# Keep original annotations for filtered sentences
filtered_sentence_ids_issue <- data_issues$sentenceid
data_issue_annotations <- data %>%
  filter(sentenceid %in% filtered_sentence_ids_issue) %>%
  mutate(code = ifelse(scale == "Economic", 0, 1)) %>%
  # Remove duplicates: keep first annotation from each coderid for each sentenceid
  group_by(coderid, sentenceid) %>%
  slice(1) %>%
  ungroup() %>%
  group_by(sentenceid) %>%
  summarise(
    original_annotations = create_annotation_string_from_pick(pick(everything())),
    .groups = 'drop'
  )

# Merge annotations with ground truth
data_issues <- merge(data_issues, data_issue_annotations, by = "sentenceid")

data_w_text <- read.csv("https://raw.githubusercontent.com/kbenoit/CSTA-APSR/refs/heads/master/01%20Create%20Intitial%20Sentence%20Dataset/master_sentence_list.csv")
colnames(data_w_text)[18] <- "text"
data_w_text <- data_w_text[c("sentenceid","text","manifestoid")]
data_w_text <- merge(data_w_text,data_issues[c("ground_truth","sentenceid","original_annotations")],all.x=T)
data_w_text <- data_w_text[!is.na(data_w_text$ground_truth),]
data_w_text <- data_w_text %>%
  separate(manifestoid, into = c("party", "year"), sep = " ")

# Now calculate Krippendorff's alpha on used dataset
data_issue_used <- data %>%
  filter(sentenceid %in% filtered_sentence_ids_issue) %>%
  mutate(code = ifelse(scale == "Economic", 0, 1)) %>%
  select(sentenceid, coderid, code)
krippendorff_alpha_issue <- calculate_krippendorff(data_issue_used)
cat("Krippendorff's Alpha on used dataset (manifestos_issue):", round(krippendorff_alpha_issue, 4), "\n")

write.csv(data_w_text, '../all_data_processed_full/manifestos_issue_NEW.csv', row.names = F)

#now get the Economic data
# First calculate Krippendorff's alpha on ALL economic sentences (before 80% filtering)
data_econ_all <- data[data$scale=="Economic",]
data_econ_all$code <- ifelse(data_econ_all$code==-2,-1,data_econ_all$code)
data_econ_all$code <- ifelse(data_econ_all$code==2,1,data_econ_all$code)

# Calculate annotations per sentence for economic ideology
econ_annotations_per_sentence <- data_econ_all %>%
  group_by(sentenceid) %>%
  summarise(n_annotations = n(), .groups = 'drop') %>%
  pull(n_annotations)
avg_annotations_econ <- mean(econ_annotations_per_sentence)
cat("Average annotations per sentence (econ ideology):", round(avg_annotations_econ, 2), "\n")

# # Calculate Krippendorff's alpha on full dataset for economic ideology
# krippendorff_alpha_econ_ideology_full_dataset <- calculate_krippendorff(data_econ_all)
# cat("Krippendorff's Alpha on full dataset (manifestos_econ_ideology):", round(krippendorff_alpha_econ_ideology_full_dataset, 4), "\n")

# Now continue with filtered data for ground truth
data_econ <- data[data$sentenceid%in%data_issues[data_issues$ground_truth=="Economy",]$sentenceid,]
data_econ$code <- ifelse(data_econ$code==-2,-1,data_econ$code)
data_econ$code <- ifelse(data_econ$code==2,1,data_econ$code)

# Keep original annotations with annotator ids
data_econ_summary <- data_econ %>% group_by(sentenceid,code) %>% 
  summarise(n = n(), .groups = 'drop') %>% 
  group_by(sentenceid) %>%
  summarise(
    total_annotations = sum(n),
    left_annotations = sum(n[code == -1]),
    right_annotations = sum(n[code == 1]),
    left_proportion = (left_annotations / total_annotations),
    right_proportion = (right_annotations / total_annotations),
    .groups = 'drop'
  ) %>%
  filter(left_proportion>0.8|right_proportion>0.8) %>%
  mutate(ground_truth = ifelse(left_proportion>0.8,"Left","Right"))

# Keep original annotations for filtered sentences
filtered_sentence_ids_econ <- data_econ_summary$sentenceid
data_econ_annotations <- data_econ %>%
  filter(sentenceid %in% filtered_sentence_ids_econ) %>%
  # Remove duplicates: keep first annotation from each coderid for each sentenceid
  group_by(coderid, sentenceid) %>%
  slice(1) %>%
  ungroup() %>%
  group_by(sentenceid) %>%
  summarise(
    original_annotations = create_annotation_string_from_pick(pick(everything())),
    .groups = 'drop'
  )

# Merge annotations with ground truth
data_econ_summary <- merge(data_econ_summary, data_econ_annotations, by = "sentenceid")

data_w_text <- read.csv("https://raw.githubusercontent.com/kbenoit/CSTA-APSR/refs/heads/master/01%20Create%20Intitial%20Sentence%20Dataset/master_sentence_list.csv")
colnames(data_w_text)[18] <- "text"
data_w_text <- data_w_text[c("sentenceid","text","manifestoid")]
data_w_text <- merge(data_w_text,data_econ_summary[c("ground_truth","sentenceid","original_annotations")],all.x=T)
data_w_text <- data_w_text[!is.na(data_w_text$ground_truth),]
data_w_text <- data_w_text %>%
  separate(manifestoid, into = c("party", "year"), sep = " ")

# Now calculate Krippendorff's alpha on used dataset
data_econ_used <- data_econ %>%
  filter(sentenceid %in% filtered_sentence_ids_econ) %>%
  select(sentenceid, coderid, code)
krippendorff_alpha_econ_ideology <- calculate_krippendorff(data_econ_used)
cat("Krippendorff's Alpha on used dataset (manifestos_econ_ideology):", round(krippendorff_alpha_econ_ideology, 4), "\n")

write.csv(data_w_text, '../all_data_processed_full/manifestos_econ_ideology_NEW.csv', row.names = F)

#social
# First calculate Krippendorff's alpha on ALL social sentences (before 80% filtering)
data_social_all <- data[data$scale=="Social",]
data_social_all$code <- ifelse(data_social_all$code==-2,-1,data_social_all$code)
data_social_all$code <- ifelse(data_social_all$code==2,1,data_social_all$code)

# Calculate annotations per sentence for social ideology
social_annotations_per_sentence <- data_social_all %>%
  group_by(sentenceid) %>%
  summarise(n_annotations = n(), .groups = 'drop') %>%
  pull(n_annotations)
avg_annotations_social <- mean(social_annotations_per_sentence)
cat("Average annotations per sentence (social ideology):", round(avg_annotations_social, 2), "\n")

# # Calculate Krippendorff's alpha on full dataset for social ideology
# krippendorff_alpha_social_ideology_full_dataset <- calculate_krippendorff(data_social_all)
# cat("Krippendorff's Alpha on full dataset (manifestos_social_ideology):", round(krippendorff_alpha_social_ideology_full_dataset, 4), "\n")

# Now continue with filtered data for ground truth
data_social <- data[data$sentenceid%in%data_issues[data_issues$ground_truth=="Social",]$sentenceid,]
data_social$code <- ifelse(data_social$code==-2,-1,data_social$code)
data_social$code <- ifelse(data_social$code==2,1,data_social$code)

# Keep original annotations with annotator ids
data_social_summary <- data_social %>% group_by(sentenceid,code) %>% 
  summarise(n = n(), .groups = 'drop') %>% 
  group_by(sentenceid) %>%
  summarise(
    total_annotations = sum(n),
    left_annotations = sum(n[code == -1]),
    right_annotations = sum(n[code == 1]),
    left_proportion = (left_annotations / total_annotations),
    right_proportion = (right_annotations / total_annotations),
    .groups = 'drop'
  ) %>%
  filter(left_proportion>0.8|right_proportion>0.8) %>%
  mutate(ground_truth = ifelse(left_proportion>0.8,"Liberal","Conservative"))

# Keep original annotations for filtered sentences
filtered_sentence_ids_social <- data_social_summary$sentenceid
data_social_annotations <- data_social %>%
  filter(sentenceid %in% filtered_sentence_ids_social) %>%
  # Remove duplicates: keep first annotation from each coderid for each sentenceid
  group_by(coderid, sentenceid) %>%
  slice(1) %>%
  ungroup() %>%
  group_by(sentenceid) %>%
  summarise(
    original_annotations = create_annotation_string_from_pick(pick(everything())),
    .groups = 'drop'
  )

# Merge annotations with ground truth
data_social_summary <- merge(data_social_summary, data_social_annotations, by = "sentenceid")

data_w_text <- read.csv("https://raw.githubusercontent.com/kbenoit/CSTA-APSR/refs/heads/master/01%20Create%20Intitial%20Sentence%20Dataset/master_sentence_list.csv")
colnames(data_w_text)[18] <- "text"
data_w_text <- data_w_text[c("sentenceid","text","manifestoid")]
data_w_text <- merge(data_w_text,data_social_summary[c("ground_truth","sentenceid","original_annotations")],all.x=T)
data_w_text <- data_w_text[!is.na(data_w_text$ground_truth),]
data_w_text <- data_w_text %>%
  separate(manifestoid, into = c("party", "year"), sep = " ")

# Now calculate Krippendorff's alpha on used dataset
data_social_used <- data_social %>%
  filter(sentenceid %in% filtered_sentence_ids_social) %>%
  select(sentenceid, coderid, code)
krippendorff_alpha_social_ideology <- calculate_krippendorff(data_social_used)
cat("Krippendorff's Alpha on used dataset (manifestos_social_ideology):", round(krippendorff_alpha_social_ideology, 4), "\n")

write.csv(data_w_text, '../all_data_processed_full/manifestos_social_ideology_NEW.csv', row.names = F)