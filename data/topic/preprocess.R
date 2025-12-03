library(tidyverse)
data_raw <- read.csv("data_raw/replication/01_data_and_preprocessing/logit/cbp_data/cbp_easy_with_proxies.csv")
data <- data.frame(text=data_raw$text,
                   label=data_raw$majortopic,
                   senate = data_raw$senate,
                   democrat = data_raw$democrat,
                   ideology = data_raw$dw1)

data <- data %>% mutate(
  ground_truth = case_when(
    label == 1 ~ "True",
    label == 12 ~ "False",
    label == 16 ~ "False",
    label == 19 ~ "False"
  ),
  label = case_when(
    label == 1 ~ "Macroeconomics",
    label == 12 ~ "Law and Crime",
    label == 16 ~ "Defense",
    label == 19 ~ "International Affairs"
  ),
  senate = case_when(
    senate == "True" ~ 1,
    senate == "False" ~ 0
  ),
  democrat = case_when(
    democrat == "True" ~ 1,
    democrat == "False" ~ 0
  )
)

write.csv(data, '../all_data_processed_full/topic.csv', row.names = F)
