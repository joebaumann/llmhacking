data <- read.csv("https://raw.githubusercontent.com/orionw/RedditHumorDetection/refs/heads/master/full_datasets/reddit_jokes/reddit_full_data.csv")

data$ground_truth <- ifelse(data$score>200, "true", "false")
barplot(table(data$ground_truth))
data$text <- data$fulltext
data$date <- as.POSIXct(data$created_utc, origin = "1970-01-01", tz = "UTC")

# get average number of votes per post
# Formula: ups = upvote_ratio * total_votes, so total_votes = ups / upvote_ratio
data$tot_nr_of_votes <- round(data$ups / data$upvote_ratio)

# The number of annotators per datapoint is the total number of votes (each vote is an annotation)
nr_of_ground_truth_annotators_per_datapoint <- mean(data$tot_nr_of_votes, na.rm = TRUE)
print(paste('nr_of_ground_truth_annotators_per_datapoint:', nr_of_ground_truth_annotators_per_datapoint))

data <- data[c("text","ground_truth","date")]
write.csv(data, '../all_data_processed_full/humor.csv', row.names = F)
