data_train <- read.delim("https://drive.google.com/uc?export=download&id=146Rz1T6N_B9tzdhWO-rjebunXzKg94Y1")
data_val <- read.delim("https://drive.google.com/uc?export=download&id=1LRjhaF45LUARKzYyVNSkZZU90-djj07J")

data_test <- read.delim("https://drive.google.com/uc?export=download&id=1WI9mrVJHhkQL_JpsOVQzhWaTAZSu7a5Z")


cancer_train <- read.delim("https://drive.google.com/uc?export=download&id=151ePDzV97yAMqBjiviLX86f_LsxQQ52c")
cancer_test <- read.delim("https://drive.google.com/uc?export=download&id=1wUnZ_8Ce_qMVBcVD_IgaMIthO30_P7Gn")

data <- rbind(data_train,data_val,data_test,cancer_train,cancer_test)

data <- data[c("headline","gold_label","date","source","type")]
colnames(data)[1] <- "text"
colnames(data)[2] <- "ground_truth"

# get value_counts for ground_truth column
table(data$ground_truth)

# then rename values misinfo to Misinformation and real to Trustworthy in the ground_truth column
data$ground_truth[data$ground_truth == "misinfo"] <- "Misinformation"
data$ground_truth[data$ground_truth == "real"] <- "Trustworthy"

write.csv(data, '../all_data_processed_full/misinfo.csv', row.names = F)
