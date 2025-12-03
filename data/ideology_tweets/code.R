countries <- c("AUSTRALIA","CANADA","DENMARK","GERMANY","NZ","POLAND","SWEDEN","TURKEY","UK","US")

datasets <- list()
i <- 1
for (c in countries){
  country_url <- paste0("http://raw.githubusercontent.com/cssmodels/llm/refs/heads/main/Data/countries/",c,"_sample_tweets.csv")
  
  temp <- read.csv(country_url)
  temp$country <- c
  colnames(temp)[1] <- "id"
  datasets[[i]] <- temp
  i <- i + 1
}
dataset <- do.call(rbind, datasets)
dataset[dataset$party=="AK Parti",]$party <- "AKP"
dataset[dataset$party=="Mod",]$party <- "Moderates"
dataset[dataset$party=="SocDem",]$party <- "Social Democrats"

dataset <- dataset[c("text","party","country","region","from_user_realname")]
# colnames(dataset)[2] <- "party"




party_lr <- c(
  "Australian Labor Party" = "left", #australia
  "Liberal Party of Australia" = "right", #australia
  "Conservative" = "right", #canada and uk
  "Liberal" = "left",#canada
  "The Liberal Party" = "right", #dk
  "The Social Democratic Party" = "left", #dk
  "CDU" = "right", #germany
  "SPD" = "left",#germany
  "Labour Party" = "left", #nz
  "National Party" = "right", #nz
  "Civic Coalition" = "left", #poland, relatively left?
  "Law and Justice" = "right", #poland
  "Moderates" = "right", #sweden
  "Social Democrats" = "left", #sweden
  "AKP" = "right", #turkey
  "CHP" = "left",#turkey
  "Labour" = "left", #uk
  "Democrat" = "left", #us
  "Republican" = "right" #us
)

country_code = c(
  "AUSTRALIA"="AU",
  "CANADA"="CA",
  "DENMARK"="DK",
  "GERMANY"="DE",
  "NZ"="NZ",
  "POLAND"="PL",
  "SWEDEN"="SE",
  "TURKEY"="TR",
  "UK"="GB",
  "US"="US"
)
dataset$ground_truth <- party_lr[dataset$party]
dataset$country_code <- country_code[dataset$country]
dataset$firstname <- dataset$from_user_realname
dataset$firstname <- gsub("Senator|Sen\\.|MP|Re-elect|Dr\\.|U.S.|\\(|Hon\\.saini4kitcen|poilievre|\\?|Hon\\.|Kr\\.|Y\\.||Dt\\.|Av\\.||CHP|🇹🇷|Vote |Hon ","",dataset$firstname)
dataset$firstname <- trimws(dataset$firstname)
dataset$firstname <- word(dataset$firstname, 1)

#genderize name
#library(devtools)
#install_github("https://github.com/kalimu/genderizeR")
library(genderizeR)
apikey = #<genderize apikey here>
dataset$gender <- NA
dataset$genderprob <- NA

for (i in 1:nrow(dataset)){
  temp <- findGivenNames(dataset[i,]$firstname,
                 country=dataset[i,]$country_code,
                 apikey=apikey,textPrepare=F,progress=F)
  dataset$gender[i] <- temp$count[1]
  dataset$genderprob[i] <- temp$country_id[1]
}

dataset$ground_truth <- party_lr[dataset$party]


# write.csv(dataset,"processed_data.csv")
write.csv(dataset, '../all_data_processed_full/ideology_tweets.csv', row.names = F)

general_prompt <- "You will be given a set of Twitter posts from different [country] politicians, sent during the two months preceding the [year] [country] election. Your task is to use your knowledge of [country] politics to make an educated guess on whether the poster is a [party1] or [party2]. Respond either '[party1]' or '[party2]'. If the message does not have enough information for an educated guess, just make your best guess."

prompts <- data.frame(country = NA,
                      prompt = NA)
i <- 0
for (c in countries){
  i <- i + 1
  if (c == "AUSTRALIA"){
    party1 <- "Australian Labor Party"
    party2 <- "Liberal Party of Australia"
    country = "Australian"
    year <- "2019"
    
    prompt_temp <- gsub("\\[country\\]", country, general_prompt)
    prompt_temp <- gsub("\\[year\\]", year, prompt_temp)
    prompt_temp <- gsub("\\[party1\\]", party1, prompt_temp)
    prompt_temp <- gsub("\\[party2\\]", party2, prompt_temp)
    
    prompts[i,"country"] <- c
    prompts[i,"prompt"] <- prompt_temp
    
    
    }
  
  if (c == "CANADA"){
    party1 <- "Liberal"
    party2 <- "Conservative"
    country <- "Canadian"
    year <- "2021"
    
    prompt_temp <- gsub("\\[country\\]", country, general_prompt)
    prompt_temp <- gsub("\\[year\\]", year, prompt_temp)
    prompt_temp <- gsub("\\[party1\\]", party1, prompt_temp)
    prompt_temp <- gsub("\\[party2\\]", party2, prompt_temp)
    
    prompts[i,"country"] <- c
    prompts[i,"prompt"] <- prompt_temp
    
  }
  if (c == "DENMARK"){
    party1 <- "The Social Democratic Party"
    party2 <- "The Liberal Party"
    country <- "Danish"
    year <- "2019"
    prompt_temp <- gsub("\\[country\\]", country, general_prompt)
    prompt_temp <- gsub("\\[year\\]", year, prompt_temp)
    prompt_temp <- gsub("\\[party1\\]", party1, prompt_temp)
    prompt_temp <- gsub("\\[party2\\]", party2, prompt_temp)
    
    prompts[i,"country"] <- c
    prompts[i,"prompt"] <- prompt_temp
    
  }
  if (c == "GERMANY"){
    party1 <- "CDU"
    party2 <- "SPD"
    country <- "German"
    year <- "2021"
    
    prompt_temp <- gsub("\\[country\\]", country, general_prompt)
    prompt_temp <- gsub("\\[year\\]", year, prompt_temp)
    prompt_temp <- gsub("\\[party1\\]", party1, prompt_temp)
    prompt_temp <- gsub("\\[party2\\]", party2, prompt_temp)
    
    prompts[i,"country"] <- c
    prompts[i,"prompt"] <- prompt_temp
    
  }
  if (c == "NZ"){
    party1 <- "Labour Party"
    party2 <- "National Party"
    country <- "New Zealand"
    year <- "2020"
    
    prompt_temp <- gsub("\\[country\\]", country, general_prompt)
    prompt_temp <- gsub("\\[year\\]", year, prompt_temp)
    prompt_temp <- gsub("\\[party1\\]", party1, prompt_temp)
    prompt_temp <- gsub("\\[party2\\]", party2, prompt_temp)
    
    prompts[i,"country"] <- c
    prompts[i,"prompt"] <- prompt_temp
    
  }
  if (c == "POLAND"){
    party1 <- "Civic Coalition"
    party2 <- "Law and Justice"
    country <- "Polish"
    year <- "2020"
    
    prompt_temp <- gsub("\\[country\\]", country, general_prompt)
    prompt_temp <- gsub("\\[year\\]", year, prompt_temp)
    prompt_temp <- gsub("\\[party1\\]", party1, prompt_temp)
    prompt_temp <- gsub("\\[party2\\]", party2, prompt_temp)
    
    prompts[i,"country"] <- c
    prompts[i,"prompt"] <- prompt_temp
    
  }
  if (c == "SWEDEN"){
    party1 <- "Social Democrats"
    party2 <- "Moderates"
    country <- "Swedish"
    year <- "2018"
    
    prompt_temp <- gsub("\\[country\\]", country, general_prompt)
    prompt_temp <- gsub("\\[year\\]", year, prompt_temp)
    prompt_temp <- gsub("\\[party1\\]", party1, prompt_temp)
    prompt_temp <- gsub("\\[party2\\]", party2, prompt_temp)
    
    prompts[i,"country"] <- c
    prompts[i,"prompt"] <- prompt_temp
    
  }
  if (c == "TURKEY"){
    party1 <- "AKP"
    party2 <- "CHP"
    country <- "Turkish"
    year <- "2018"
    
    prompt_temp <- gsub("\\[country\\]", country, general_prompt)
    prompt_temp <- gsub("\\[year\\]", year, prompt_temp)
    prompt_temp <- gsub("\\[party1\\]", party1, prompt_temp)
    prompt_temp <- gsub("\\[party2\\]", party2, prompt_temp)
    
    prompts[i,"country"] <- c
    prompts[i,"prompt"] <- prompt_temp
    
  }
  if (c == "UK"){
    party1 <- "Conservative"
    party2 <- "Labour"
    country <- "UK"
    year <- "2019"
    
    prompt_temp <- gsub("\\[country\\]", country, general_prompt)
    prompt_temp <- gsub("\\[year\\]", year, prompt_temp)
    prompt_temp <- gsub("\\[party1\\]", party1, prompt_temp)
    prompt_temp <- gsub("\\[party2\\]", party2, prompt_temp)
    
    prompts[i,"country"] <- c
    prompts[i,"prompt"] <- prompt_temp
    
  }
  if (c == "US"){
    party1 <- "Democrat"
    party2 <- "Republican"
    country <- "US"
    year <- "2020"
    
    prompt_temp <- gsub("\\[country\\]", country, general_prompt)
    prompt_temp <- gsub("\\[year\\]", year, prompt_temp)
    prompt_temp <- gsub("\\[party1\\]", party1, prompt_temp)
    prompt_temp <- gsub("\\[party2\\]", party2, prompt_temp)
    
    prompts[i,"country"] <- c
    prompts[i,"prompt"] <- prompt_temp
    
  }
}

#lr is ground_truth, keep user_name, add gender via genderize.io
write.csv(prompts,"country_prompts.csv",row.names=F)


### estimate annotator agreement ###

data <- read.csv("https://raw.githubusercontent.com/orionw/RedditHumorDetection/refs/heads/master/full_datasets/reddit_jokes/reddit_full_data.csv")