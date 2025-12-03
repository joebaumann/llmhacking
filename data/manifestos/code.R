library(manifestoR)
library(zoo)
library(tidyverse)

fill_party_years <- function(df) {
  
  party_ranges <- df %>%
    group_by(countryname, party) %>%
    summarise(
      min_year = min(year),
      max_year = max(year),
      .groups = 'drop'
    )
  
  complete_years <- party_ranges %>%
    rowwise() %>%
    do(data.frame(
      countryname = .$countryname,
      party = .$party,
      year = seq(.$min_year, .$max_year, by = 1)
    )) %>%
    ungroup()
  
  result <- complete_years %>%
    left_join(df, by = c("countryname", "party", "year")) %>%
    arrange(countryname, party, year) %>%
    group_by(countryname, party) %>%
    fill(partyname, partyabbrev, parfam, .direction = "down") %>%
    ungroup()
  
  return(result)
}
codebook <- data.frame(
  cmp_code = c(
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    201, 202, 203, 204, 301, 302, 303, 304, 305,
    401, 402, 403, 404, 405, 406, 407, 408, 409, 410,
    411, 412, 413, 414, 415, 416,
    501, 502, 503, 504, 505, 506, 507,
    601, 602, 603, 604, 605, 606, 607, 608,
    701, 702, 703, 704, 705, 706
  ),
  ground_truth = c(
    # 101/102 collapsed
    "Foreign Special Relationships",
    "Foreign Special Relationships",
    "Anti-Imperialism",
    # 104/105 collapsed
    "Military/Defence",
    "Military/Defence",
    "Peace",
    # 107/109 collapsed
    "Internationalism",
    "European Community/Union",
    "Internationalism",
    "European Community/Union",
    "Freedom and Human Rights",
    "Democracy",
    # 203/204 collapsed
    "Constitutionalism",
    "Constitutionalism",
    "Federalism",
    "Centralisation",
    "Governmental and Administrative Efficiency",
    "Political Corruption",
    "Political Authority",
    "Free Market Economy",
    "Incentives",
    "Market Regulation",
    "Economic Planning",
    "Corporatism/Mixed Economy",
    # 406/407 collapsed
    "Protectionism",
    "Protectionism",
    "Economic Goals",
    "Keynesian Demand Management",
    "Economic Growth",
    "Technology and Infrastructure",
    "Controlled Economy",
    "Nationalisation",
    "Economic Orthodoxy",
    "Marxist Analysis",
    "Anti-Growth Economy",
    # 501
    "Environmental Protection",
    "Culture",
    "Equality",
    "Welfare State Expansion",
    "Welfare State Limitation",
    "Education Expansion",
    "Education Limitation",
    # 601/602 collapsed
    "National Way of Life",
    "National Way of Life",
    # 603/604 collapsed
    "Traditional Morality",
    "Traditional Morality",
    "Law and Order",
    "Civic Mindedness",
    # 607/608 collapsed
    "Multiculturalism",
    "Multiculturalism",
    # 701/702 collapsed
    "Labour Groups",
    "Labour Groups",
    "Agriculture and Farmers",
    "Middle Class and Professional Groups",
    "Underprivileged Minority Groups",
    "Non-economic Demographic Groups"
  ),
  stringsAsFactors = FALSE
)

apikey <- #<apikeyhere>

data <- mp_corpus(edate > as.Date("2000-01-01"),
                  apikey=apikey,translation="en",as_tibble = T)
data <- merge(data,codebook,by=c("cmp_code"),all.x=T)
data <- data[!is.na(data$ground_truth),c("text","ground_truth","manifesto_id","party","date","cmp_code")]
data$year <- substr(data$date,1,4)


party_data <- read.csv("data_raw/MPDataset_MPDS2024a.csv")
party_data$year <- substr(party_data$date,1,4)
party_data <- party_data[,c("countryname","partyname","partyabbrev","parfam","party","year","testresult")]
party_data$year <- as.numeric(party_data$year)
data2 <- fill_party_years(party_data)
data2$party_year <- paste0(data2$party,data2$year)
data2 <- data2[!duplicated(data2$party_year),]

data3 <- merge(data,data2,by=c("year","party"),all.x=T)
data3$party_year <- NULL

# calculate mean krippendorff alpha: mean of available testresult values
calculate_krippendorff <- function(data) {
  valid_testresults <- data$testresult[!is.na(data$testresult)]
  if(length(valid_testresults) == 0) {
    return(NA)
  }
  return(mean(valid_testresults))
}

krippendorff_alpha <- calculate_krippendorff(data3)
cat("Mean Krippendorff's Alpha:", krippendorff_alpha, "\n")

write.csv(data3, '../all_data_processed_full/manifestos_issues_detailed.csv', row.names = F)
