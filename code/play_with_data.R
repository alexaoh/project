# Have a manual look at the numerical variables in the Adult Census dataset.

rm(list = ls())  # make sure to remove previously loaded variables into the Session.

setwd("/home/ajo/gitRepos/project")
library(dplyr)
load("data/adult_data_binarized.RData", verbose = T) 

cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")

boxplot(adult.data %>% select(age))

fnl <- (adult.data %>% select(fnlwgt))[[1]]
summary(fnl)
boxplot(fnl)
fnl <- fnl[fnl < as.integer(1.5*quantile(fnlwgt, 0.75)[[1]])] # We remove all values larger than 1.5*third quantile. 
summary(fnl)
boxplot(fnl)

ed_num <- (adult.data %>% select(education_num))[[1]]
summary(ed_num)
boxplot(ed_num)

cap_gain <- (adult.data %>% select(capital_gain))[[1]]
summary(cap_gain)
boxplot(cap_gain)
length(cap_gain[cap_gain != 0])
length(cap_gain) # Most entries are 0! Thus, this variable is extremely difficult to model!
head(sort(cap_gain, decreasing = T), 1000)
cap_gain <- cap_gain[cap_gain < 50000]
summary(cap_gain)
boxplot(cap_gain)
cap_gain <- cap_gain[cap_gain < 20000]
summary(cap_gain)
boxplot(cap_gain)
# Still extremely disperse data!! 
cap_gain <- cap_gain[cap_gain < 1000]
summary(cap_gain)
boxplot(cap_gain)
cap_gain <- cap_gain[cap_gain < 100]
summary(cap_gain)
boxplot(cap_gain)

cap_loss <- (adult.data %>% select(capital_loss))[[1]]
summary(cap_loss)
boxplot(cap_loss)
length(cap_loss[cap_loss != 0]) # Same as for cap_gain!
length(cap_loss)

hours <- (adult.data %>% select(hours_per_week))
summary(hours)
boxplot(hours)
