# Test how dummyVars from Caret works. I have only used data.matrix earlier!
rm(list = ls())  # make sure to remove previously loaded variables into the Session.
setwd("/home/ajo/gitRepos/project")
library(dplyr)
library(caret) # For confusion matrix.

load("data/adult_data_binarized.RData", verbose = T) # Binarized factors in the data. 


data <- data.frame(adult.data)

train_text <- adult.data[,1:13] %>% select_if(is.factor)
train_numbers <- adult.data[,1:13] %>% select_if(is.integer)

encoded <- caret::dummyVars(" ~ .", data = train_text)
train_encoded <- data.frame(predict(encoded, newdata = train_text))

# This makes a not-fullrank dataset! (which I think we want!). Model.matrix makes a fullRank by default!!
adult.data.new <- cbind(train_numbers,train_encoded, adult.data["y"])
