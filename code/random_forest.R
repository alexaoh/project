# Fit a simple random forest classifier with Ranger. 

setwd("/home/ajo/gitRepos/project")
source("code/utilities.R")

library(ranger) 
library(hmeasure)
library(pROC)
load("data/adult_data_categ.RData", verbose = T) 

cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")

response <- array(0,dim(adult.data)[1])
response[which(as.character(adult.data[,14])==" >50K")] <- 1
adult.data[,14] <- as.numeric(response)

# Make train and test data.
train_and_test_data <- make.train.and.test(data = adult.data) # The function returns two matrices (x) and two vectors (y). 
# In addition, it returns two dataframes that are the original dataframe split into train and test (containing y's and x's).
summary(train_and_test_data) # Returned list. 
x_train <- train_and_test_data[[1]]
y_train <- train_and_test_data[[2]]
x_test <- train_and_test_data[[3]]
y_test <- train_and_test_data[[4]]

train <- train_and_test_data[[5]]
test <- train_and_test_data[[6]]

rfor <- ranger(as.factor(train[,14]) ~ ., data = train[,-14], probability = TRUE)
               
preds <- predict(rfor, data = test[,-14])$predictions[,2]
print(roc(response = as.numeric(test["y"][[1]]), predictor = as.numeric(preds), plot = T))
results <- HMeasure(as.numeric(test["y"][[1]]),preds,threshold=0.15)
print(results$metrics$AUC)
