# Try to fit a random forest classifier, mostly in order to see if I actually am able to build a classifier that is good for our data.

setwd("/home/ajo/gitRepos/project")
source("code/utilities.R")

library(ranger) # I am not getting ranger to work!!? It works in the shell, but not in Rstudio!?
load("data/adult_data_binarized.RData", verbose = T) 

# We try to normalize first. 
cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
adult.data.normalized <- normalize.data(data = adult.data, continuous_vars = cont) # returns list with data, mins and maxs.
summary(adult.data.normalized)
adult.data <- adult.data.normalized[[1]] # we are only interested in the data for now. 

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

rfor <- ranger("y~.", data = train)
preds <- predict(rfor, data = test)$predictions
preds[preds >= 0.5] <- 1
preds[preds < 0.5] <- 0
tab <- table("Predictions" = preds, "Labels" = as.numeric(test["y"][[1]]))
library(caret)
print(confusionMatrix(factor(preds), factor(as.numeric(test["y"][[1]]))))
library(pROC)
print(roc(response = as.numeric(test["y"][[1]]), predictor = as.numeric(preds), plot = T))
# This is not any better at predicting!!

# Try with randomForest package. Having trouble installing this as well!
