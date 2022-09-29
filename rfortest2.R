rm(list = ls())  # make sure to remove previously loaded variables into the Session.
library(ranger)
library(hmeasure)
library(pROC)

setwd("/home/ajo/gitRepos/project")

source("code/utilities.R")

########################################### Build ML models for classification: which individuals obtain an income more than 50k yearly?
set.seed(42) # Set seed to begin with!

# Load the data we want first. Loading and cleaning the original data is done in separate files. 
load("data/adult_data_categ.RData", verbose = T) # Categorical factors as they come originally. 

# List of continuous variables.
cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")


# Do stuff with response for adult.data here first to check (before doing it in the adult_data_categ file)-
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

# These two are used when I want to make dataframes later, in order to easier keep all correct datatypes in the columns. 
train <- train_and_test_data[[5]]
test <- train_and_test_data[[6]]

adultTrain <- train
adultTest <- test

model <- ranger(adultTrain[,14] ~ ., data = adultTrain[,-c(14)], num.trees = 500, num.threads = 6,
                verbose = TRUE,
                probability = TRUE,
                importance = "impurity",
                mtry = sqrt(13))
pred.rf <- predict(model, data = adultTest[,-c(14)])
results <- HMeasure(y_test,pred.rf$predictions[,2],threshold=0.15)
print(results$metrics$AUC)
print(roc(response = y_test, predictor = as.numeric(pred.rf$predictions[,2]), plot = T))
