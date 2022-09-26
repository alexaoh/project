library(ranger)
library(hmeasure)
library(pROC)

setwd("/home/ajo/gitRepos/project")
source("code/utilities.R")

library(ranger) # I am not getting ranger to work!!? It works in the shell, but not in Rstudio!?
# Ranger gives 0.91 AUC for both binarized and non-binarized data!
 # load("data/adult_data_categ.RData", verbose = T) 
load("data/adult_data_binarized.RData", verbose = T)
# We try to normalize first. 
cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
#adult.data.normalized <- normalize.data(data = adult.data, continuous_vars = cont) # returns list with data, mins and maxs.
#summary(adult.data.normalized)
#adult.data <- adult.data.normalized[[1]] # we are only interested in the data for now. 

# sample.size <- floor(nrow(adult.data) * 2/3)
# train.indices <- sample(1:nrow(adult.data), size = sample.size)
# train <- adult.data[train.indices, ]
# test <- adult.data[-train.indices, ]
# 
# write.csv(train, "rfor/adultTrainAlex.csv", row.names = F)
# write.csv(test, "rfor/adultTestAlex.csv", row.names = F)

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

train[,14] <- as.factor(train[,14])
rownames(train) <- 1:nrow(train)
rownames(test) <- 1:nrow(test)

adultTrain = read.table("rfor/adult_income.csv",sep=",",header=FALSE)
adultTest = read.table("rfor/adult_income_test2.csv",sep=",",header=FALSE)

train2 <- read.table("rfor/adultTrainAlex.csv",sep=",",header=TRUE)
test2 <- read.table("rfor/adultTestAlex.csv",sep=",",header=TRUE)

# adultTrain <- train2
# adultTest <- test2

# We remove "education" right away! Now the number of columns are 14 instead of 15. 
adultTrain <- adultTrain[,-4]
adultTest <- adultTest[,-4]

varNames <- c("age", "workclass",  "fnlwgt", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income")    

#colnames(adultTrain) <- colnames(adultTest) <- varNames

#Response
response <- array(0,dim(adultTrain)[1])
response[which(as.character(adultTrain[,14])==" >50K")] <- 1
adultTrain[,14] <- as.factor(response)


factor_var = c("workclass",  "marital-status", "occupation", "relationship", "race", "sex", "native-country")
num_var = c("age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week", "income")

ind <- match(factor_var,varNames)

for(i in ind)
adultTrain[,i] <- as.factor(adultTrain[,i])

for(i in ind)
adultTest[,i] <- as.factor(adultTest[,i])


runMod.rforest <- function(train, test, KJ){
  # We try with randomForest also (easier package to install!)
  # Drit i randomForest!!
  library(randomForest)
  rf <- randomForest(train[,14] ~ ., ntree = 100, data=train[,-14])
  print(summary(rf))
  print(importance(rf))
  if (KJ){
    truePred <- array(0,dim(adultTest)[1])
    truePred[which(as.character(adultTest[,14])==" >50K")] <- 1
    truePred[which(as.character(adultTest[,14])==" >50K.")] <- 1
  } else {
    truePred <- test[,14]
  }
  y_pred_rf <- predict(rf, test[,-14], type = "response")
  #y_pred_rf[y_pred_rf >= 0.5] <- 1
  #y_pred_rf[y_pred_rf < 0.5] <- 0
  print(confusionMatrix(factor(y_pred_rf), factor(test[,14])))
  print(roc(response = as.factor(truePred), predictor = as.numeric(y_pred_rf), plot = T))
  results <- HMeasure(truePred,y_pred_rf,threshold=0.15)
  print(results$metrics$AUC)
}

runMod <- function(train, test, KJ){
  # Ranger
  model <- ranger(train[,14] ~ ., data = train[,-c(14)], num.trees = 500, num.threads = 6,
                  verbose = TRUE,
                  probability = TRUE,
                  importance = "impurity",
                  mtry = sqrt(13))
  pred.rf <- predict(model, data = test[,-c(14)])
  if (KJ){
    truePred <- array(0,dim(adultTest)[1])
    truePred[which(as.character(adultTest[,14])==" >50K")] <- 1
    truePred[which(as.character(adultTest[,14])==" >50K.")] <- 1
  } else {
    truePred <- test[,14]
  }
  results <- HMeasure(truePred,pred.rf$predictions[,2],threshold=0.15)
  print(results$metrics$AUC)
  print(roc(response = truePred, predictor = as.numeric(pred.rf$predictions[,2]), plot = T))
}

#runMod(adultTrain, adultTest, T)
#runMod(train, test, F)

runMod.rforest(adultTrain, adultTest, T)
runMod.rforest(train, test, T)
