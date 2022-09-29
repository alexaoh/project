# Testing ANN with binary one-hot encoded data. 

rm(list = ls())  # make sure to remove previously loaded variables into the Session.

setwd("/home/ajo/gitRepos/project")
library(dplyr)
library(keras) # for deep learning models. 

# Source some of the needed code. 
source("code/utilities.R")

set.seed(42)

load("data/adult_data_binarized.RData", verbose = T) # Binarized factors in the data. 


cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")

adult.data.normalized <- normalize.data(data = adult.data, continuous_vars = cont) # returns list with data, mins and maxs.
summary(adult.data.normalized)
adult.data <- adult.data.normalized[[1]] # we are only interested in the data for now. 

# This is for one-hot encoding of the data. 
adult.data2 <- as.data.frame(model.matrix(~.,data = adult.data)) 


# Make train and test data.
train_and_test_data <- make.train.and.test(data = adult.data2) # The function returns two matrices (x) and two vectors (y). 
# In addition, it returns two dataframes that are the original dataframe split into train and test (containing y's and x's).
summary(train_and_test_data) # Returned list. 
x_train <- train_and_test_data[[1]]
y_train <- train_and_test_data[[2]]
x_test <- train_and_test_data[[3]]
y_test <- train_and_test_data[[4]]

# These two are used when I want to make dataframes later, in order to easier keep all correct datatypes in the columns. 
train <- train_and_test_data[[5]]
test <- train_and_test_data[[6]]

ANN <- keras_model_sequential() %>%
  layer_dense(units = 18, activation = 'relu', input_shape = c(ncol(x_train))) %>%
  layer_dense(units = 9, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

# compile (define loss and optimizer)
ANN %>% compile(loss = 'binary_crossentropy',
                optimizer = optimizer_rmsprop(),
                metrics = c('accuracy'))

# train (fit)
history <- ANN %>% fit(data.matrix(x_train), y_train, epochs = 40, 
                       batch_size = 1024, validation_split = 0.2)
# plot
plot(history)

print(summary(ANN))

# evaluate on training data. 
ANN %>% evaluate(data.matrix(x_train), y_train)

# evaluate on test data. 
ANN %>% evaluate(data.matrix(x_test), y_test)

y_pred <- ANN %>% predict(data.matrix(x_test)) %>% `>`(0.5) %>% k_cast("int32")
y_pred <- as.array(y_pred)
tab <- table("Predictions" = y_pred, "Labels" = y_test)
print(confusionMatrix(factor(y_pred), factor(y_test)))
print(roc(response = y_test, predictor = as.numeric(y_pred), plot = T))




###################################### Let us try something similar for the categorical data (not-binarized)

# Testing ANN with binary one-hot encoded data. 

rm(list = ls())  # make sure to remove previously loaded variables into the Session.

setwd("/home/ajo/gitRepos/project")
library(dplyr)
library(keras) # for deep learning models. 

# Source some of the needed code. 
source("code/utilities.R")

set.seed(42)

load("data/adult_data_categ.RData", verbose = T) # Original categorical data.


cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")

adult.data.normalized <- normalize.data(data = adult.data, continuous_vars = cont) # returns list with data, mins and maxs.
summary(adult.data.normalized)
adult.data <- adult.data.normalized[[1]] # we are only interested in the data for now. 

# This is for one-hot encoding of the data. 
adult.data2 <- as.data.frame(model.matrix(~.,data = adult.data)) 


# Make train and test data.
train_and_test_data <- make.train.and.test(data = adult.data2) # The function returns two matrices (x) and two vectors (y). 
# In addition, it returns two dataframes that are the original dataframe split into train and test (containing y's and x's).
summary(train_and_test_data) # Returned list. 
x_train <- train_and_test_data[[1]]
y_train <- train_and_test_data[[2]]
x_test <- train_and_test_data[[3]]
y_test <- train_and_test_data[[4]]

# These two are used when I want to make dataframes later, in order to easier keep all correct datatypes in the columns. 
train <- train_and_test_data[[5]]
test <- train_and_test_data[[6]]

ANN <- keras_model_sequential() %>%
  layer_dense(units = 18, activation = 'relu', input_shape = c(ncol(x_train))) %>%
  layer_dense(units = 9, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

# compile (define loss and optimizer)
ANN %>% compile(loss = 'binary_crossentropy',
                optimizer = optimizer_rmsprop(),
                metrics = c('accuracy'))

# train (fit)
history <- ANN %>% fit(data.matrix(x_train), y_train, epochs = 40, 
                       batch_size = 1024, validation_split = 0.2)
# plot
plot(history)

print(summary(ANN))

# evaluate on training data. 
ANN %>% evaluate(data.matrix(x_train), y_train)

# evaluate on test data. 
ANN %>% evaluate(data.matrix(x_test), y_test)

y_pred <- ANN %>% predict(data.matrix(x_test)) %>% `>`(0.5) %>% k_cast("int32")
y_pred <- as.array(y_pred)
tab <- table("Predictions" = y_pred, "Labels" = y_test)
print(confusionMatrix(factor(y_pred), factor(y_test)))
print(roc(response = y_test, predictor = as.numeric(y_pred), plot = T))

