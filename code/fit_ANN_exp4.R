# Fit the classifier for the binarized data. 
# This exact classifier is used in Experiment 3 and 5.

setwd("/home/ajo/gitRepos/project")
library(dplyr)
library(keras) # for deep learning models. 
library(pROC) # For ROC curve.
library(hmeasure) # For AUC (I am testing this for comparison to pROC).
library(caret) # For confusion matrix.

# Source some of the needed code. 
source("code/utilities.R")

# Parameter for choosing standardscaler or not. 
standardscaler = T

set.seed(42) # Set seed to begin with!

# Load the data we want first. Loading and cleaning the original data is done in separate files. 
load("data/adult_data_categ.RData", verbose = T) # Binarized factors in the data. 

cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
# List of categorical variables (used to reverse onehot encode later!)
categ <- setdiff(names(adult.data), cont)
categ <- categ[-length(categ)] # Remove the label "y"!

############ We do not care about the first CLI-argument (the model).
############ We simply (for now) only implement the ANN, for less clutter :)
adult.data.onehot <- data.frame(adult.data) # make a copy of the dataframe for one hot encoding in ANN.
tracemem(adult.data) == tracemem(adult.data.onehot) # it is a deep copy.
data.table::address(adult.data)
data.table::address(adult.data.onehot)
# The memory addresses are different. 

# Make the design matrix for the DNN.
adult.data.onehot <- make.data.for.ANN(adult.data.onehot, cont, label = T) 

# Make train and test data for our model matrix adult.data.
sample.size <- floor(nrow(adult.data.onehot) * 2/3)
train.indices <- sample(1:nrow(adult.data.onehot), size = sample.size)
train <- adult.data.onehot[train.indices, ]
test <- adult.data.onehot[-train.indices, ]

# Scale training data. 
train.normalization <- normalize.data(data = train, continuous_vars = cont, standardscaler = standardscaler) # returns list with data, mins and maxs.
train <- train.normalization[[1]]
m <- train.normalization[[2]]
M <- train.normalization[[3]]

x_train <- train[,-which(names(train) == "y")]
y_train <- train[, "y"]

# Make validation data also.
sample.size.valid <- floor(nrow(test) * 1/3)
valid.indices <- sample(1:nrow(test), size = sample.size.valid)
valid <- test[valid.indices, ]
test <- test[-valid.indices, ]

# Scaling according to the same values obtained when scaling the training data! This is very important in all applications for generalizability!!
if (standardscaler){
  # Centering and scaling according to scales and centers from training data. 
  d_test <- scale(test[,cont], center = m, scale = M)
  catego <- setdiff(names(test), cont)
  test <- cbind(d_test, test[,catego])[,colnames(test)]
  
  d_valid <- scale(valid[,cont], center = m, scale = M)
  catego <- setdiff(names(valid), cont)
  valid <- cbind(d_valid, valid[,catego])[,colnames(valid)]
} else {
  # min-max normalization according to mins and maxes from training data. 
  for (j in 1:length(cont)){
    cont_var <- cont[j]
    test[,cont_var] <- (test[,cont_var]-m[j])/(M[j]-m[j])
    valid[,cont_var] <- (valid[,cont_var]-m[j])/(M[j]-m[j])
  }
}

x_test <- test[,-which(names(test) == "y")]
y_test <- test[,"y"]

x_valid <- valid[,-which(names(valid) == "y")]
y_valid <- valid[,"y"]


##################### Step 1: Fit the ANN.
ANN <- keras_model_sequential() %>%
  layer_dense(units = 18, activation = 'relu', input_shape = c(ncol(x_train))) %>%
  layer_dense(units = 9, activation = 'relu') %>% 
  layer_dense(units = 3, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

# Compile (define loss and optimizer).
ANN %>% compile(loss = 'binary_crossentropy',
                optimizer = optimizer_adam(), # Could try other optimizers also.  #learning_rate = 0.002
                metrics = c('accuracy'))

# Train (fit).
history <- ANN %>% fit(x = data.matrix(x_train), 
                       y = y_train, 
                       epochs = 30, 
                       batch_size = 1024, 
                       validation_data = list(data.matrix(x_valid), y_valid)
)

# Plot.
plot(history)

print(summary(ANN))

# Evaluate on training data. 
ANN %>% evaluate(data.matrix(x_train), y_train)

# Evaluate on test data. 
ANN %>% evaluate(data.matrix(x_test), y_test)

y_pred <- ANN %>% predict(data.matrix(x_test))
print(confusionMatrix(factor(as.numeric(y_pred %>% `>=`(0.5))), factor(y_test)))
print(roc(response = y_test, predictor = as.numeric(y_pred), plot = T))
results <- HMeasure(y_test,as.numeric(y_pred),threshold=0.5)
print(results$metrics$AUC)
######################### Step 1 of predictor fitting is complete.

ANN %>% save_model_hdf5("classifiers/ANN_experiment4.h5") # Save the model such that we can load it (pretrained) in experiment 3 and 5. 

# We also save the train and test split (exact data), in case we want to use it later. 
# Rename the variables here, for convenience. Save as .RData in order to keep all datatypes, etc. 
train_ANN <- train
test_ANN <- test
valid_ANN <- valid
save(train_ANN, file = paste("data/exp4_data/train_data_exp4_ANN",".RData",sep=""))
save(test_ANN, file = paste("data/exp4_data/test_data_exp4_ANN",".RData",sep=""))
save(valid_ANN, file = paste("data/exp4_data/valid_data_exp4_ANN",".RData",sep=""))

# We also need to save m and M for de-normalization later. 
normalization_constants <- data.frame("m" = m, "M" = M)
write.csv(normalization_constants,"data/exp4_data/normalization_constants_exp4.csv", row.names = F)
