# Writing the outline for algorithm 1 in MCCE by Aas et al.

# Next need to add some data to test this out. 
# Adult data set: 
# https://archive.ics.uci.edu/ml/datasets/Adult

# Give me some credit data: 
# https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset

setwd("/home/ajo/gitRepos/project")
set.seed(42) # Set seed to begin with!

library(tree) # For regression trees. 
library(rpart) # Try this for building CART trees instead!
library(rpart.plot) # For plotting rpart trees in a more fancy way.
library(dplyr)
library(keras)
library(pROC) # For ROC curve.
library(caret) # For confusion matrix.

###################################### Loading and cleaning the Adult data.
data1 <- read.csv("adult.data", header = F) 
data2 <- read.csv("adult.test", header = F) 
colnames(data1) <- colnames(data2) <- c("age","workclass","fnlwgt","education","education_num",
                     "marital_status","occupation","relationship","race","sex",
                     "capital_gain","capital_loss","hours_per_week","native_country", "y")
dim(data1)[1] + dim(data2)[1] # Need to concat the test data and the other data given on the website to get all the data used in article. 
adult.data <- rbind(data1, data2) # This is the full dataset.
any(is.na(adult.data))

# Make corrections to the variables (data types, binarization, etc).
adult.data$y[adult.data$y == " <=50K."] <- "<=50K"
adult.data$y[adult.data$y == " <=50K"] <- "<=50K"
adult.data$y[adult.data$y == " >50K."] <- ">50K"
adult.data$y[adult.data$y == " >50K"] <- ">50K"
adult.data$y[adult.data$y == ">50K"] <- 1
adult.data$y[adult.data$y == "<=50K"] <- 0
adult.data$y <- as.numeric(adult.data$y)
adult.data$sex <- as.factor(adult.data$sex)

# binarize <- function(vector){
#   most_frequent <- names(sort(table(vector), decreasing = T))[1]
#   if (vector == most_frequent) vector = most.frequent else vector = "Other"
#   vector
# }

# Find most frequent value.
most_frequent <- function(vec){
  names(sort(table(vec), decreasing = T))[1]
}


binarize <- function(vec){
  vec[which(vec != most_frequent(vec))] <- "Other"
  vec <- as.factor(vec)
  vec
}

# Binarize all necessary columns
adult.data$workclass <- binarize(adult.data$workclass)
adult.data$education <- binarize(adult.data$education)
adult.data$marital_status <- binarize(adult.data$marital_status)
adult.data$occupation <- binarize(adult.data$occupation)
adult.data$relationship <- binarize(adult.data$relationship)
adult.data$race <- binarize(adult.data$race)
adult.data$native_country <- binarize(adult.data$native_country)

# write.csv(adult.data, file = "adult_data_binarized.csv", row.names = F)
save(adult.data, file = "adult_data_binarized.RData") # Save the dataset including all factors etc.

summary(adult.data)

########################################### Build ML models for classification: which individuals obtain an income more than 50k yearly?

# Normalize the continuous variables.
cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")

normalize <- function(x) {
  return((x- min(x))/(max(x)-min(x)))
}

for (c in cont){
  adult.data[,c] <- normalize(adult.data[,c])
}

summary(adult.data) # Now the data has been normalized. 

# Make train and test data.
train.ratio <- 2/3
sample.size <- floor(nrow(adult.data) * train.ratio)
train.indices <- sample(1:nrow(adult.data), size = sample.size)
train <- adult.data[train.indices, ]
test <- adult.data[-train.indices, ]

x_train <- data.matrix(train[,-which(names(train) == "y")]) # Training covariates. 
y_train <- train[,c("y")] # Training label.
x_test <- data.matrix(test[,-which(names(test) == "y")]) # Testing covariates. 
y_test <- test[,c("y")] # Testing label.

ANN <- keras_model_sequential() %>%
  layer_dense(units = 18, activation = 'relu', input_shape = c(ncol(data))) %>%
  layer_dense(units = 9, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

summary(ANN)

# compile (define loss and optimizer)
ANN %>% compile(loss = 'binary_crossentropy',
                  optimizer = optimizer_rmsprop(),
                  metrics = c('accuracy'))
# train (fit)
history <- ANN %>% fit(x_train, y_train, epochs = 20, 
                         batch_size = 1024, validation_split = 0.2)
# plot
plot(history)

# evaluate on training data. 
ANN %>% evaluate(x_train, y_train)

# evaluate on test data. 
ANN %>% evaluate(x_test, y_test)

y_pred <- ANN %>% predict(x_test) %>% `>`(0.5) %>% k_cast("int32")
y_pred <- as.array(y_pred)
(tab <- table("Predictions" = y_pred, "Labels" = y_test))
confusionMatrix(factor(y_pred), factor(y_test))
roc(response = y_test, predictor = as.numeric(y_pred), plot = T)

# Linear model (logistic regression).
lin_mod <- glm(y ~ ., family=binomial(link='logit'), data=train)
summary(lin_mod)
y_pred_logreg <- predict(lin_mod, test, type = "response")
y_pred_logreg[y_pred_logreg >= 0.5] <- 1
y_pred_logreg[y_pred_logreg < 0.5] <- 0
confusionMatrix(factor(y_pred_logreg), factor(y_test))
roc(response = y_test, predictor = as.numeric(y_pred_logreg), plot = T)

prediction_model <- function(x,method){
  if (method == "logreg"){
    return(predict(lin_mod, x, type = "response")) 
  } else if (method == "ANN"){
    # This will not work as I want to with the ANN.
    y_pred <- ANN %>% predict(x_test) %>% `>`(0.5) %>% k_cast("int32")
    y_pred <- as.array(y_pred)
    return(y_pred)
  } else {
    stop("Methods 'logreg' and 'ANN' are the only two implemented thus far")
  }
}

# Add the predictions to the dataframe. Here we choose the logistic regression for now!
new_predicted_data <- cbind(test, "y_pred" = y_pred_logreg)

############################################ This is where the generation algorithm begins. 
data_min_response <- adult.data[,-which(names(adult.data) == "y")] # All covariates (removed the response from the data frame).
# Do not think that this is being used anywhere at this point!

K <- 100 # Number of returned possible counterfactuals before pre-processing.
fixed_features <- c("age", "sex") # Names of fixed features from the data. 
mut_features <- base::setdiff(colnames(data_min_response), fixed_features) # Names of mutable features from the data.
mut_datatypes <- sapply(data_min_response[mut_features], class)
u <- length(fixed_features) # Number of fixed features. 
q <- length(mut_features) # Number of mutable features. 
p <- q+u # Total number of features.
all.equal(ncol(data_min_response), p) # We want to check that p is correctly defined. Looks good!

adult.data <- adult.data[,c(fixed_features, mut_features)] # Rearrange the data in order to match the ordering of D_h.
# This is simply an implementation detail (for the steps in the post-processing).

# Fit the regression trees and add all these objects to a list.
T_j <- list() # Vector of fitted trees!
fixed_form <- paste(fixed_features, collapse = "+") # Fixed features, for making the formula. 
total_formulas <- list()
for (i in 1:q){
  #print(i)
  covariates <- paste(c(fixed_features,mut_features[1:i-1]), collapse = "+")
  tot_form <- as.formula(paste(mut_features[i]," ~ ", covariates, sep= ""))
  total_formulas[[i]] <- tot_form
  print(tot_form)
  if (mut_datatypes[[i]] == "factor"){ 
    #T_j[[i]] <- tree(tot_form, data = adult.data, control = tree.control(nobs = nrow(adult.data), mincut = 80, minsize = 160), split = "gini", x = T)
    T_j[[i]] <- rpart(tot_form, data = adult.data, method = "class", control = rpart.control(minsplit = 2, minbucket = 1)) 
    # Method = "class": Uses Gini index, I believe. Check the docs again. 
  } else if (mut_datatypes[[i]] == "numeric"){ # mean squared error.
    #T_j[[i]] <- tree(tot_form, data = adult.data, control = tree.control(nobs = nrow(adult.data), mincut = 5, minsize = 10), split = "deviance", x = T)
    T_j[[i]] <- rpart(tot_form, data = adult.data, method = "anova", control = rpart.control(minsplit = 2, minbucket = 1)) 
    # Method = "anova": SST-(SSL+SSR). Check out the docs. This should (hopefully) be the same as Mean Squared Error. 
  } else { 
    stop("Error: Datatypes need to be either factor or numeric.")
  } # Flere av trærne som blir kun en root node. Mulig noe må endres på!?
}

plot_tree <- function(index){
  # Helper function to plot each tree nicely (to see if it makes sense). Also prints the formula that was used to construct the tree. 
  par(mar = c(1,1,1,1))
  cat("Formula fitted: ")
  print(total_formulas[[index]])
  cat("\n")
  tree.mod <- T_j[[index]]
  print(summary(tree.mod))
  if (tree.mod$method == "class"){
    rpart.plot::prp(tree.mod, extra = 4)  
  } else {
    rpart.plot::prp(tree.mod)
  }
  
  
}

plot_tree(1)
plot_tree(2)
plot_tree(3)
plot_tree(4)
plot_tree(5)
plot_tree(6)
plot_tree(7)
plot_tree(8)
plot_tree(9)
plot_tree(10)
plot_tree(11)
plot_tree(12)

############### Generate counterfactuals based on trees etc. 
# Generate counterfactual per sample. 
generate <- function(h, K = K){ # Use K from above as standard.
  # Instantiate entire D_h-matrix for all features. 
  D_h <- as.data.frame(matrix(data = rep(NA, K*p), nrow = K, ncol = p))
  colnames(D_h) <- c(fixed_features, mut_features)
  

  # Fill the matrix D_h with copies of the vectors of fixed features. 
  # All rows should have the same value in all the fixed features. 
  D_h[,fixed_features] <- h %>% dplyr::select(all_of(fixed_features))
  D_h[, mut_features] <- h %>% dplyr::select(all_of(mut_features)) # Add this to get the correct datatypes (these are not used when predicting though!)

  # Now setup of D_h is complete. We move on to the second part, where we append columns to D_h. 
  
  for (j in 1:q){
    cat("feat",feature_regressed <- mut_features[j], "\n")
    cat("dtype",feature_regressed_dtype <- mut_datatypes[[j]], "\n")
    
    d <- rep(NA, K) # Empty vector of length K. 
    # Will be inserted into D_h later (could col-bind also, but chose to instantiate entire D_h from the beginning).
    
    for (i in 1:K){
      # Add a single sample from the end node of tree T_j[j] based on data D_h[i,u+j] to d[i].
      cat("end",end_node_distr <- predict(T_j[[j]], newdata = D_h[i,1:(u+j-1)]), "\n")
      sorted <- sort(end_node_distr, decreasing = T, index.return = T)
      largest_class <- sorted$x
      largest_index <- sorted$ix
      cat("largest",largest_class,"\n")
      cat("largest_index",largest_index,"\n")
      if (feature_regressed_dtype == "factor"){
        cat("s:",s <- runif(1),"\n")
        if (s >= largest_class[1]){ # This only works for two classes at this point!
          d[i] <- levels(adult.data[,feature_regressed])[largest_index[2]]
        } else {
          d[i] <- levels(adult.data[,feature_regressed])[largest_index[1]]
        }
      } else { # Numeric
        d[i] <- end_node_distr
      }
    }
    D_h[,u+j] <- d # Add all the tree samples based on the jth mutable feature to the next column. 
  }
  D_h
}

# Points we want to explain. This is the list of factuals. Based on logreg for now. The ML model it is based on is set above, in "new_predicted_data".
# Here we say that we want to explain predictions that are predicted as 0 (less than 50k a year). We want to find out what we need to change to change
# this prediction into 1. This is done in the post-processing after generating all the possible counterfactuals. 
preds <- prediction_model(test, method = "logreg")
preds[preds >= 0.5] <- 1
preds[preds < 0.5] <- 0
new_predicted_data <- cbind(test[,colnames(adult.data)], "y_pred" = preds)
H <- new_predicted_data[new_predicted_data["y_pred"] == 0, -which(names(new_predicted_data) %in% "y_pred")] 

# Generation of counterfactuals for each point, before post-processing.
generate_counterfact_for_H <- function(){
  D_h_per_point <- list()
  for (i in 1:nrow(H)){
    # x_h is a factual. 
    x_h <- H[i,]
    D_h_per_point[[i]] <- generate(x_h)
  }
  D_h_per_point
}

x_h <- H[1,]
D_h <- generate(x_h, K = 100)

####################### Post-processing.
# Remove the rows of D_h not satisfying the listed criteria. 

# Fulfill criterion 3.
c <- 0.5 # Threshold for removal. Not sure what value this should have to begin with. We assume a binary response in our case. 
D_h_crit3 <- D_h[prediction_model(D_h, method = "logreg") >= c,] # prediction_model(*) is the R function that predicts 
# according to the model we want to make explanations for. 
# We can see that many rows are the same. The duplicates are removed below. 
unique_D_h <- unique(D_h_crit3)

# fulfilling criterion 4.
# Sparsity and Gower's distance.
# These are calculated for each x_h separately!

# Calculate Sparsity first: Number of features changed between x_h and the counterfactual.
unique_D_h$sparsity <- rep(NA, nrow(unique_D_h))
for (i in 1:nrow(unique_D_h)){
  unique_D_h[i,"sparsity"] <- sum(x_h != unique_D_h[i,-ncol(unique_D_h)]) 
}

# Calculate Gower's distance next. 
library(gower) # Could try to use this package instead of calculating everything by hand below!
unique_D_h$gower <- rep(NA, nrow(unique_D_h))
for (i in 1:nrow(unique_D_h)){
  g <- 0 # Sum for Gower's distance.
  p <- ncol(x_h)
  dtypes <- sapply(x_h[colnames(x_h)], class)
  for (j in 1:p){ # Assuming that the features are already normalized! Perhaps they need to be normalized again!?
    d_j <- unique_D_h[i,j]
    if (dtypes[j] == "numeric"){
      R_j <- 1 # normalization factor, see not in line above.
      g <- g + 1/R_j*abs(d_j-x_h[,j])
    } else if (dtypes[j] == "factor"){
      if (x_h[,j] != d_j){
        g <- g + 1
      }
    }
  }
  unique_D_h[i,"gower"] <- g/p
}
