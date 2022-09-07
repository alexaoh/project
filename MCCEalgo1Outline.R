# Writing the outline for algorithm 1 in MCCE by Aas et al.

# Next need to add some data to test this out. 
# Adult data set: 
# https://archive.ics.uci.edu/ml/datasets/Adult

# Give me some credit data: 
# https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset

setwd("/home/ajo/gitRepos/project")

library(tree) # For regression trees. 
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
set.seed(42)
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
y_pred_logreg[y_pred_logreg >= 0.5 ] <- 1
y_pred_logreg[y_pred_logreg < 0.5 ] <- 0
confusionMatrix(factor(y_pred_logreg), factor(y_test))
roc(response = y_test, predictor = as.numeric(y_pred_logreg), plot = T)

############################################ This is where the generation algorithm begins. 
data_min_response <- adult.data[,-which(names(adult.data) == "y")]
H <- y_pred_logreg[y_pred_logreg == 0] # Points we want to explain. This is the list of factuals. Based on logreg for now. 
K <- 2 # Number of returned possible counterfactuals before pre-processing.
fixed_features <- c("age", "sex") # Names of fixed features from the data. 
mut_features <- base::setdiff(colnames(data_min_response), fixed_features) # Names of mutable features from the data.
mut_datatypes <- sapply(data_min_response[mut_features], class)
u <- length(fixed_features) # Number of fixed features. 
q <- length(mut_features) # Number of mutable features. 
p <- q+u # Total number of features.
all.equal(ncol(data_min_response), p) # We want to check that p is correctly defined. Looks good!


# Fit the regression trees and add all these objects to a list.
T_j <- list() # Vector of fitted trees!
fixed_form <- paste(fixed_features, collapse = "+") # Fixed features, for making the formula. 
for (i in 1:q){
  #print(i)
  covariates <- paste(c(fixed_features,mut_features[1:i-1]), collapse = "+")
  tot_form <- as.formula(paste(mut_features[i]," ~ ", covariates, sep= ""))
  print(tot_form)
  if (mut_datatypes[[i]] == "factor"){ 
    T_j[[i]] <- tree(tot_form, data = train, control = tree.control(nobs = nrow(train), mincut = 80, minsize = 160), split = "gini", x = T)
  } else if (mut_datatypes[[i]] == "numeric"){
    T_j[[i]] <- tree(tot_form, data = train, control = tree.control(nobs = nrow(train), mincut = 1, minsize = 2), split = "deviance", x = T)
  } else {
    stop("Error: Datatypes need to be either factor or numeric.")
  } # Noe rart med de som bruker gini index her!!!! Mulig disse trærne må bygges bedre senere, men nå er de i hvert fall der!
      # Den som brukes deviance splittes ikke heller tror jeg! Generelt sett noe som må gjøres med trærne her!
}

plot_tree <- function(tree.mod){
  # Helper function to plot each tree nicely (to see if it makes sense).
  print(summary(tree.mod))
  plot(tree.mod)
  text(tree.mod, pretty = 0)
}

plot_tree(T_j[[1]])
plot_tree(T_j[[2]])
plot_tree(T_j[[3]])
plot_tree(T_j[[4]])
plot_tree(T_j[[5]])

############### Generate counterfactuals based on trees etc. 
# Generate counterfactual per sample. 
generate <- function(h){
  # Instantiate entire D_h-matrix for all features. 
  D_h <- as.data.frame(matrix(data = rep(NA, K*p), nrow = K, ncol = p))
  colnames(D_h) <- c(fixed_features, mut_features)
  
  
  # Fill the matrix D_h with copies of the vectors of fixed features. 
  # All rows should have the same value in all the fixed features. 
  D_h[,fixed_features] <- h %>% dplyr::select(fixed_features) 

  # Now setup of D_h is complete. We move on to the second part, where we append columns to D_h. 
  
  for (j in 1:q){
    d <- rep(NA, K) # Empty vector of length K. 
    # Will be inserted into D_h later (could col-bind also, but chose to instantiate entire D_h from the beginning).
    
    for (i in 1:K){
      # Add a single sample from the end node of tree T_j[j] based on data D_h[i,u+j] to d[i].
      # Kan tenkes at det blir noe indekseringstrøbbel her!! Første tre skal kun være basert på de fikserte featuresene!!
      d[i] <- predict(T_j[[j]], newdata = D_h[i,1:(u+j-1)], type = "class") # predict blir feil! Ønsker å legge til en training sample som faller innunder leaf node her!
      # Tree where kan kanskje brukes! Se docs for å se hva jeg faktisk kan bruke!
      # kan bruke tree$where og tree$x eller tree$y her tenker jeg. Må finne ut hvordan jeg skal lage liste over treningsdataene per node!
    }
    D_h[,u+j] <- d # Add all the tree samples based on the jth mutable feature to the next column. 
  }
  D_h
}

generate(data_min_response[1,]) # Wrong for classification trees only with class! Need to fix something for regression trees in the loop above. 

# Generation of counterfactuals for each point, before post-processing.
for (x_h in H){
  # x_h is a factual. 
  D_h <- generate(x_h)
}

# Post-processing.
# fulfilling criterion 3.
c <- 0.6 # Threshold for removal. Not sure what value this should have to begin with. We assume a binary response in our case. 
D_h <- D_h[f(D_h) >= c,] # f(*) is the R function that predicts according to the model we want to make explanations for. 

# fulfilling criterion 4.
# Sparsity and Gower's distance.
