# Writing the outline for algorithm 1 in MCCE by Aas et al.

# Next need to add some data to test this out. 
# Adult data set: 
# https://archive.ics.uci.edu/ml/datasets/Adult

# Give me some credit data: 
# https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset

setwd("/home/ajo/gitRepos/project")

library(tree)
library(dplyr)
library(keras)

# How to use the library tree with some example data (for future reference)
# library(ISLR)
# data("Carseats")
# set.seed(4268)
# n = nrow(Carseats)
# train = sample(1:n, 0.7 * n, replace = F)
# test = (1:n)[-train]
# Carseats.train = Carseats[train, ]
# Carseats.test = Carseats[-train, ]
# tree.mod = tree(Sales ~ ., data = Carseats.train)
# summary(tree.mod)
# plot(tree.mod)
# text(tree.mod, pretty = 0)


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
adult.data$y[adult.data$y == ">50K"] <- 0
adult.data$y[adult.data$y == "<=50K"] <- 1
adult.data$y <- as.factor(adult.data$y)

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
adult.data$race <- binarize(adult.data$race)
adult.data$native_country <- binarize(adult.data$native_country)

# write.csv(adult.data, file = "adult_data_binarized.csv", row.names = F)
save(adult.data, file = "adult_data_binarized.RData") # Save the dataset including all factors etc.

data <- adult.data[,-which(names(adult.data) == "y")] # Training features. 
y <- adult.data[,c("y")] # Training label.

########################################### Build ML models for classification: which individuals obtain an income more than 50k yearly?

# We should normalize the continuous variables. This will be done later. 

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
history <- ANN %>% fit(data.matrix(data), y, epochs = 20, 
                         batch_size = 1024, validation_split = 0.2)
# plot
plot(history)

# evaluate on training data. 
model %>% evaluate(x_train, y_train)

# evaluate on test data. 
model %>% evaluate(x_test, y_test)


############################################ This is where the generation algorithm begins. 
H <- 1:10 # Points we want to explain. This is the list of factuals. 
K <- 5 # Number of returned possible counterfactuals before pre-processing.
fixed_features <- c("age", "sex") # Names of fixed features from the data. 
mut_features <- base::setdiff(colnames(data), fixed_features) # Names of mutable features from the data.
u <- length(fixed_features) # Number of fixed features. 
q <- length(mut_features) # Number of mutable features. 
p <- q+u # Total number of features.
all.equal(dim(data)[2], p) # We want to check that p is correctly defined. Looks good!

T_j <- 1:10 # Vector of fitted trees!

# Generate counterfactual per sample. 
generate <- function(h){
  # Instantiate entire D_h-matrix for all features. 
  D_h <- as.data.frame(matrix(data = rep(NA, K*p), nrow = K, ncol = p))
  colnames <- c(fixed_features, mut_features)
  
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
      d[i] <- predict(T_j[j], data = D_h[i,u+j]) # predict blir feil! Ønsker å legge til en training sample som faller innunder leaf node her!
      # Tree where kan kanskje brukes! Se docs for å se hva jeg faktisk kan bruke!
    }
    D_h[,u+j] <- d # Add all the tree samples based on the jth mutable feature to the next column. 
  }
  D_h
}

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
