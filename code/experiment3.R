# Experiment 3: Reproduce results from MCCE with a ANN.

rm(list = ls())  # make sure to remove previously loaded variables into the Session.

setwd("/home/ajo/gitRepos/project")
library(rpart) # Try this for building CART trees instead!
library(rpart.plot) # For plotting rpart trees in a more fancy way.
library(dplyr)
library(keras) # for deep learning models. 
library(pROC) # For ROC curve.
library(hmeasure) # For AUC (I am testing this for comparison to pROC).
library(caret) # For confusion matrix.

# Source some of the needed code. 
source("code/utilities.R")

# Get command line arguments.
CLI.args <- take.arguments()
# Arguments: method, length(H), K, generate (TRUE) or load (FALSE), binarized data (TRUE) or not (FALSE)
for (i in CLI.args){
  print(i)
}

# Parameter for choosing standardscaler or not. 
standardscaler = T

set.seed(42) # Set seed to begin with!

# Load the data we want first. Loading and cleaning the original data is done in separate files. 
if (CLI.args[5]){
  load("data/adult_data_binarized.RData", verbose = T) # Binarized factors in the data. 
} else if (CLI.args[5] != T){
  load("data/adult_data_categ.RData", verbose = T) # Categorical factors as they come originally. 
} else {
  stop("Please supply either T (binarized data) of F (categorical data) as the fit CLI argument.")
}

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
adult.data.onehot <- make.data.for.ANN(adult.data.onehot, cont) 

#normalized <- normalize.data(data = adult.data.onehot, continuous_vars = cont, standardscaler = standardscaler)
#adult.data.onehot <- normalized[[1]]

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

# compile (define loss and optimizer)
ANN %>% compile(loss = 'binary_crossentropy',
                optimizer = optimizer_adam(), # Could try other optimizers also.  #learning_rate = 0.002
                metrics = c('accuracy'))

# train (fit)
history <- ANN %>% fit(x = data.matrix(x_train), 
                       y = y_train, 
                       epochs = 30, 
                       batch_size = 1024, 
                       validation_data = list(data.matrix(x_valid), y_valid)
                       )

# plot
plot(history)

print(summary(ANN))

# evaluate on training data. 
ANN %>% evaluate(data.matrix(x_train), y_train)

# evaluate on test data. 
ANN %>% evaluate(data.matrix(x_test), y_test)

y_pred <- ANN %>% predict(data.matrix(x_test)) #%>% `>=`(0.5) #%>% k_cast("int32")
print(confusionMatrix(factor(as.numeric(y_pred %>% `>=`(0.5))), factor(y_test)))
print(roc(response = y_test, predictor = as.numeric(y_pred), plot = T))
results <- HMeasure(y_test,as.numeric(y_pred),threshold=0.5)
print(results$metrics$AUC)
######################### Step 1 of predictor fitting is complete!!




#########################Step 2: Use MCCE to generate counterfactuals for 100 randomly sample individuals in the test data.
# First we need to build the trees and build the latent distribution model. 
data_min_response <- adult.data[,-which(names(adult.data) == "y")] # All covariates (removed the response from the data frame).

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
total_formulas <- list()
for (i in 1:q){
  covariates <- paste(c(fixed_features,mut_features[1:i-1]), collapse = "+")
  tot_form <- as.formula(paste(mut_features[i]," ~ ", covariates, sep= ""))
  total_formulas[[i]] <- tot_form
  #print(tot_form)
  if (mut_datatypes[[i]] == "factor"){ 
    #T_j[[i]] <- tree(tot_form, data = adult.data, control = tree.control(nobs = nrow(adult.data), mincut = 80, minsize = 160), split = "gini", x = T)
    #T_j[[i]] <- rpart(tot_form, data = adult.data, method = "class", control = rpart.control(minsplit = 2, minbucket = 1)) 
    T_j[[i]] <- rpart(tot_form, data = adult.data, method = "class", control = rpart.control(minbucket = 5, cp = 1e-6)) 
    # Method = "class": Uses Gini index, I believe. Check the docs again. 
  } else if (mut_datatypes[[i]] == "integer" || mut_datatypes[[i]] == "numeric"){ # mean squared error.
    #T_j[[i]] <- tree(tot_form, data = adult.data, control = tree.control(nobs = nrow(adult.data), mincut = 5, minsize = 10), split = "deviance", x = T)
    #T_j[[i]] <- rpart(tot_form, data = adult.data, method = "anova", control = rpart.control(minsplit = 2, minbucket = 1)) 
    T_j[[i]] <- rpart(tot_form, data = adult.data, method = "anova", control = rpart.control(minbucket = 5, cp = 1e-10)) 
    # Method = "anova": SST-(SSL+SSR). Check out the docs. This should (hopefully) be the same as Mean Squared Error. 
  } else { 
    stop("Error: Datatypes need to be either factor or integer/numeric.") # We need to use "numeric" if we have normalized the data!
  } 
}

# Med så liten cp blir det et problem å plotte trærne!
# plot_tree(1)
# plot_tree(2)
# plot_tree(3)
# plot_tree(4)
# plot_tree(5)
# plot_tree(6)
# plot_tree(7)
# plot_tree(8)
# plot_tree(9)
# plot_tree(10)
# plot_tree(11)

############################### Generate counterfactuals based on trees etc. 
# Generate counterfactual per sample. 
generate <- function(h, K){ # K = 10000 is used in the article for the experiments.
  # Instantiate entire D_h-matrix for all features. 
  D_h <- as.data.frame(matrix(data = rep(NA, K*p), nrow = K, ncol = p))
  colnames(D_h) <- c(fixed_features, mut_features)
  
  # Fill the matrix D_h with copies of the vectors of fixed features. 
  # All rows should have the same value in all the fixed features. 
  D_h[, fixed_features] <- h %>% dplyr::select(all_of(fixed_features))
  D_h[, mut_features] <- h %>% dplyr::select(all_of(mut_features)) # Add this to get the correct datatypes (these are not used when predicting though!)
  
  # Now setup of D_h is complete. We move on to the second part, where we append columns to D_h. 
  
  for (j in 1:q){
    feature_regressed <- mut_features[j]
    feature_regressed_dtype <- mut_datatypes[[j]]
    
    d <- rep(NA, K) # Empty vector of length K. 
    # Will be inserted into D_h later (could col-bind also, but chose to instantiate entire D_h from the beginning).
    
    for (i in 1:K){
      # Add a single sample from the end node of tree T_j[j] based on data D_h[i,u+j] to d[i].
      end_node_distr <- predict(T_j[[j]], newdata = D_h[i,1:(u+j-1)]) # Usikker på om "predict" blir korrekt her? Burde det vært en "where" for å finne indeks først?
      sorted <- sort(end_node_distr, decreasing = T, index.return = T)
      largest_class <- sorted$x
      largest_index <- sorted$ix
      if (feature_regressed_dtype == "factor"){
        # s <- runif(1)
        # if (s >= largest_class[1]){ # This only works for two classes at this point! Perhaps I can simply use the sample function with the list of probabilities?
        #   d[i] <- levels(adult.data[,feature_regressed])[largest_index[2]]
        # } else {
        #   d[i] <- levels(adult.data[,feature_regressed])[largest_index[1]]
        # }
        # I think the following is a better solution. This works for the categorical data as well!
        d[i] <- sample(x = levels(adult.data[,feature_regressed])[largest_index], size = 1, prob = largest_class) 
      } else { # Numeric
        d[i] <- end_node_distr
      }
    }
    D_h[,u+j] <- d # Add all the tree samples based on the jth mutable feature to the next column. 
  }
  D_h[,colnames(adult.data)[-length(colnames(adult.data))]] %>% mutate_if(is.character,as.factor) 
  # Change characters to factors! THIS IS NOT TESTED THOROUGHLY BUT SEEMS TO WORK OK.
  # We also rearrange the columns to match the column orders in the original data. 
  # This is an implementation detail that is done to be able to easier calculated sparsity etc in the pre-processing. 
}

# Make predictions from ANN on all of the the test data.
preds <- as.numeric(ANN %>% predict(data.matrix(x_test)))

# Add predictions to original set of testing data. Then locate 100 points that are unfavourably predicted. 
new_predicted_data <- data.frame(cbind(adult.data[row.names(test), ], "y_pred" = preds)) 
H <- new_predicted_data[new_predicted_data$y_pred < 0.5, ] 
s <- sample(1:nrow(H), size = CLI.args[2]) # Sample CLI.args[2] random points from H.
H <- H[s,-which(names(new_predicted_data) %in% c("y_pred","y"))] 

K <- as.numeric(CLI.args[3]) # Assume that the third command line input is an integer. 

# Generate counterfactuals for each of the 100 points in H.
generate_counterfact_for_H <- function(H_l, K.num){
  D_h_per_point <- list()
  for (i in 1:nrow(H_l)){
    # x_h is a factual. 
    x_h <- H_l[i,]
    D_h_per_point[[i]] <- generate(x_h, K = K.num) # I artikkelen hadde de 10000.  
    cat("Generated for point ",i,"\n")
  }
  return(D_h_per_point)
}

# Use CLI.args to make the name of the file automatically.
filename_generation <- paste(CLI.args[1],"_H",CLI.args[2],"_K",CLI.args[3],"_bin",CLI.args[5], sep="") 
if (CLI.args[4]){
  D_h_per_point <- generate_counterfact_for_H(H_l = H, K) # Generate the matrix D_h for each factual we want to explain (in H)
  save(D_h_per_point, file = paste("results/D_hs/",filename_generation,".RData",sep="")) # Save the generated D_h per point.
} else if (CLI.args[4] != T){
  load(paste("results/D_hs/",filename_generation,".RData",sep=""), verbose = T)
}

####### Post-processing and calculating performance metrics!
post.processing <- function(D_h, H, data){ # 'data' is used to calculate normalization factors for Gower.
  # Remove the rows of D_h (per point) not satisfying the listed criteria. 
  
  # Find the normalization factors for Gower.
  norm.factors <- list()
  for (i in 1:length(colnames(data))){
    colm <- (data %>% select(colnames(data)[i]))[[1]]
    if (class(colm) == "integer" || class(colm) == "numeric"){
      q <- quantile(colm, c(0.01, 0.99))
      norm.factors[[i]] <- c(q[1][[1]],q[2][[1]]) # Using min-max scaling, but with 0.01 and 0.99 quantiles!
    } else {
      norm.factors[[i]] <- NA
    }
  }
  
  fulfill_crit3_D_h <- function(D_h, c, pred.method){
    #D_h <- D_h_per_point[[1]]
    #pred.method <- "ANN"
    if (pred.method == "ANN"){
      onehot_test_dat <- as.data.frame(model.matrix(~.,data = D_h)) # Det er her den failer (for D_h'er som ikke har nok verdier!!)
      predictions <- prediction_model(onehot_test_dat, method = pred.method)
    } else {
      predictions <- prediction_model(D_h, method = pred.method)
    }
    #c <- 0.5
    D_h_crit3 <- D_h[predictions >= c,] # prediction_model(*) is the R function that predicts 
    # according to the model we want to make explanations for. 
    # We can see that many rows are the same. The duplicates are removed below. 
    unique_D_h <- unique(D_h_crit3)
    return(unique_D_h)
  }
  
  # Fulfill criterion 3.
  fulfill_crit3 <- function(D_h_pp, c, pred.method){
    for (i in 1:length(D_h_pp)){
      D_h <- D_h_pp[[i]]
      D_h_pp[[i]] <- fulfill_crit3_D_h(D_h, c, pred.method)
      #D_h_pp[[i]] <- # Make sure that the counterfactuals are actionable (not necessary for trees, necessary for VAE).
    }
    return(D_h_pp)
  }
  
  ##### Fulfilling criterion 4.
  # Calculate Sparsity and Gower's distance for each D_h (per point).
  
  add_metrics_D_h_all_points <- function(D_h_pp, H_l, norm.factors){
    # Calculates sparsity and Gower for all counterfactuals and adds the columns to each respective D_h.
    #D_h_pp <- D_h_per_point
    #H_l <- H
    for (i in 1:length(D_h_pp)){
      D_h_pp[[i]] <- gower_D_h(H_l[i,], D_h_pp[[i]], norm.factors)
      D_h_pp[[i]] <- sparsity_D_h(H_l[i,], D_h_pp[[i]])
    }
    return(D_h_pp)
  }
  
  # Remove non-valid counterfactuals (those that don't change the prediction).
  crit3_D_h_per_point <- fulfill_crit3(D_h_pp = D_h, c = 0.5, pred.method = CLI.args[1]) # Fulfill criterion 3 for all (unique) generated possible counterfactuals. 
  
  # Add sparsity and Gower distance to each row. 
  crit4_D_h_all_points <- add_metrics_D_h_all_points(crit3_D_h_per_point,H, norm.factors)
  return(crit4_D_h_all_points) # Return D_h with Gower and sparsity added as columns. Also, non-valid counterfactuals are removed. 
}

D_h_post_processed <- post.processing(D_h_per_point, H, adult.data[,-14])

# Sjekker at alt fungerer som det skal!
# crit3_D_h_per_point <- fulfill_crit3(D_h_per_point, 0.5, CLI.args[1])
# d <- D_h_per_point[[3]]
# d$relationship <- factor(d$relationship, levels = c(levels(d$relationship), "Husband"))
# onehot_test_dat <- as.data.frame(model.matrix(~.,data = d, contrasts.arg = list(
#   relationship = contrasts(adult.data$relationship, contrasts = FALSE)
# )))
# Ser at jeg må legge til ekstra levels for hver faktor der det mangler en level! (for at det skal være mulig å lage en model.matrix!)
# Finnes det noen annen måte jeg kan gjøre dette på!?!?? Høre med Kjersti!!!


############ Do we want several counterfactuals per factual or only one? Below we select one!
generate_one_counterfactual_D_h <- function(D_h){
  # Generate one counterfactual for one factual, i.e. reduce the corresponding D_h to 1 row.
  if (nrow(D_h) >= 1){
    # Find minimum sparsity among all rows, remove all rows with sparsity larger than this.
    min_sparsity <- min(D_h$sparsity)
    D_h <- D_h[D_h$sparsity == min_sparsity, ]
    
    # Next: Among the remaining rows, find the row with the smallest Gower distance.   
    index <- which(D_h$gower == min(D_h$gower), arr.ind = T)
    D_h <- D_h[index,]
  }
  return(D_h)
}

generate_one_counterfactual_all_points <- function(D_h_pp){
  # Generate one counterfactual for all factuals, i.e. reduce each D_h to 1 row. 
  for (i in 1:length(D_h_pp)){
    gen <- generate_one_counterfactual_D_h(D_h_pp[[i]])
    if (nrow(gen) > 1){ # It may happen that a several possible counterfactuals have the same sparsity and Gower distance.
      # In such cases we simply choose the first one.
      D_h_pp[[i]] <- gen[1,]
    } else {
      D_h_pp[[i]] <- gen
    }
    
  }
  return(D_h_pp)
}


final_counterfactuals <- generate_one_counterfactual_all_points(D_h_post_processed)

########################### Performance metrics. All these should be calculated on "final_counterfactuals"!
# Violation: Number of actionability constraints violated by the counterfactual. 
# This should inherently be zero if I have implemented the algorithm correctly!! 
# Thus, this is an ok check to do. 

violate <- function(){
  # We leave this out for now!
  unique_D_h$violation <- rep(NA, nrow(unique_D_h))
  for (i in 1:nrow(unique_D_h)){
    unique_D_h[i,"violation"] <- sum(x_h[,fixed_features] != unique_D_h[i,fixed_features]) 
  }
  
  # Success: if the counterfactual produces a positive predictive response.
  # This is 1 inherently, from the post-processing step done above (where we only keep the rows in D_h that have positive predictive response).
  prediction_model(unique_D_h, method = CLI.args[1]) # As we can see, success = 1 for these counterfactuals. 
  
}

############################## Experiments. 
# Experiment 1:
# Averages of all the metrics calculated and added to unique_D_h.

L0s <- c()
L2s <- c()
N_CEs <- rep(NA, length(final_counterfactuals))
for (i in 1:length(final_counterfactuals)){
  l <- final_counterfactuals[[i]]
  n <- nrow(l)
  N_CEs[i] <- n
  if (n >= 0){
    L0s <- c(L0s,l$sparsity)
    L2s <- c(L2s,l$gower)
  } 
}

exp1_MCCE <- data.frame("L0" = mean(L0s), "L2" = mean(L2s), "N_CE" = sum(N_CEs))
knitr::kable(exp1_MCCE)
write.csv(exp1_MCCE, file = paste("results/resulting_metrics_", filename_generation, ".csv", sep = ""))
save(final_counterfactuals, file = paste("results/final_counterfactuals_", filename_generation, ".RData", sep = ""))
