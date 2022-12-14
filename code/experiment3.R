# Experiment 3: Reproduce results from MCCE with a ANN.

rm(list = ls())  # make sure to remove previously loaded variables into the Session. Just in case. 

setwd("/home/ajo/gitRepos/project")
library(rpart) # For building CART trees. 
library(rpart.plot) # For plotting rpart trees in a more fancy way.
library(dplyr) # For data manipulation. 
library(keras) # For deep learning models. 
library(pROC) # For ROC curve.
library(hmeasure) # For AUC (I am testing this for comparison to pROC).
library(caret) # For confusion matrix.

# Source some of the needed code. 
source("code/utilities.R")

# Get command line arguments.
CLI.args <- take.arguments()
# Arguments: method, length(H), K, generate (TRUE) or load (FALSE), binarized data (TRUE) or not (FALSE)
print(CLI.args)

# Parameter for choosing standardscaler or not. 
standardscaler = T

set.seed(42) # Set seed to begin with!

# Load the data we want first. Loading and cleaning the original data is done in separate files. 
# We also load the classifier (and the testing data) based on the binarized or the categorical data. 
if (CLI.args[5]){
  load("data/adult_data_binarized.RData", verbose = T) # Binarized factors in the data. 
  ANN <- load_model_hdf5("classifiers/ANN_experiment3.h5") # Load the classifier for step 1. 
  load("data/exp3_data/test_data_exp3_ANN.RData", verbose = T)
  normalization_constants <- read.csv("data/exp3_data/normalization_constants_exp3.csv")
} else if (CLI.args[5] != T){
  load("data/adult_data_categ.RData", verbose = T) # Categorical factors as they come originally. 
  ANN <- load_model_hdf5("classifiers/ANN_experiment4.h5") # Load the classifier for step 1. 
  load("data/exp4_data/test_data_exp4_ANN.RData", verbose = T)
  normalization_constants <- read.csv("data/exp4_data/normalization_constants_exp4.csv")
} else {
  stop("Please supply either T (binarized data) of F (categorical data) as the fit CLI argument.")
}

# Quickly check how the fitted classifier actually performs. 
x_test_ANN <- test_ANN[,-which(names(test_ANN) == "y")]
y_test_ANN <- test_ANN[,"y"]

ANN %>% evaluate(data.matrix(x_test_ANN), y_test_ANN)
y_pred <- ANN %>% predict(data.matrix(x_test_ANN)) #%>% `>=`(0.5) #%>% k_cast("int32")
print(confusionMatrix(factor(as.numeric(y_pred %>% `>=`(0.5))), factor(y_test_ANN)))
print(roc(response = y_test_ANN, predictor = as.numeric(y_pred), plot = T))
results <- HMeasure(y_test_ANN,as.numeric(y_pred),threshold=0.5)
print(results$metrics$AUC)

# Make sure we have the correct normalization constants for later. 
m <- normalization_constants$m
M <- normalization_constants$M

cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
# List of categorical variables (used to reverse onehot encode later!)
categ <- setdiff(names(adult.data), cont)
categ <- categ[-length(categ)] # Remove the label "y"!

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
  if (mut_datatypes[[i]] == "factor"){ 
    T_j[[i]] <- rpart(tot_form, data = adult.data, method = "class", control = rpart.control(minbucket = 5, cp = 1e-6)) 
    # Method = "class": Uses Gini index by default. 
  } else if (mut_datatypes[[i]] == "integer" || mut_datatypes[[i]] == "numeric"){ # mean squared error.
    T_j[[i]] <- rpart(tot_form, data = adult.data, method = "anova", control = rpart.control(minbucket = 5, cp = 1e-10)) 
    # Method = "anova": SST-(SSL+SSR).
  } else { 
    stop("Error: Datatypes need to be either factor or integer/numeric.") # We need to use "numeric" if we have normalized the data!
  } 
}

# For plotting the trees if desired. 
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
  D_h[,fixed_features] <- h[fixed_features]
  D_h[,mut_features] <- h[mut_features]
  
  # Now setup of D_h is complete. We move on to the second part, where we append columns to D_h. 
  
  for (j in 1:q){
    feature_regressed <- mut_features[j]
    feature_regressed_dtype <- mut_datatypes[[j]]
    
    d <- rep(NA, K) # Empty vector of length K. 
    # Will be inserted into D_h later (could col-bind also, but chose to instantiate entire D_h from the beginning).
    
    for (i in 1:K){
      # Add a single sample from the end node of tree T_j[j] based on data D_h[i,u+j] to d[i].
      end_node_distr <- predict(T_j[[j]], newdata = D_h[i,1:(u+j-1)]) # Usikker p?? om "predict" blir korrekt her? Burde det v??rt en "where" for ?? finne indeks f??rst?
      sorted <- sort(end_node_distr, decreasing = T, index.return = T)
      largest_class <- sorted$x
      largest_index <- sorted$ix
      if (feature_regressed_dtype == "factor"){
        d[i] <- sample(x = levels(adult.data[,feature_regressed])[largest_index], size = 1, prob = largest_class) 
      } else { # Numeric.
        d[i] <- end_node_distr
      }
    }
    D_h[,u+j] <- d # Add all the tree samples based on the jth mutable feature to the next column. 
  }
  D_h[,colnames(adult.data)[-length(colnames(adult.data))]] %>% mutate_if(is.character,as.factor) 
  # Change characters to factors.
  # We also rearrange the columns to match the column orders in the original data. 
  # This is an implementation detail that is done to be able to easier calculated sparsity etc in the pre-processing. 
}

# Make predictions from ANN on all of the test data.
preds <- as.numeric(ANN %>% predict(data.matrix(x_test_ANN)))

# Add predictions to original set of testing data. Then locate 100 points that are unfavourably predicted. 
new_predicted_data <- data.frame(cbind(adult.data[row.names(test_ANN), ], "y_pred" = preds)) 

# Use CLI.args to make the name of the file automatically.
filename_generation <- paste(CLI.args[1],"_H",CLI.args[2],"_K",CLI.args[3],"_bin",CLI.args[5], sep="") 

if (CLI.args[4]) {
  H <- new_predicted_data[new_predicted_data$y_pred < 0.5, ] 
  s <- sample(1:nrow(H), size = CLI.args[2]) # Sample CLI.args[2] random points from H.
  # We also simply assume that nrow(H) > CLI.args[2].
  H <- H[s,-which(names(new_predicted_data) %in% c("y_pred","y"))]  
} else if (CLI.args[4] != T){
  load(paste0("results/Hs/H_",filename_generation,".RData"), verbose = T)
} 

K <- as.numeric(CLI.args[3]) # Assume that the third command line input is an integer. 

# Generate counterfactuals for each of the 100 points in H.
generate_counterfact_for_H <- function(H_l, K.num){
  D_h_per_point <- list()
  for (i in 1:nrow(H_l)){
    # x_h is a factual. 
    x_h <- H_l[i,]
    D_h_per_point[[i]] <- generate(x_h, K = K.num) 
    cat("Generated for point ",i,"\n")
  }
  return(D_h_per_point)
}

if (CLI.args[4]){
  D_h_per_point <- generate_counterfact_for_H(H_l = H, K) # Generate the matrix D_h for each factual we want to explain (in H)
  save(D_h_per_point, file = paste("results/D_hs/",filename_generation,".RData",sep="")) # Save the generated D_h per point.
  save(H, file = paste("results/Hs/H_",filename_generation,".RData",sep=""))
} else if (CLI.args[4] != T){
  load(paste("results/D_hs/",filename_generation,".RData",sep=""), verbose = T)
}

########### Post-processing and calculating performance metrics!
post.processing <- function(D_h, H, data){ # 'data' is used to calculate normalization factors for Gower.
  # Remove the rows of D_h (per point) not satisfying the listed criteria. 

  # Find the normalization factors for Gower.
  norm.factors <- list()
  for (i in 1:length(colnames(data))){
    colm <- (data %>% dplyr::select(colnames(data)[i]))[[1]]
    if (class(colm) == "integer" || class(colm) == "numeric"){
      q <- quantile(colm, c(0.01, 0.99))
      norm.factors[[i]] <- c(q[1][[1]],q[2][[1]]) # Divide each term in Gower by M_j-m_j, but with 0.99 and 0.01 quantiles respectively.
    } else {
      norm.factors[[i]] <- NA
    }
  }
  
  fulfill_crit3_D_h <- function(D_h, c){
    # Build design matrix manually to avoid contrast problems with factors with missing levels (when not generating "enough" data).
    col_names <- colnames(x_test_ANN)
    col_names_categ <- setdiff(col_names,cont)
    onehot_test_dat <- data.frame(D_h[,cont])
    col_names_D_h <- colnames(D_h)
    col_names_D_h <- setdiff(col_names_D_h,cont)
    for (n in col_names_D_h){
      true_factors <- levels(adult.data[,n]) # Find the factors we want from adult.data
      for (new_col in 1:length(true_factors)){ # Make one new column per factor in the design matrix.
        column_name_new <- paste0(n,"..",substring(true_factors[new_col], 2, nchar(true_factors[new_col])))
        onehot_test_dat[,column_name_new] <- ifelse(D_h[n] == true_factors[new_col], 1,0)
      }
    }
    # Now the manual design matrix has been built!
    
    # Normalize the data before prediction.
    if (standardscaler){
      d_onehot_test <- scale(onehot_test_dat[,cont], center = m, scale = M)
      catego <- setdiff(names(onehot_test_dat), cont)
      onehot_test_dat <- cbind(d_onehot_test, onehot_test_dat[,catego])[,colnames(onehot_test_dat)]
    } else {
      # min-max normalization according to mins and maxes from training data. 
      for (j in 1:length(cont)){
        cont_var <- cont[j]
        onehot_test_dat[,cont_var] <- (onehot_test_dat[,cont_var]-m[j])/(M[j]-m[j])
      }
    }
    
    predictions <- as.numeric(ANN %>% predict(data.matrix(onehot_test_dat)))
    D_h_crit3 <- D_h[predictions >= c,] 
    # We can see that many rows are the same. The duplicates are removed below. 
    unique_D_h <- unique(D_h_crit3)
    return(unique_D_h)
  }
  
  # Fulfill criterion 3.
  fulfill_crit3 <- function(D_h_pp, c){
    for (i in 1:length(D_h_pp)){
      D_h <- D_h_pp[[i]]
      D_h_pp[[i]] <- fulfill_crit3_D_h(D_h, c)
    }
    return(D_h_pp)
  }
  
  ##### Fulfilling criterion 4.
  # Calculate Sparsity and Gower's distance for each D_h (per point).
  
  add_metrics_D_h_all_points <- function(D_h_pp, H_l, norm.factors){
    # Calculates sparsity and Gower for all counterfactuals and adds the columns to each respective D_h.
    for (i in 1:length(D_h_pp)){
      D_h_pp[[i]] <- gower_D_h(H_l[i,], D_h_pp[[i]], norm.factors)
      D_h_pp[[i]] <- sparsity_D_h(H_l[i,], D_h_pp[[i]])
    }
    return(D_h_pp)
  }
  
  # Remove non-valid counterfactuals (those that don't change the prediction).
  crit3_D_h_per_point <- fulfill_crit3(D_h_pp = D_h, c = 0.5) # Fulfill criterion 3 for all (unique) generated possible counterfactuals. 
  
  # Add sparsity and Gower distance to each row. 
  crit4_D_h_all_points <- add_metrics_D_h_all_points(crit3_D_h_per_point,H, norm.factors)
  return(crit4_D_h_all_points) # Return D_h with Gower and sparsity added as columns. Also, non-valid counterfactuals are removed. 
}

D_h_post_processed <- post.processing(D_h_per_point, H, adult.data[,-14])

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

# Averages of all the metrics calculated and added to unique_D_h.
L0s <- c()
L2s <- c()
N_CEs <- rep(NA, length(final_counterfactuals))
for (i in 1:length(final_counterfactuals)){
  l <- final_counterfactuals[[i]]
  n <- nrow(l)
  N_CEs[i] <- n
  if (n >= 1){
    L0s <- c(L0s,l$sparsity)
    L2s <- c(L2s,l$gower)
  } 
}

exp_MCCE <- data.frame("L0" = mean(L0s), "L2" = mean(L2s), "N_CE" = sum(N_CEs))
knitr::kable(exp_MCCE)
write.csv(exp_MCCE, file = paste("results/resulting_metrics_", filename_generation, ".csv", sep = ""))
save(final_counterfactuals, file = paste("results/final_counterfactuals_", filename_generation, ".RData", sep = ""))

# After generation is done, make latex tables I can paste into report. 
knitr::kable(exp_MCCE, format = "latex", linesep = "", digits = 4, booktabs = T) %>% print()
