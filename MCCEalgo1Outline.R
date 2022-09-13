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
library(keras) # for deep learning models. 
library(pROC) # For ROC curve.
library(caret) # For confusion matrix.

# Loading and cleaning the original data is done in separate files. 

########################################### Build ML models for classification: which individuals obtain an income more than 50k yearly?

# Load the data we want first. 
load("adult_data_binarized.RData", verbose = T)

# Normalize the continuous variables.
cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")

normalize <- function(x) {
  return((x- min(x))/(max(x)-min(x)))
}

# Save the minimums and maximums such that I can transform back later!
mins <- c()
maxs <- c()

for (j in cont){
  adult.data[,j] <- normalize(adult.data[,j])
  mins <- c(mins, min(adult.data[,j]))
  maxs <- c(maxs, max(adult.data[,j]))
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

# compile (define loss and optimizer)
ANN %>% compile(loss = 'binary_crossentropy',
                  optimizer = optimizer_rmsprop(),
                  metrics = c('accuracy'))

# train (fit)
history <- ANN %>% fit(x_train, y_train, epochs = 20, 
                         batch_size = 1024, validation_split = 0.2)
# plot
plot(history)

summary(ANN)

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

# This is used to return the predicted probabilities according to the model we want to use (for later).
prediction_model <- function(x,method){
  # This returns the predicted probabilities of class 1 (>= 50k per year).
  if (method == "logreg"){
    return(predict(lin_mod, x, type = "response")) 
  } else if (method == "ANN"){
    return(as.numeric(ANN %>% predict(x_test)))
  } else {
    stop("Methods 'logreg' and 'ANN' are the only two implemented thus far")
  }
}

# Add the predictions to the dataframe. Here we choose the logistic regression for now!
# THIS IS NOT NEEDED RIGHT NOW I BELIEVE. 
# new_predicted_data <- cbind(test, "y_pred" = y_pred_logreg)

############################################ This is where the generation algorithm begins. 
data_min_response <- adult.data[,-which(names(adult.data) == "y")] # All covariates (removed the response from the data frame).

K <- 100 # Number of returned possible counterfactuals before pre-processing.
fixed_features <- c("age", "sex") # Names of fixed features from the data. 
mut_features <- base::setdiff(colnames(data_min_response), fixed_features) # Names of mutable features from the data.
mut_datatypes <- sapply(data_min_response[mut_features], class)
u <- length(fixed_features) # Number of fixed features. 
q <- length(mut_features) # Number of mutable features. 
p <- q+u # Total number of features.
all.equal(ncol(data_min_response), p) # We want to check that p is correctly defined. Looks good!

#adult.data <- adult.data[,c(fixed_features, mut_features)] # Rearrange the data in order to match the ordering of D_h.
# This is simply an implementation detail (for the steps in the post-processing).

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

############################### Generate counterfactuals based on trees etc. 
# Generate counterfactual per sample. 
generate <- function(h, K = 100){ # Use K from above as standard. K = 10000 is used in the article for the experiments.
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
      end_node_distr <- predict(T_j[[j]], newdata = D_h[i,1:(u+j-1)])
      sorted <- sort(end_node_distr, decreasing = T, index.return = T)
      largest_class <- sorted$x
      largest_index <- sorted$ix
      if (feature_regressed_dtype == "factor"){
        s <- runif(1)
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
  D_h[,colnames(adult.data)[-length(colnames(adult.data))]] %>% mutate_if(is.character,as.factor) # Change characters to factors! THIS IS NOT TESTED THOROUGHLY BUT SEEMS TO WORK OK.
  # We also rearrange the columns to match the column orders in the original data. 
  # This is an implementation detail that is done to be able to easier calculated sparsity etc in the pre-processing. 
}
 
# Points we want to explain. This is the list of factuals. 
# Here we say that we want to explain predictions that are predicted as 0 (less than 50k a year). We want to find out what we need to change to change
# this prediction into 1. This is done in the post-processing after generating all the possible counterfactuals. According to the experiments in the article
# we only generate one counterfactual per factual, for the first 100 undesirable observations we want to explain.
preds <- prediction_model(test, method = "logreg")
#preds_sorted <- sort(preds, decreasing = F, index.return = T) # Vil tro det kanskje ikke er dette de er på jakt etter!?
#preds_sorted_values <- preds_sorted$x[1:10] # Skal egentlig ha de 100 første! Gjør dette for testing nå!
#preds_sorted_indices <- preds_sorted$ix[1:10]
#new_predicted_data <- cbind(test[preds_sorted_indices,colnames(adult.data)], "y_pred" = preds_sorted_values)
new_predicted_data <- cbind(test[,colnames(adult.data)], "y_pred" = preds)
H <- new_predicted_data[new_predicted_data$y_pred < 0.5, ] 
H <- H[1:20,-which(names(new_predicted_data) %in% c("y_pred","y"))] # Prøver bare med de 10 første foreløpig.
# We select the 100 test observations with the lowest predicted probability? Perhaps this is not what they mean by 
# "the first 100 observations with an undesirable prediction"? 
# I have gone away from this for now!

# Generation of counterfactuals for each point, before post-processing.
generate_counterfact_for_H <- function(H_l = H){
  D_h_per_point <- list()
  for (i in 1:nrow(H_l)){
    # x_h is a factual. 
    x_h <- H_l[i,]
    D_h_per_point[[i]] <- generate(x_h, K = 500) # I artikkelen hadde de 10000.
    cat("Generated for point ",i,"\n")
  }
  D_h_per_point
}

D_h_per_point <- generate_counterfact_for_H() # Generate the matrix D_h for each factual we want to explain (in H)
save(D_h_per_point, file = "H20K500.RData") # Save the generated D_h per point with K = 100 for the first 100 undesirable predictions.
#load("H10K1000.RData", verbose = T)


######################################## Post-processing.
# Remove the rows of D_h (per point) not satisfying the listed criteria. 

fulfill_crit3_D_h <- function(D_h, c, pred.method){
  D_h_crit3 <- D_h[prediction_model(D_h, method = pred.method) >= c,] # prediction_model(*) is the R function that predicts 
  # according to the model we want to make explanations for. 
  # We can see that many rows are the same. The duplicates are removed below. 
  unique_D_h <- unique(D_h_crit3)
}

# Fulfill criterion 3.
fulfill_crit3 <- function(D_h_pp, c = 0.5, pred.method = "logreg"){
  for (i in 1:length(D_h_pp)){
    D_h <- D_h_pp[[i]]
    D_h_pp[[i]] <- fulfill_crit3_D_h(D_h, c, pred.method)
  }
  D_h_pp
}

crit3_D_h_per_point <- fulfill_crit3(D_h_pp = D_h_per_point) # Fullfil criterion 3 for all (unique) generated possible counterfactuals. 

##### fulfilling criterion 4.
# Calculate Sparsity and Gower's distance for each D_h (per point).

# Calculate Sparsity first: Number of features changed between x_h and the counterfactual.
sparsity_D_h <- function(x_h,D_h){
  # Calculates sparsity for one counterfactual x_h.
  D_h$sparsity <- rep(NA, nrow(D_h))
  if (nrow(D_h) >= 1){
    for (i in 1:nrow(D_h)){
      D_h[i,"sparsity"] <- sum(x_h != D_h[i,-ncol(D_h)]) # We remove the last column, which is D_h$sparsity!
    }
  }
  return(D_h)
}

sparsity_D_h_all_points <- function(D_h_pp, H_l){
  # Calculates sparsity for all counterfactuals and adds the column to each respective D_h.
  for (i in 1:length(D_h_pp)){
    D_h_pp[[i]] <- sparsity_D_h(H_l[i,], D_h_pp[[i]])
  }
  return(D_h_pp)
}

# Testing for sparsity, seems to work fine!
sparsity_D_h(H[2,],crit3_D_h_per_point[[2]])
sparsity_D_h(H[1,],crit3_D_h_per_point[[1]])
sparsity_D_h(H[3,],crit3_D_h_per_point[[3]])
sparsity_D_h(H[4,],crit3_D_h_per_point[[4]])

spars_D_h_per_point <- sparsity_D_h_all_points(crit3_D_h_per_point,H)
spars_D_h_per_point[[2]]
spars_D_h_per_point[[1]]
spars_D_h_per_point[[3]]
spars_D_h_per_point[[4]]

gower_D_h <- function(x_h, D_h){
  # Calculates Gower's distance for one counterfactual x_h.
  library(gower) # Could try to use this package instead of calculating everything by hand below!
  D_h$gower <- rep(NA, nrow(D_h))
  D_h$gowerpack <- rep(NA, nrow(D_h)) # Result from package.
  dtypes <- sapply(x_h[colnames(x_h)], class)
  
  if (nrow(D_h) >= 1){
    for (i in 1:nrow(D_h)){
      g <- 0 # Sum for Gower's distance.
      p <- ncol(x_h)
      
      for (j in 1:p){ # Assuming that the features are already normalized! Perhaps they need to be normalized again!?
        d_j <- D_h[i,j]
        if (dtypes[j] == "numeric"){
          R_j <- 1 # normalization factor, see not in line above.
          g <- g + 1/R_j*abs(d_j-x_h[,j])
        } else if (dtypes[j] == "factor"){
          if (x_h[,j] != d_j){
            g <- g + 1
          }
        }
      }
      # Disse to er ulike!! Finn ut hvorfor!? Noe med normaliseringen jeg nevner ovenfor å gjøre?
      # Det kommer en warning om "zero or non-finite range" ved bruke av pakken!
      # Kanskje jeg bare skal sjekke at det gir mening det jeg har gjort manuelt først!
      D_h[i,"gower"] <- g/p
      #D_h[i,"gowerpack"] <- gower_dist(x_h,D_h[i,colnames(x_h)])
    }
  }
  return(D_h)
}

gower_D_h_all_points <- function(D_h_pp, H_l){
  # Calculates Gower's distance for all counterfactuals and adds the column to each respective D_h.
  for (i in 1:length(D_h_pp)){
    D_h_pp[[i]] <- gower_D_h(H_l[i,], D_h_pp[[i]])
  }
  return(D_h_pp)
}

# Tester Gower's distance om det virker korrekt i implementasjonen. Virker som at denne delen av implementasjonen er god!
gower_D_h(H[1,],spars_D_h_per_point[[1]])
gower_D_h(H[2,],spars_D_h_per_point[[2]])

crit4_D_h_all_points <- gower_D_h_all_points(D_h_pp = spars_D_h_per_point, H_l = H)
crit4_D_h_all_points[[1]]
crit4_D_h_all_points[[2]]
all.equal(crit4_D_h_all_points[[2]],gower_D_h(H[2,],spars_D_h_per_point[[2]]))

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

final_counterfactuals <- generate_one_counterfactual_all_points(crit4_D_h_all_points)

########################### Performance metrics. All these should be calculated on "final_counterfactuals"!
# Violation: Number of actionability constraints violated by the counterfactual. 
# This should inherently be zero if I have implemented the algorithm correctly!! 
# Thus, this is an ok check to do. 
unique_D_h$violation <- rep(NA, nrow(unique_D_h))
for (i in 1:nrow(unique_D_h)){
  unique_D_h[i,"violation"] <- sum(x_h[,fixed_features] != unique_D_h[i,fixed_features]) 
}

# Success: if the counterfactual produces a positive predictive response.
# This is 1 inherently, from the post-processing step done above (where we only keep the rows in D_h that have positive predictive response).
prediction_model(unique_D_h, method = "logreg") # As we can see, success = 1 for these counterfactuals. 

# Feasibility: distance between the counterfactual and the training data.
# As in the article, we choose Euclidean distance, k = 1/5 and w^[i] = 1/k = 1/5.

feasibility <- function(){
  k <- 5
  w <- 1/k
  p <- ncol(x_h)
  euclidean <- function(x1,x2) sqrt(sum(x1-x2)^2)
  f <- 0
  e <- unique_D_h[1,]
  
  # The line below is not feasible! Perhaps need to loop over each row and save each answer. Then sort after.
  #first_k_distances <- order(as.matrix(dist(rbind(e[,-which(names(e) %in% c("sparsity","gower","violation"))],adult.data)))[1,-1],decreasing = F)[1:k]
  
  # Find k nearest neighbors in dataset. 
  k_nearest <- function(e, k = 5,data = adult.data){
    n <- nrow(adult.data)
    distances <- rep(NA, n)
    for (r in 1:n){
      distances[r] <- euclidean(e,adult.data[r,]) # Hva kan jeg gjøre med factors??
    }
    distances.ordered <- order(distances, decreasing = F)
    return(distances.ordered[1:5])
  }
  
  k_nearest_to_e <- adult.data[k_nearest(e[,-which(names(e) %in% c("sparsity","gower","violation"))], k = 5, data = adult.data),]
  all.equal(k, length(k_nearest_to_e))
  
  for (i in 1:k){
    f <- f + w/p*euclidean(e,k_nearest_to_e[i]) # Kunne sikkert bare brukt distances.ordered her, i stedet for å beregne dette på nytt her!!
  }
  
  # Feasibility SPM: "K nearest observed data points" står det i artikkelen. Mener de da mellom e og dataen?
}


# Experiment 1:
# Averages of all the metrics calculated and added to unique_D_h
# Prøver kun med logreg nå!

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
