# Implementation of the Modified MCCE (using VAE for step 1 and 2 in MCCE by Aas et al.)

# Add some data to test the code. 
# Adult data set: 
# https://archive.ics.uci.edu/ml/datasets/Adult

rm(list = ls())  # make sure to remove previously loaded variables into the Session.

setwd("/home/ajo/gitRepos/project")
library(rpart) # Try this for building CART trees instead!
library(rpart.plot) # For plotting rpart trees in a more fancy way.
library(dplyr)
library(keras) # for deep learning models. 
library(pROC) # For ROC curve.
library(hmeasure) # For AUC (I am testing this for comparison to pROC).
library(caret) # For confusion matrix.
library(ranger) # For implementing a random forest classifier.

# Source some of the needed code. 
source("code/utilities.R")

# Get command line arguments.
CLI.args <- take.arguments()
# Arguments: method, length(H), K, generate (TRUE) or load (FALSE), binarized data (TRUE) or not (FALSE)
for (i in CLI.args){
  print(i)
}

# Just for testing right now, should be removed later!
CLI.args <- c("logreg",10,100, TRUE, TRUE)

########################################### Build ML models for classification: which individuals obtain an income more than 50k yearly?
set.seed(42) # Set seed to begin with!

# Load the data we want first. Loading and cleaning the original data is done in separate files. 
if (CLI.args[5]){
  load("data/adult_data_binarized.RData", verbose = T) # Binarized factors in the data. 
} else if (CLI.args[5] != T){
  load("data/adult_data_categ.RData", verbose = T) # Categorical factors as they come originally. 
} else {
  stop("Please supply either T (binarized data) of F (categorical data) as the fit CLI argument.")
}

# List of continuous variables.
cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
# List of categorical variables. 
categ <- setdiff(names(adult.data), cont)
categ <- categ[-length(categ)] # Remove the label "y"!

if (CLI.args[1] == "ANN"){
  adult.data.onehot <- data.frame(adult.data) # make a copy of the dataframe for one hot encoding in ANN.
  tracemem(adult.data) == tracemem(adult.data.onehot) # it is a deep copy.
  data.table::address(adult.data)
  data.table::address(adult.data.onehot)
  # The memory addresses are different. 

  # Make the design matrix for the DNN.
  adult.data.onehot <- make.data.for.ANN(adult.data, cont) # Compare this with the model.matrix approach from below, if time. 
  
  # Make train and test data for our model matrix adult.data.
  train_and_test_data <- make.train.and.test(data = adult.data.onehot) # The function returns two matrices (x) and two vectors (y). 
  # In addition, it returns the indices of the data set used in the training data. 
  summary(train_and_test_data) # Returned list. 
  x_train <- train_and_test_data[[1]]
  y_train <- train_and_test_data[[2]]
  x_test <- train_and_test_data[[3]]
  y_test <- train_and_test_data[[4]]
  # The train indices are used to construct H later. 
  train_indices <- train_and_test_data[[5]]
  
  # The data is normalized (for performance) when we want to use the ANN as a predictive model. 
  # It is done after splitting into training and testing, to avoid data leakage!
  # Normalize the data AFTER splitting to avoid data leakage!
  x_train.normalization <- normalize.data(data = x_train, continuous_vars = cont, standardscaler = T) # returns list with data, mins and maxs.
  x_train <- x_train.normalization[[1]] # we are only interested in the data for now. 
  x_test.normalization <- normalize.data(data = x_test, continuous_vars = cont, standardscaler = T) # returns list with data, mins and maxs.
  x_test <- x_test.normalization[[1]] # we are only interested in the data for now. 
  
  # Scale the adult data as well, such that we can compare the data sets after generation. 
  adult.data.normalization <- normalize.data(data = adult.data, continuous_vars = cont, standardscaler = T)
  adult.data <- adult.data.normalization[[1]]
} else {
  # Make train and test data for our adult.data.
  train_and_test_data <- make.train.and.test(data = adult.data) 
  summary(train_and_test_data) # Returned list. 
  x_train <- train_and_test_data[[1]]
  y_train <- train_and_test_data[[2]]
  x_test <- train_and_test_data[[3]]
  y_test <- train_and_test_data[[4]]
  train_indices <- train_and_test_data[[5]]
}

if (CLI.args[1] == "ANN"){
  ANN <- fit.ANN(data.matrix(x_train), y_train, data.matrix(x_test), y_test)
} else if (CLI.args[1] == "logreg"){
  logreg <- fit.logreg(x_train, y_train, x_test, y_test)
} else if (CLI.args[1] == "randomForest"){
  random.forest <- fit.random.forest(x_train, y_train, x_test, y_test)
} else {
  stop("We have only implemented three prediction models: 'ANN', 'logreg' or 'randomForest'.")
}

# This is used to return the predicted probabilities according to the model we want to use (for later).
prediction_model <- function(x_test,method){
  # This returns the predicted probabilities of class 1 (>= 50k per year).
  # We need to make sure that we feed the ANN with the proper design/model matrix for it to work!
  if (method == "logreg"){
    return(predict(logreg, data.frame(x_test), type = "response")) 
  } else if (method == "ANN"){
    return(as.numeric(ANN %>% predict(data.matrix(x_test))))
  } else if (method == "randomForest"){
    return(predict(random.forest, x_test)$predictions[,2]) 
  } else { # This 'stop' is strictly not necessary since I checked this earlier in the compilation, but it is ok to leave here. 
    stop("Methods 'logreg', 'ANN' and 'randomForest' are the only implemented thus far")
  }
}

######################## This is where distribution modeling begins.  
data_min_response <- adult.data[,-which(names(adult.data) == "y")] # All covariates (removed the response from the data frame).

fixed_features <- c("age", "sex") # Names of fixed features from the data. 
mut_features <- base::setdiff(colnames(data_min_response), fixed_features) # Names of mutable features from the data.
mut_datatypes <- sapply(data_min_response[mut_features], class)
u <- length(fixed_features) # Number of fixed features. 
q <- length(mut_features) # Number of mutable features. 
p <- q+u # Total number of features.
all.equal(ncol(data_min_response), p) # We want to check that p is correctly defined. Looks good!

##### Fit the VAE to the data. 

####### Make data for the VAE.
# Make the design matrix for the DNN.
adult.data.onehot <- make.data.for.ANN(adult.data, cont) # Compare this with the model.matrix approach from below, if time. 

# Make train and test data for our model matrix adult.data.
train_and_test_data <- make.train.and.test(data = adult.data.onehot) # The function returns two matrices (x) and two vectors (y). 
# In addition, it returns the indices of the data set used in the training data. 
summary(train_and_test_data) # Returned list. 
x_train_vae <- train_and_test_data[[1]]
y_train_vae <- train_and_test_data[[2]]
x_test_vae <- train_and_test_data[[3]]
y_test_vae <- train_and_test_data[[4]]
# The train indices are used to construct H later. 
train_indices_vae <- train_and_test_data[[5]]

# The data is normalized (for performance) when we want to use the ANN as a predictive model. 
# It is done after splitting into training and testing, to avoid data leakage!
# Normalize the data AFTER splitting to avoid data leakage!
x_train.normalization <- normalize.data(data = x_train_vae, continuous_vars = cont, standardscaler = T) # returns list with data, mins and maxs.
x_train_vae <- x_train.normalization[[1]] # we are only interested in the data for now. 
x_test.normalization <- normalize.data(data = x_test_vae, continuous_vars = cont, standardscaler = T) # returns list with data, mins and maxs.
x_test_vae <- x_test.normalization[[1]] # we are only interested in the data for now. 

####### Build the VAE.
latent_dim <- 8L # This sets the size of the latent representation (vector), 
intermediate_dim <- 15L
# defined by the parameters z_mean and z_log_var. 
input_size <- ncol(x_train_vae) 

####### First we define the encoder. 
enc_input <- layer_input(shape = c(input_size), name = "enc_input")
layer_one <- layer_dense(enc_input, units=intermediate_dim, activation = "relu", name = "enc_hidden")
enc_mean <- layer_dense(layer_one, latent_dim, name = "enc_mean")
enc_log_var <- layer_dense(layer_one, latent_dim, name = "enc_log_var")

# Sampling function in the latent space. It samples the Gaussian distribution by using the mean and variance
# that will be learned. It returns a sampled latent vector. 
sampling <- function(args){
  # Here we clearly assume a Gaussian distribution via the method of sampling from the assumed latent distribution as shown below. 
  c(z_mean, z_log_var) %<-% args
  epsilon <- k_random_normal(shape = k_shape(z_mean),
                             mean = 0, stddev = 1) # Draws samples from standard normal, adds element-wise to a vector. 
  return(z_mean + k_exp(z_log_var/2) * epsilon) # element-wise exponential, in order to transform the standard normal samples to 
  # sampling from our latent variable distribution (the reparametrization trick).
}

# Point randomly sampled from the variational / approximated posterior.   
z <- list(enc_mean, enc_log_var) %>% 
  layer_lambda(sampling, name = "enc_output")


#latent_inputs <- layer_input(shape = c(latent_dim), name = "dec_input")
decoder_layer <- layer_dense(units = intermediate_dim, activation = "relu", name = "dec_hidden")
outputs <- layer_dense(units = input_size, name = "dec_output") 
# Could perhaps use the sigmoid activation to restrict data to (0,1) since we are dealing with normalized input data. 
# outputs <- layer_dense(decoder_layer, input_size) # We test this here.  

vae_encoder_output <- decoder_layer(z)
vae_decoder_output <- outputs(vae_encoder_output)
vae <- keras_model(enc_input, vae_decoder_output, name = "VAE")
summary(vae)

vae_loss <- function(enc_mean, enc_log_var){
  # Loss function for our VAE (with Gaussian assumptions).
  vae_reconstruction_loss <- function(y_true, y_predict){
    loss_factor <- 100 # Give weight to the reconstruction in the loss function ("hyperparameter")
    #reconstruction_loss <- metric_mean_squared_error(y_true, y_predict) 
    #reconstruction_loss <- loss_binary_crossentropy(y_true, y_predict) # Or binary cross entropy?
    reconstruction_loss <- k_mean(k_square(y_true - y_predict))
    return(reconstruction_loss*loss_factor)
  }
  
  vae_kl_loss <- function(encoder_mu, encoder_log_variance){
    kl <- -0.5*k_sum(1 + encoder_log_variance - k_square(encoder_mu) - k_exp(encoder_log_variance), axis = -1L) # Or axis = -1?
    return(kl)
  }
  
  v_loss <- function(y_true, y_pred){
    reconstruction <- vae_reconstruction_loss(y_true, y_pred)
    kl <- vae_kl_loss(enc_mean, enc_log_var)#*input_size # Trying to scale the KL divergence such that it is of a similar scale as the reconstruction loss.
    return(reconstruction + kl) # + reconstruction
  }
  return(v_loss)
}

# Define encoder. 
encoder <- keras_model(enc_input, list(enc_mean, enc_log_var, z), name = "encoder_model") 
summary(encoder)

# Define decoder. 
decoder_input <- layer_input(shape = latent_dim)
decoder_hidden <- decoder_layer(decoder_input)
decoder_output <- outputs(decoder_hidden)
decoder <- keras_model(decoder_input, decoder_output, name = "decoder_model")
summary(decoder)


vae %>% compile(optimizer = optimizer_adam(learning_rate = 1e-2), loss = vae_loss(enc_mean, enc_log_var))
summary(vae)

# Add some callbacks.
callbacks_list <- list(
  callback_early_stopping(
    monitor = "val_loss",
    patience=3
  ),
  callback_reduce_lr_on_plateau(
    monitor="val_loss",
    factor = 0.5,
    patience = 2
  )
)

# Training/fitting the model.
history <- vae %>% fit(
  x = data.matrix(x_train_vae),
  y = data.matrix(x_train_vae),
  shuffle = T,
  epochs = 30,
  batch_size = 16,
  validation_data = list(data.matrix(x_test_vae), data.matrix(x_test_vae)),
  verbose = 1,
  callbacks = callbacks_list
)

plot(history)


############################### Generate counterfactuals based on the modeled data from the VAE. 
# Generate counterfactuals per factual (sample).
generate <- function(K){ 
  variational_parameters <- encoder %>% predict(data.matrix(x_test_vae)) # Trening?
  v_means <- variational_parameters[[1]]
  v_log_vars <- variational_parameters[[2]]
  v_zs <- variational_parameters[[3]]
  
  # This is the second generation method from "VAE.R" (Add uniform noise to the zs produced from the test data inputted to the encoder).
  r <- slice_sample(as.data.frame(v_zs), n = K, replace = T) # Sample K rows from the z's. 
  s <- jitter(data.matrix(r), amount = 0) # Add some uniform noise to r.
  
  # Decode our latent sample s.
  decoded_data_rand <- decoder %>% predict(s)
  decoded_data_rand <- as.data.frame(decoded_data_rand)
  colnames(decoded_data_rand) <- colnames(x_train_vae) 
  #head(decoded_data_rand)
  
  # Revert the one-hot encoding.
  decoded_data_rand <- reverse.onehot.encoding(decoded_data_rand, cont, categ, has.label = F)
  
  ####### Re-center and re-scale the data set "back" to our original scales. 
  # There is no clear way of doing it for the generated/decoded data, but we will use the center and the scale of the test-data here. 
  data_orig2 <- t(apply(decoded_data_rand[,cont], 1, function(r)r*x_test.normalization$sds + x_test.normalization$means))
  data_orig2 <- cbind(data_orig2, as.data.frame(decoded_data_rand[,-which(names(decoded_data_rand) %in% cont)]))
  decoded_data_rand <- data_orig2
  
  return(decoded_data_rand[,colnames(adult.data)[-length(colnames(adult.data))]] %>% mutate_if(is.character,as.factor)) # Return the decoded data ("fake" data).
}
 
if (CLI.args[1] %in% c("ANN", "logreg", "randomForest")){
  preds <- prediction_model(x_test, method = CLI.args[1]) 
} else { # Perhaps we need to add a special if else for ANN here! (because of the model matrix being different!)
  stop("Please supply either 'ANN', 'logreg' or 'randomForest' as the first CLI argument.")
}

new_predicted_data <- data.frame(cbind(adult.data[-train_indices, ], "y_pred" = preds)) 
H <- new_predicted_data[new_predicted_data$y_pred < 0.5, ] 
s <- sample(1:nrow(H), size = CLI.args[2]) # Sample CLI.args[2] random points from H.
H <- H[s,-which(names(new_predicted_data) %in% c("y_pred","y"))] 

K <- as.numeric(CLI.args[3]) # Assume that the third command line input is an integer. 

# Generation of counterfactuals for each point, before post-processing.
generate_counterfact_for_H <- function(H_l, K.num){
  D_h_per_point <- list()
  for (i in 1:nrow(H_l)){
    # x_h is a factual. 
    x_h <- H_l[i,] # This is not needed for the generation from the VAE, since we do not condition on any values. 
    D_h_per_point[[i]] <- generate(K)
    cat("Generated for point ",i,"\n")
  }
  return(D_h_per_point)
}

# Use CLI.args to make the name of the file automatically.
filename_generation <- paste(CLI.args[1],"_H",CLI.args[2],"_K",CLI.args[3],"_bin",CLI.args[5], sep="") 
if (CLI.args[4]){
  D_h_per_point <- generate_counterfact_for_H(H_l = H, K) # Generate the matrix D_h for each factual we want to explain (in H)
  save(D_h_per_point, file = paste("resultsVAE/D_hs/",filename_generation,".RData",sep="")) # Save the generated D_h per point.
} else if (CLI.args[4] != T){
  load(paste("resultsVAE/D_hs/",filename_generation,".RData",sep=""), verbose = T)
}

############################################# Post-processing.

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
  
  make_actionable <- function(D_h, fixed_covariates, factual){
    # Function used to remove all points from D_h which do not have the correct fixed covariate values as in the factual.
    factual_values <- factual[fixed_covariates]
    return(D_h[D_h[,fixed_covariates] == factual_values])
  }
  
  fulfill_crit2 <- function(D_h_pp, H){
    # Make sure that each possible counterfactual (per factual) has the correct fixed values. 
    # This is not necessary for MCCE, but is necessary for ModMCCE (using VAE, not conditioned on the fixed features).
    for (i in 1:length(D_h_pp)){
      D_h <- D_h_pp[[i]]
      D_h$age <- round(D_h$age) # This is done to be certain that ages are whole numbers. Should definitely be done somewhere else!
      D_h_pp[[i]] <- make_actionable(D_h, fixed_features, H[i,])# Make sure that the counterfactuals are actionable (not necessary for trees, necessary for VAE).
    }
    return(D_h_pp)
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
  crit3_D_h_per_point <- fulfill_crit3(D_h_pp = list_of_values, c = 0.5, pred.method = CLI.args[1]) # Fulfill criterion 3 for all (unique) generated possible counterfactuals. 
  
  # Add sparsity and Gower distance to each row. 
  crit4_D_h_all_points <- add_metrics_D_h_all_points(crit3_D_h_per_point,H, norm.factors)
  return(crit4_D_h_all_points) # Return D_h with Gower and sparsity added as columns. Also, non-valid counterfactuals are removed. 
}

D_h_post_processed <- post.processing(D_h_per_point, H, adult.data[,-14])

# Sjekker at alt fungerer som det skal!
crit3_D_h_per_point <- fulfill_crit3(D_h_per_point, 0.5, CLI.args[1])
d <- D_h_per_point[[3]]
d$relationship <- factor(d$relationship, levels = c(levels(d$relationship), "Husband"))
onehot_test_dat <- as.data.frame(model.matrix(~.,data = d, contrasts.arg = list(
  relationship = contrasts(adult.data$relationship, contrasts = FALSE)
)))
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
