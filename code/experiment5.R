# Experiment 5: Use Modified MCCE to generate counterfactuals for the same factuals as in MCCE. 
# Then the results can be compared. 

rm(list = ls())  # make sure to remove previously loaded variables into the Session.

# This is needed for the way I have coded the VAE (without eager execution, since I have used the sequential API).
if (tensorflow::tf$executing_eagerly()){
  tensorflow::tf$compat$v1$disable_eager_execution()
}
  
setwd("/home/ajo/gitRepos/project")

library(dplyr)
library(keras) # for deep learning models. 
library(pROC) # For ROC curve.
library(hmeasure) # For AUC (I am testing this for comparison to pROC).
library(caret) # For confusion matrix.
library(MASS)

# Source some of the needed code. 
source("code/utilities.R")

# Get command line arguments.
CLI.args <- take.arguments()
# Arguments: method, length(H), K, generate (TRUE) or load (FALSE), binarized data (TRUE) or not (FALSE)
for (i in CLI.args){
  print(i)
} # We do not really use all these parameters, but we add them here for "resemblance" with exp3 and exp4. 
  # Should make the code more general (all of it) if time before delivery. 

CLI.args <- c("ANN",100,10000,"TRUE","FALSE") # For continuous data. 
# K = 1000000 punkter tar veldig lang tid i minnet.

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
  normalization_constants <- read.csv("data/exp3_data/normalization_constants_exp4.csv")
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
m_ANN <- normalization_constants$m
M_ANN <- normalization_constants$M

cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
# List of categorical variables (used to reverse onehot encode later!)
categ <- setdiff(names(adult.data), cont)
categ <- categ[-length(categ)] # Remove the label "y"!

############ We do not care about the first CLI-argument (the model).
adult.data.onehot <- data.frame(adult.data) # make a copy of the dataframe for one hot encoding in ANN.
tracemem(adult.data) == tracemem(adult.data.onehot) # it is a deep copy.
data.table::address(adult.data)
data.table::address(adult.data.onehot)
# The memory addresses are different, as we wanted. 

# Make the design matrix for the DNN.
adult.data.onehot <- make.data.for.ANN(adult.data.onehot, cont, label = T) 

# Make train and test data for VAE. 
sample.size <- floor(nrow(adult.data.onehot) * 6/7)
train.indices <- sample(1:nrow(adult.data.onehot), size = sample.size)
train <- adult.data.onehot[train.indices, ]
test <- adult.data.onehot[-train.indices, ]

# Scale training data. 
train.normalization <- normalize.data(data = train, continuous_vars = cont, standardscaler = standardscaler) # returns list with data, mins and maxs.
train <- train.normalization[[1]]
m_vae <- train.normalization[[2]]
M_vae <- train.normalization[[3]]

x_train <- train[,-which(names(train) == "y")]
y_train <- train[, "y"] # Use this for visualization later only.

# Make validation data also. THIS IS NOT NECESSARY WHEN FOR THE VAE, SINCE I WANT TO TRAIN IT ON ALL THE DATA. 
# I simply use the testing data as validation data when fitting the VAE.
# Similarly as to the trees in MCCE, we use all the data for fitting the VAE (most as training, a little bit as validation data, which is necessary for FNN's).
#sample.size.valid <- floor(nrow(test) * 1/3)
#valid.indices <- sample(1:nrow(test), size = sample.size.valid)
#valid <- test[valid.indices, ]
#test <- test[-valid.indices, ]

# Scaling according to the same values abtained when scaling the training data! This is very important in all applications for generalizability!!
if (standardscaler){
  # Centering and scaling according to scales and centers from training data. 
  d_test <- scale(test[,cont], center = m_vae, scale = M_vae)
  catego <- setdiff(names(test), cont)
  test <- cbind(d_test, test[,catego])[,colnames(test)]
  
  #d_valid <- scale(valid[,cont], center = m, scale = M)
  #catego <- setdiff(names(valid), cont)
  #valid <- cbind(d_valid, valid[,catego])[,colnames(valid)]
} else {
  # min-max normalization according to mins and maxes from training data. 
  for (j in 1:length(cont)){
    cont_var <- cont[j]
    test[,cont_var] <- (test[,cont_var]-m_vae[j])/(M_vae[j]-m_vae[j])
    #valid[,cont_var] <- (valid[,cont_var]-m[j])/(M[j]-m[j])
  }
}

x_test <- test[,-which(names(test) == "y")]

# We add the following lines for similarity with experiment 3 and 4. 
# The fixed features are needed for post-processing (ensuring actionability).
data_min_response <- adult.data[,-which(names(adult.data) == "y")] # All covariates (removed the response from the data frame).
fixed_features <- c("age", "sex") # Names of fixed features from the data. 
mut_features <- base::setdiff(colnames(data_min_response), fixed_features) # Names of mutable features from the data.
mut_datatypes <- sapply(data_min_response[mut_features], class)
u <- length(fixed_features) # Number of fixed features. 
q <- length(mut_features) # Number of mutable features. 
p <- q+u # Total number of features.
all.equal(ncol(data_min_response), p) # We want to check that p is correctly defined. Looks good!


################################## Fit the VAE (for generation of counterfactuals in point 2)
# Build the VAE.
latent_dim <- 8L # This sets the size of the latent representation (vector), 
intermediate_dim <- 15L
# defined by the parameters z_mean and z_log_var. 
input_size <- ncol(x_train) 

####### First we define the encoder. 
enc_input <- layer_input(shape = c(input_size), name = "enc_input")
layer_one <- layer_dense(enc_input, units=intermediate_dim, activation = "relu", name = "enc_hidden")
enc_mean <- layer_dense(layer_one, latent_dim, name = "enc_mean")
enc_log_var <- layer_dense(layer_one, latent_dim, name = "enc_log_var") #activation = "softplus"

# Sampling function in the latent space. It samples the Gaussian distribution by using the mean and variance
# that will be learned. It returns a sampled latent vector. 
sampling <- function(args){
  # Here we clearly assume a Gaussian distribution via the method of sampling from the assumed latent distribution as shown below. 
  c(z_mean, z_log_var) %<-% args
  epsilon <- k_random_normal(shape = k_shape(z_mean),
                             mean = 0, stddev = 1) # Draws samples from standard normal, adds element-wise to a vector. 
  return(z_mean + k_exp(z_log_var/2) * epsilon) # element-wise exponential, in order to transform the standard normal samples to 
  # sampling from our latent variable distribution (the reparameterization trick).
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
    kl <- vae_kl_loss(enc_mean, enc_log_var)
    return(reconstruction + kl) 
  }
  return(v_loss)
}


# vae_loss <- function(y_true, y_pred){
#   xent_loss=1.0*loss_mean_squared_error(y_true, y_pred) # Prøver med mean squared error eller kullback leibler.
#   #xent_loss=(input_size/1.0)*loss_mean_squared_error(input, z_decoded_mean)
#   #xent_loss=(input_size/1.0)*loss_kullback_leibler_divergence(input, z_decoded_mean)
#   kl_loss=-0.5*k_mean(1+enc_log_var-k_square(enc_mean)-k_exp(enc_log_var), axis=-1L)
#   return(xent_loss + kl_loss)
# }

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
  x = data.matrix(x_train),
  y = data.matrix(x_train),
  shuffle = T,
  epochs = 30,
  batch_size = 16,
  validation_data = list(data.matrix(x_test), data.matrix(x_test)), # We do not use validation sets now, use test as validation. 
  verbose = 1,
  callbacks = callbacks_list
)

plot(history)

##############################Step 2: Make some counterfactuals for the same Hs as in experiment 3/4.
# Below we load the Hs from experiment 3 or 4 (for binarized or categorical data respectively).
filename_generation1 <- paste(CLI.args[1],"_H",CLI.args[2],"_K10000_bin",CLI.args[5], sep="") 
load(paste("results/Hs/H_",filename_generation1,".RData",sep=""), verbose = T)

generation_method <- 4 # Choose the method for generation, corresponding to the numbering in the report. 
K <- as.numeric(CLI.args[3]) # Choose the size of the sample from the VAE per factual. 

#K <- 3*nrow(x_train)

generate_latent_sample <- function(generation_method, K){
  # Generate a sample in the latent space according to the given method. 
  # The four different methods are described in the report. 
  # The sample is returned and decoded later, using the decoder. 
  if (generation_method == 4){
    # Sample from prior p(z), N(0,1)
    s <- mvrnorm(n = K, mu = rep(0,latent_dim), Sigma = diag(rep(1,latent_dim)))
    
  } else if (generation_method == 3){
    # Sample from the z's and add some uniform noise. 
    z_s <- (encoder %>% predict(data.matrix(x_train)))[[3]] 
    r <- slice_sample(as.data.frame(z_s), n = K, replace = T) # Sample K rows from the z's. 
    s <- apply(data.matrix(r), 2, jitter, factor = 5, amount = 0) # Add 1/10*(max(column)-min(column)) uniform noise to each value in r. 
    
  } else if (generation_method == 2){
    # Sample K/nrow(x_train) times from latent space PER training data point. 
    # Her gir det ikke helt mening å generere noe annet enn n*nrow(x_train) antall samples, for at det skal bli balansert. 
    # Det samme gjelder vel egentlig i metoden nedenfor?! Dette har jeg valgt å ikke bry meg om her og heller diskutere i rapporten!
    encoded <- encoder %>% predict(data.matrix(x_train))
    mus <- encoded[[1]]
    log_vars <- encoded[[2]]
    
    sample.after <- function(K, v_means, v_log_vars){
      # Sample from one test point. 
      mu <- as.numeric(v_means)
      sigma <- diag(as.numeric((exp(v_log_vars/2))^2))
      mvrnorm(n = K, mu = mu, Sigma = sigma)
    }
    
    s <- matrix(NA, nrow = K, ncol = latent_dim)
    indices <- seq(1,floor(K/nrow(x_train))*nrow(x_train), by = 2) # Trick for placing in the matrix above. 
    for (r in 1:(nrow(x_train))){ # Sample whole times from each training sample latent distribution.
      samp <- sample.after(floor(K/nrow(x_train)), mus[r,], log_vars[r,])
      s[indices[r],] <- samp[1,] # Trick for placing in the matrix above. 
      s[indices[r]+1,] <- samp[2,] # Trick for placing in the matrix above. 
    }
    # Some checks while developing. 
    if (K %% nrow(x_train) != 0){all(is.na(s[(floor(K/nrow(x_train))*nrow(x_train)+1):K,]))}
    any(is.na(s[1:(floor(K/nrow(x_train))*nrow(x_train)),]))
    
    if (K %% nrow(x_train) != 0){
      # Then we need to fill the remaining K - (floor(K/nrow(x_train))*nrow(x_train)) = K %% nrow(x_train) with 1 sample from each randomly sampled input data point. 
      sampl <- sample(x = 1:nrow(x_train), size = K %% nrow(x_train), replace = F) # Sample the remaining data points from the input sample. 
      
      n <- floor(K/nrow(x_train))*nrow(x_train)
      for (r in (1:(K %% nrow(x_train)))){ # Sample one time from the latent distribution of each of the randomly sampled input data points. 
        s[r+n,] <- sample.after(1, mus[sampl[r],], log_vars[sampl[r],])
      }
      # Check while developing --> Now the entire sample has been filled up. 
      any(is.na(s))
    }
    
  } else if (generation_method == 1){
    # Run entire training data through entire VAE K/nrow(x_train) times.
    # Generation method 4 and 3 should be equivalent in theory, but might be different in practice because of seed-problems.
    # They should be similar because sigma and mu should be deterministically given per data point (when running x_train through the encoder).
    data_through <- do.call("rbind", replicate(floor(K/nrow(x_train)), x_train, simplify = FALSE)) # Replicate training set whole times. 
    data_through <- rbind(data_through, slice_sample(x_train, n = K %% nrow(x_train), replace = F)) # Sample the rest without replacement from x_train.
    # Data set to run through model is now constructed — based on replicating the training data "enough" times. 
    s <- (encoder %>% predict(data.matrix(data_through)))[[3]]
  }
  return(s)
}

decoded_latent_sample <- function(s){
  # Decode our latent sample s.
  decoded_data_rand <- decoder %>% predict(s)
  decoded_data_rand <- as.data.frame(decoded_data_rand)
  colnames(decoded_data_rand) <- colnames(x_train) 
  
  # Revert the one hot encoding
  decoded_data_rand <- reverse.onehot.encoding(decoded_data_rand, cont, categ, has.label = F)
  
  # Change the names of the categorical values in the decoded data. This was done with the objective of making matching plots. 
  decoded_data_rand$workclass[decoded_data_rand$workclass == "workclass..Private"] <- " Private"
  decoded_data_rand$workclass[decoded_data_rand$workclass == "workclass..Other"] <- " Other"
  decoded_data_rand$marital_status[decoded_data_rand$marital_status == "marital_status..Other"] <- " Other"
  decoded_data_rand$marital_status[decoded_data_rand$marital_status == "marital_status..Married.civ.spouse"] <- " Married-civ-spouse"
  decoded_data_rand$occupation[decoded_data_rand$occupation == "occupation..Other"] <- " Other"
  decoded_data_rand$occupation[decoded_data_rand$occupation == "occupation..Craft.repair"] <- " Craft-repair"
  decoded_data_rand$relationship[decoded_data_rand$relationship == "relationship..Husband"] <- " Husband"
  decoded_data_rand$relationship[decoded_data_rand$relationship == "relationship..Other"] <- " Other"
  decoded_data_rand$race[decoded_data_rand$race == "race..Other"] <- " Other"
  decoded_data_rand$race[decoded_data_rand$race == "race..White"] <- " White"
  decoded_data_rand$native_country[decoded_data_rand$native_country == "native_country..Other"] <- " Other"
  decoded_data_rand$native_country[decoded_data_rand$native_country == "native_country..United.States"] <- " United-States"
  decoded_data_rand$sex[decoded_data_rand$sex == "sex..Male"] <- " Male"
  decoded_data_rand$sex[decoded_data_rand$sex == "sex..Female"] <- " Female"
 
  # De-normalize the generated data, such that it is on the same scale as the adult data. 
  if (standardscaler){
    data_orig2 <- t(apply(decoded_data_rand[,cont], 1, function(r)r*M_vae + m_vae))
    data_orig2 <- cbind(data_orig2, as.data.frame(decoded_data_rand[,-which(names(decoded_data_rand) %in% cont)]))
    decoded_data_rand <- data_orig2  
  } else {
    # This needs some more testing eventually also, if I want to use it. 
    decoded_data_rand <- de.normalize.data(decoded_data_rand, cont, m_vae, M_vae)
  }
  
  decoded_data_rand[,cont] <- round(decoded_data_rand[,cont]) # Round all the numerical variable predictions to the closest integer. 
  
  # We need to change the data types below (for post-processing).
  decoded_data_rand <- decoded_data_rand %>% mutate_if(is.numeric, as.integer)
  decoded_data_rand <- decoded_data_rand %>% mutate_if(is.character, as.factor)
  return(decoded_data_rand[,colnames(adult.data)[-length(colnames(adult.data))]]) # Make sure that the column order is the same as in H and adult.data. 
}

generate_counterfactuals_for_Hs <- function(method, K){
  # Generate K counterfactual per factual in H.
  # We use the functions 'generate_latent_sample' and 'decoded_latent_sample' for this. 
  D_h_per_point <- list()
  for (i in 1:nrow(H)){
    s <- generate_latent_sample(method, K)
    D_h_per_point[[i]] <- decoded_latent_sample(s)
    cat("Generated for point ",i,"\n")
  }
  return(D_h_per_point)
}

D_h_per_point <- generate_counterfactuals_for_Hs(generation_method, K)
filename_generation2 <- paste(CLI.args[1],"_H",CLI.args[2],"_K",CLI.args[3],"_bin",CLI.args[5], sep="") 
save(D_h_per_point, file = paste("results/D_hs/",filename_generation2,".RData",sep="")) # Save the generated D_hs. 

############################### Post-processing (inspired by Experiment 3/4).
# We do the same steps as in those experiments, with some added (because of the VAE instead of the trees. )
post.processing <- function(D_h, H, data){ # 'data' is used to calculate normalization factors for Gower.
  # Remove the rows of D_h (per point) not satisfying the listed criteria. 
  
  # Find the normalization factors for Gower.
  norm.factors <- list()
  for (i in 1:length(colnames(data))){
    colm <- (data %>% dplyr::select(colnames(data)[i]))[[1]]
    if (class(colm) == "integer" || class(colm) == "numeric"){
      q <- quantile(colm, c(0.01, 0.99))
      norm.factors[[i]] <- c(q[1][[1]],q[2][[1]]) # Divide each term in Gower by M_j-m_j, but with 0.99 and 0.01 quantiles respectively!
    } else {
      norm.factors[[i]] <- NA
    }
  }
  
  
  fulfill_crit3_D_h <- function(D_h, c){
    # Build design matrix manually to avoid contrast problems with factors with missing levels (when not generating "enough" data)!!
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
      d_onehot_test <- scale(onehot_test_dat[,cont], center = m_ANN, scale = M_ANN)
      catego <- setdiff(names(onehot_test_dat), cont)
      onehot_test_dat <- cbind(d_onehot_test, onehot_test_dat[,catego])[,colnames(onehot_test_dat)]
    } else {
      # min-max normalization according to mins and maxes from training data. 
      for (j in 1:length(cont)){
        cont_var <- cont[j]
        onehot_test_dat[,cont_var] <- (onehot_test_dat[,cont_var]-m_ANN[j])/(M_ANN[j]-m_ANN[j])
      }
    }
    
    predictions <- as.numeric(ANN %>% predict(data.matrix(onehot_test_dat)))
    D_h_crit3 <- D_h[predictions >= c,] # prediction_model(*) is the R function that predicts 
    # according to the model we want to make explanations for. 
    # We can see that many rows are the same. The duplicates are removed below. 
    unique_D_h <- unique(D_h_crit3)
    return(unique_D_h)
  }
  
  # Fulfill criterion 3. We also fulfill criterion 2 (actionability) by removing the counterfactuals that do not have the correct fixed values. 
  fulfill_crit3 <- function(H, D_h_pp, c){
    for (i in 1:length(D_h_pp)){
      D_h <- D_h_pp[[i]]
      D_h <- fulfill_crit3_D_h(D_h, c) # Since we cannot use Keras to predict on an empty matrix, we satisfy actionability (criterion 2) after predicting. 
      
      h <- H[i,]
      D_h_pp[[i]] <- D_h[interaction(D_h[,fixed_features], drop = T) %in% as.character(interaction(h[,fixed_features])),] # Fulfill criterion 2.
      # This is a very clever solution I found on StackOverflow!
      
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
      #D_h_pp[[i]] <- violation_D_h[H_l[i,], D_h_pp[[i]]] # Calculate violation. 
    }
    return(D_h_pp)
  }
  
  # Remove non-valid counterfactuals (those that don't change the prediction).
  crit3_D_h_per_point <- fulfill_crit3(H, D_h_pp = D_h, c = 0.5) # Fulfill criterion 3 for all (unique) generated possible counterfactuals. 
  
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
    if (nrow(gen) > 1){ # It may happen that several possible counterfactuals have the same sparsity and Gower distance.
      # In such cases we simply choose the first one.
      D_h_pp[[i]] <- gen[1,]
    } else {
      D_h_pp[[i]] <- gen
    }
    
  }
  return(D_h_pp)
}


final_counterfactuals_exp6 <- generate_one_counterfactual_all_points(D_h_post_processed)

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
N_CEs <- rep(NA, length(final_counterfactuals_exp6))
for (i in 1:length(final_counterfactuals_exp6)){
  l <- final_counterfactuals_exp6[[i]]
  n <- nrow(l)
  N_CEs[i] <- n
  if (n >= 0){
    L0s <- c(L0s,l$sparsity)
    L2s <- c(L2s,l$gower)
  } 
}

exp_MCCE <- data.frame("L0" = mean(L0s), "L2" = mean(L2s), "N_CE" = sum(N_CEs))
knitr::kable(exp_MCCE)
write.csv(exp_MCCE, file = paste("resultsVAE/resulting_metrics_", filename_generation2, ".csv", sep = ""))
save(final_counterfactuals_exp6, file = paste("resultsVAE/final_counterfactuals_", filename_generation2, ".RData", sep = ""))

# After generation is done, make latex tables I can paste into report. 
knitr::kable(exp_MCCE, format = "latex", linesep = "", digits = 4, booktabs = T) %>% print()
