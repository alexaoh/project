# We try to implement our first VAE for our adult dataset


rm(list = ls())  # make sure to remove previously loaded variables into the Session.

if (tensorflow::tf$executing_eagerly())
 tensorflow::tf$compat$v1$disable_eager_execution()

setwd("/home/ajo/gitRepos/project")
library(keras)
library(caret)
source("code/utilities.R")

# We load the original (non-binarized data).
#load("data/adult_data_binarized.RData", verbose = T)
#load("data/adult_data_categ.RData", verbose = T)

# Try with Wine data and see if I can get similar results to https://lschmiddey.github.io/fastpages_/2021/03/14/tabular-data-variational-autoencoder.html
adult.data1 <- read.table("wine.data", header = F, sep = ",")
colnames(adult.data1) <- c("Wine", "Alcohol", "Malic.acid","Ash", "Acl", "Mg", "Phenols", "Flavanois", "Nonflavanoid.phenols", "Proanth", "Color-int", "Hue", "OD", "Proline")
data_scaled <- scale(adult.data1)
scale.centers <- attr(data_scaled, "scaled:center")
scale.scales <- attr(data_scaled, "scaled:scale")
#adult.data <- cbind(adult.data1[,1], as.data.frame(data_scaled))
adult.data <- as.data.frame(data_scaled)
colnames(adult.data) <- c("Wine", "Alcohol", "Malic.acid","Ash", "Acl", "Mg", "Phenols", "Flavanois", "Nonflavanoid.phenols", "Proanth", "Color-int", "Hue", "OD", "Proline")
sample.size <- floor(nrow(adult.data) * 0.7)
train.indices <- sample(1:nrow(adult.data), size = sample.size)
x_train <- adult.data[train.indices, ]
x_test <- adult.data[-train.indices, ]
################################### This is just for testing wrt the guide (testing my thought that the Gaussian assumptions are too restrictive for our adult.data!)

# List of continuous variables.
cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
# List of categorical variables (used to reverse onehot encode later!)
cat <- setdiff(names(adult.data), cont)
cat <- cat[-length(cat)] # Remove the label "y"!

adult.data.normalized <- normalize.data(data = adult.data, continuous_vars = cont, F) # returns list with data, mins and maxs.
summary(adult.data.normalized)
adult.data <- adult.data.normalized[[1]] # we are only interested in the data for now. 

adult.data <- make.data.for.ANN(adult.data, cont)

# Make train and test data.
train_and_test_data <- make.train.and.test(data = adult.data) # The function returns two matrices (x) and two vectors (y). 
# In addition, it returns two dataframes that are the original dataframe split into train and test (containing y's and x's).
summary(train_and_test_data) # Returned list. 
x_train <- train_and_test_data[[1]]
# We do not really need the output variable y when building a VAE.
x_test <- train_and_test_data[[3]]
train_indices <- train_and_test_data[[4]]

# Following this guide: https://www.datatechnotes.com/2020/06/how-to-build-variational-autoencoders-in-R.html

latent_dim <- 3L # This sets the size of the latent representation (vector), 
intermediate_dim <- 12L
# defined by the parameters z_mean and z_log_var. 
input_size <- ncol(x_train) 

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
# Use the sigmoid activation to restrict data to (0,1) since we are dealing with normalized input data. 
# Would it be possible to test linear activation function here and normalize the decoded data later instead?
# outputs <- layer_dense(decoder_layer, input_size) # We test this here.  


vae_encoder_output <- decoder_layer(z)
vae_decoder_output <- outputs(vae_encoder_output)
vae <- keras_model(enc_input, vae_decoder_output, name = "VAE")
summary(vae)

vae_loss <- function(enc_mean, enc_log_var){
  # Loss function for our VAE (with Gaussian assumptions).
  vae_reconstruction_loss <- function(y_true, y_predict){
    loss_factor <- 1 # Give weight to the reconstruction in the loss function ("hyperparameter")
    reconstruction_loss <- metric_mean_squared_error(y_true, y_predict) # Or binary cross entropy?
    #reconstruction_loss <- loss_binary_crossentropy(y_true, y_predict)
    #reconstruction_loss <- k_mean(k_square(y_true - y_predict))
    return(reconstruction_loss*loss_factor)
  }
  
  vae_kl_loss <- function(encoder_mu, encoder_log_variance){
    kl <- -1/2*k_sum(1 + encoder_log_variance - k_square(encoder_mu) - k_exp(encoder_log_variance), axis = -1L) # Or axis = -1?
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


vae %>% compile(optimizer = optimizer_adam(learning_rate = 1e-3), loss = vae_loss(enc_mean, enc_log_var))
summary(vae)


# Training/fitting the model.
vae %>% fit(
  x = data.matrix(x_train),
  y = data.matrix(x_train),
  shuffle = T,
  epochs = 200,
  batch_size = 100,
  validation_data = list(data.matrix(x_test), data.matrix(x_test)),
  verbose = 1
)



############################ Make some fake data!
generation_method <- 1
K <- 10000
variational_parameters <- encoder %>% predict(data.matrix(x_test))
v_means <- variational_parameters[[1]]
v_log_vars <- variational_parameters[[2]] # Det virker som om den ikke klarer å lære brede standardavvik, siden dataen ikke er normalfordelt!
# Jeg mistenker at antagelsen om Gaussisk latent fordeling har skyld i dette!
v_zs <- variational_parameters[[3]]

# Could generate data in two different ways now!
# 1) Use the means and vars from the encoder to sample from MVN, then decode the data.
# 2) Use the z's: Sample from them, add some small noise to the samples, then decode the data. 

if (generation_method == 1){
  library(MASS)
  s <- mvrnorm(n = K, mu = colMeans(v_means), Sigma = diag(colMeans((exp(v_log_vars/2))^2))) 
} else if (generation_method == 2){
  r <- slice_sample(as.data.frame(v_zs), n = K, replace = T) # Sample K rows from the z's. 
  s <- jitter(data.matrix(r), amount = 0) # Add some uniform noise to r.
}

# Decode our latent sample s.
decoded_data_rand <- decoder %>% predict(s)
decoded_data_rand <- as.data.frame(decoded_data_rand)
colnames(decoded_data_rand) <- colnames(x_train)
head(decoded_data_rand)

##### Revert scaling for both datasets (when using standardscaler)
####### This was used when testing with the wine data!
#whole_centers <- as.data.frame(transpose(data.table(replicate(scale.centers,n= nrow(data)))))
#colnames(whole_centers) <- c("Alcohol", "Malic.acid","Ash", "Acl", "Mg", "Phenols", "Flavanois", "Nonflavanoid.phenols", "Proanth", "Color-int", "Hue", "OD", "Proline")
#data <- adult.data[,-1]*scale.scales + whole_centers
data_orig <- t(apply(adult.data, 1, function(r)r*scale.scales + scale.centers))
#adult.data <- cbind(adult.data[,1], as.data.frame(data_orig))
adult.data <- as.data.frame(data_orig)
colnames(adult.data) <- c("Wine", "Alcohol", "Malic.acid","Ash", "Acl", "Mg", "Phenols", "Flavanois", "Nonflavanoid.phenols", "Proanth", "Color-int", "Hue", "OD", "Proline")
decoded_data_nonscaled <- t(apply(decoded_data_rand, 1, function(r)r*scale.scales + scale.centers))
#decoded_data_rand <- cbind(decoded_data_rand[,1], as.data.frame(decoded_data_nonscaled))
decoded_data_rand <- as.data.frame(decoded_data_nonscaled)
colnames(decoded_data_rand) <- c("Wine", "Alcohol", "Malic.acid","Ash", "Acl", "Mg", "Phenols", "Flavanois", "Nonflavanoid.phenols", "Proanth", "Color-int", "Hue", "OD", "Proline")

## Sample some from each to see how they look.
slice_sample(adult.data, n = 10)
slice_sample(decoded_data_rand, n = 10)


# Group by wines
for (i in 1:nrow(decoded_data_rand)){
  if (decoded_data_rand[i,1] <= 1){
    decoded_data_rand[i,1] <- 1  
  } else if (decoded_data_rand[i,1] <= 2){
    decoded_data_rand[i,1] <- 2
  } else {
    decoded_data_rand[i,1] <- 3
  }
  
}

summary(adult.data)
summary(decoded_data_rand)

colMeans(decoded_data_rand %>% filter(Wine == 1))
colMeans(decoded_data_rand %>% filter(Wine == 2))
colMeans(decoded_data_rand %>% filter(Wine == 3))


# Revert the one hot encoding
adult.data.reverse.onehot <- reverse.onehot.encoding(adult.data, cont, cat, has.label = T)
decoded_data_rand <- reverse.onehot.encoding(decoded_data_rand, cont, cat, has.label = F)

summary(adult.data.reverse.onehot)
summary(decoded_data_rand)

summary(adult.data.reverse.onehot[, cont])
summary(decoded_data_rand[, cont])

cap_loss_real <- (adult.data %>% dplyr::select(capital_loss))[[1]]
cap_loss_gen <- (decoded_data_rand %>% dplyr::select(capital_loss))[[1]]
length(cap_loss_real[cap_loss_real != 0]) # Same as for cap_gain!
length(cap_loss_real)
length(cap_loss_gen[cap_loss_gen != 0]) # Same as for cap_gain!
length(cap_loss_gen) # Almost all data points are != 0 from VAE!

table(adult.data.reverse.onehot$workclass)/sum(table(adult.data.reverse.onehot$workclass))
table(decoded_data_rand$workclass)/sum(table(decoded_data_rand$workclass))

table(adult.data.reverse.onehot$marital_status)/sum(table(adult.data.reverse.onehot$marital_status))
table(decoded_data_rand$marital_status)/sum(table(decoded_data_rand$marital_status))

table(adult.data.reverse.onehot$occupation)/sum(table(adult.data.reverse.onehot$occupation))
table(decoded_data_rand$occupation)/sum(table(decoded_data_rand$occupation))

table(adult.data.reverse.onehot$relationship)/sum(table(adult.data.reverse.onehot$relationship))
table(decoded_data_rand$relationship)/sum(table(decoded_data_rand$relationship))

table(adult.data.reverse.onehot$race)/sum(table(adult.data.reverse.onehot$race))
table(decoded_data_rand$race)/sum(table(decoded_data_rand$race))

table(adult.data.reverse.onehot$sex)/sum(table(adult.data.reverse.onehot$sex))
table(decoded_data_rand$sex)/sum(table(decoded_data_rand$sex))

table(adult.data.reverse.onehot$native_country)/sum(table(adult.data.reverse.onehot$native_country))
table(decoded_data_rand$native_country)/sum(table(decoded_data_rand$native_country))



#### For illustration purposes we make a scatter plot of the latent space (when 2 dimensional).
library(ggplot2)
library(dplyr)
latent_train_data <- (encoder %>% predict(data.matrix(x_train)))[[3]] # We only need the latent values for this. 
if (latent_dim == 2){
  df <- data.frame(cbind(latent_train_data, train_and_test_data[[2]]))
  df$X3 <- as.factor(df$X3)
  colnames(df) <- c("Z1", "Z2", "Label")
  plt <- tibble(df) %>% 
    ggplot(aes(x = Z1, y = Z2, color = Label)) +
    geom_point() +
    ggtitle("Training data representation from VAE in 2D latent space") +
    theme_minimal()
  print(plt)
} else {
  # Use PCA to represent our training data (first two principal components)
  pca <- princomp(latent_train_data)
  print(summary(pca))
  scores <- pca$scores
  df <- data.frame(cbind(scores[,1], scores[,2], train_and_test_data[[2]]))
  df$X3 <- as.factor(df$X3)
  colnames(df) <- c("Z1", "Z2", "Label")
  plt <- tibble(df) %>% 
    ggplot(aes(x = Z1, y = Z2, color = Label)) +
    geom_point() +
    ggtitle(paste0("PCA training data representation from VAE in ", latent_dim,"D latent space")) +
    theme_minimal()
  print(plt)
}

# Noen nøkkeltall. 
mean(df$Z1)
mean(df$Z2)
sd(df$Z1)
sd(df$Z2)
mean(s[,1])
mean(s[,2])
sd(s[,1])
sd(s[,2])
mean((df %>% filter(Label == 1) %>% dplyr::select(Z1))$Z1)
mean((df %>% filter(Label == 1) %>% dplyr::select(Z2))$Z2)
mean((df %>% filter(Label == 0) %>% dplyr::select(Z1))$Z1)
mean((df %>% filter(Label == 0) %>% dplyr::select(Z2))$Z2)
sd((df %>% filter(Label == 1) %>% dplyr::select(Z1))$Z1)
sd((df %>% filter(Label == 1) %>% dplyr::select(Z2))$Z2)
sd((df %>% filter(Label == 0) %>% dplyr::select(Z1))$Z1)
sd((df %>% filter(Label == 0) %>% dplyr::select(Z2))$Z2)
