# VAE_MNIST_2
# 
# Create a function to customize the VAE loss. It is a diferent solution to VAE_MNIST_1


#' This script demonstrates how to build a variational autoencoder with Keras
#' and deconvolution layers.
#' Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114

# Note: This code reflects pre-TF2 idioms.
# For an example of a TF2-style modularized VAE, see e.g.: https://github.com/rstudio/keras/blob/master/vignettes/examples/eager_cvae.R
# Also cf. the tfprobability-style of coding VAEs: https://rstudio.github.io/tfprobability/

# With TF-2, you can still run this code due to the following line:
if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()

K <- keras::backend()

setwd("/home/ajo/gitRepos/project")
library(keras)
library(caret)
source("code/utilities.R")

# We load the original (non-binarized data).
load("data/adult_data_binarized.RData", verbose = T)
#load("data/adult_data_categ.RData", verbose = T)

# List of continuous variables.
cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
# List of categorical variables (used to reverse onehot encode later!)
cat <- setdiff(names(adult.data), cont)
cat <- cat[-length(cat)] # Remove the label "y"!

adult.data.normalized <- normalize.data(data = adult.data, continuous_vars = cont, T) # returns list with data + mins and maxs (or means and sds).
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

#### Data Preparation Done Above ####


#### Parameterization ####

# Input size.
input_size <- ncol(x_train) 

latent_dim <- 2L
intermediate_dim <- 14L
epsilon_std <- 1.0

# training parameters
batch_size <- 64L
epochs <- 10L


#### Model Construction ####

x <- layer_input(shape = c(input_size), name = "enc_input")

hidden <- layer_dense(x, units = intermediate_dim, activation = "relu", name = "enc_hidden")

z_mean <- layer_dense(hidden, units = latent_dim, name = "z_mean")
z_log_var <- layer_dense(hidden, units = latent_dim, name = "z_log_var")

sampling <- function(args) {
  z_mean <- args[, 1:(latent_dim)]
  z_log_var <- args[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]),
    mean = 0,
    stddev = epsilon_std
  )
  z_mean + k_exp(z_log_var/2) * epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling)


decoder_hidden_layer <- layer_dense(units = intermediate_dim, activation = "relu", name = "dec_hidden")
decoder_output_layer <- layer_dense(units = input_size, name = "dec_output") # Or activation = "sigmoid". I believe that a linear activation is more correct however. 


decoder_hidden <- decoder_hidden_layer(z)
decoder_output <- decoder_output_layer(decoder_hidden)

# custom loss function
vae_loss <- function(x, x_decoded) {
  xent_loss <- 1.0 * 
    loss_mean_squared_error(x, x_decoded)
  kl_loss <- -0.5 * k_mean(1 + z_log_var - k_square(z_mean) -
                             k_exp(z_log_var), axis = -1L)
  k_mean(xent_loss + kl_loss)
}

## variational autoencoder
vae <- keras_model(x, decoder_output)
vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
summary(vae)

## encoder: model to project inputs on the latent space
encoder <- keras_model(x, list(z_mean, z_log_var, z))

## build a generator that can sample from the learned distribution
gen_decoder_input <- layer_input(shape = latent_dim)
gen_hidden_decoded <- decoder_hidden_layer(gen_decoder_input)
gen_x_decoded <- decoder_output_layer(gen_hidden_decoded)
generator <- keras_model(gen_decoder_input, gen_x_decoded)



#### Model Fitting ####

vae %>% fit(
  data.matrix(x_train), data.matrix(x_train), 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size, 
  validation_data = list(data.matrix(x_test), data.matrix(x_test))
)


####### Generation of fake data. 
generation_method <- 1
K <- 10000
variational_parameters <- encoder %>% predict(data.matrix(x_test))
v_means <- variational_parameters[[1]]
v_log_vars <- variational_parameters[[2]]
v_zs <- variational_parameters[[3]]

# Could generate data in two different ways now!
# 1) Use the means and vars from the encoder to sample from MVN, then decode the data.
# 2) Use the z's: Sample from them, add some small noise to the samples, then decode the data. 

if (generation_method == 1){
  library(MASS)
  s <- mvrnorm(n = K, mu = colMeans(v_means), Sigma = diag(colMeans((exp(v_log_vars/2))^2))) # Skal denen være ^2 eller ikke?
} else if (generation_method == 2){
  r <- slice_sample(as.data.frame(v_zs), n = K, replace = T) # Sample K rows from the z's. 
  s <- jitter(data.matrix(r)) # Add some uniform noise to r.
}

# Decode our latent sample s.
decoded_data_rand <- generator %>% predict(s)
decoded_data_rand <- as.data.frame(decoded_data_rand)
colnames(decoded_data_rand) <- colnames(x_train)
head(decoded_data_rand)

# Revert the one hot encoding.
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
  tibble(df) %>% 
    ggplot(aes(x = Z1, y = Z2, color = Label)) +
    geom_point() +
    ggtitle("Training data representation from VAE in 2D latent space") +
    theme_minimal()
} else {
  # Use PCA to represent our training data (first two principal components)
  pca <- princomp(latent_train_data)
  print(summary(pca))
  scores <- pca$scores
  df <- data.frame(cbind(scores[,1], scores[,2], train_and_test_data[[2]]))
  df$X3 <- as.factor(df$X3)
  colnames(df) <- c("Z1", "Z2", "Label")
  tibble(df) %>% 
    ggplot(aes(x = Z1, y = Z2, color = Label)) +
    geom_point() +
    ggtitle(paste0("PCA training data representation from VAE in ", latent_dim,"D latent space")) +
    theme_minimal()
}
