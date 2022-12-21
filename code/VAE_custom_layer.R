# Did not use this in the end. Could be an interesing alternative to the implementation I used though. 
# Implement a VAE with a custom layer instead of custom loss function. 


rm(list = ls())  # make sure to remove previously loaded variables into the Session.

if (tensorflow::tf$executing_eagerly())
 tensorflow::tf$compat$v1$disable_eager_execution()

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

# Wrong way to normalize now. Fix this. 
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

latent_dim <- 10 # This sets the size of the latent representation (vector), 
intermediate_dim <- 18
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

encoder <- keras_model(enc_input, list(enc_mean, enc_log_var), name = "encoder_model") # c(z_mean, z_log_var, z) 
summary(encoder)

latent_inputs <- layer_input(shape = c(latent_dim), name = "dec_input")
decoder_layer <- layer_dense(latent_inputs, units = intermediate_dim, activation = "relu", name = "dec_hidden")
outputs <- layer_dense(decoder_layer, input_size, name = "dec_output") # activation = "sigmoid",

decoder <- keras_model(latent_inputs, outputs, name = "decoder_model")
summary(decoder)


# Instead of using a custom loss function, we make a custom layer, where the loss function is added!
library(R6)

CustomVariationalLayer <- R6Class("CustomVariationalLayer",

                                  inherit = KerasLayer,

                                  public = list(

                                    vae_loss = function(x_true, x_pred) {
                                      xent_loss <- metric_mean_squared_error(x_true, x_pred)
                                      kl_loss <- -0.5 * k_mean(
                                        1 + enc_log_var - k_square(enc_mean) - k_exp(enc_log_var),
                                        axis = -1L
                                      )
                                      k_mean(1*xent_loss + kl_loss)
                                    },

                                    call = function(inputs, mask = NULL) {
                                      x <- inputs[[1]]
                                      x_pred <- inputs[[2]]
                                      loss <- self$vae_loss(x, x_pred)
                                      self$add_loss(loss, inputs = inputs)
                                      x
                                    }
                                  )
)

layer_variational <- function(object) {
  create_layer(CustomVariationalLayer, object, list())
}

x_pred <- decoder(z)
# Call the custom layer on the input and the decoded output to obtain
# the final model output.
y <- list(enc_input, x_pred) %>%
  layer_variational()

vae <- keras_model(enc_input, y)
summary(vae)

vae %>% compile(
  optimizer = optimizer_adam(),
  loss = NULL
)

vae %>% fit(
  x = data.matrix(x_train), y = NULL,
  epochs = 10,
  batch_size = 64,
  validation_data = list(data.matrix(x_test), NULL)
) 

# Manual tests for generating data. 
n = 3
test =  x_test[0:n,]
x_test_encoded <- predict(encoder, data.matrix(test))

decoded_data = decoder %>% predict(x_test_encoded) 

head(test[,cont])
decoded_data <- as.data.frame(decoded_data)
colnames(decoded_data) <- colnames(test)
head(decoded_data[,cont])

head(test[,1:6])
head(decoded_data[,1:6])

adult.data.reverse.onehot <- reverse.onehot.encoding(adult.data, cont, cat, has.label = T)
decoded_data_rand <- reverse.onehot.encoding(decoded_data_rand, cont, cat, has.label = F)

summary(decoded_data_rand$age)
summary(adult.data$age)

summary(decoded_data_rand$fnlwgt)
boxplot(decoded_data_rand$fnlwgt)
summary(adult.data$fnlwgt)
boxplot(adult.data$fnlwgt)

summary(decoded_data_rand[,cont])
summary(adult.data[,cont])

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


########################### We try to feed the autoencoder entirely with test data in order to generate new points.
reconstructed_data <- vae %>% predict(data.matrix(x_test)) 
reconstructed_data <- as.data.frame(reconstructed_data)
colnames(reconstructed_data) <- colnames(x_test)

adult.data.reverse.onehot <- reverse.onehot.encoding(adult.data, cont, cat, has.label = T)
decoded_data_rand <- reverse.onehot.encoding(reconstructed_data, cont, cat, has.label = F)
x_test.reverse.onehot <- reverse.onehot.encoding(x_test, cont, cat, has.label = F)

summary(x_test.reverse.onehot)
summary(decoded_data_rand)

##################################### Another try: generate synthetic data. 
# Sample from a Gaussian with mu = mean(encoder_mu) og sigma = mean(encoder_sigma).
number_samples <- 100000
variational_parameters <- encoder %>% predict(data.matrix(x_test))
variational_means <- variational_parameters[[1]]
variational_sigmas <- exp(variational_parameters[[2]]/2)
avg_variational_means <- colMeans(variational_means)
avg_variational_sigmas <- colMeans(variational_sigmas)
library(MASS)
s <- mvrnorm(n = number_samples, mu = avg_variational_means, Sigma = diag(avg_variational_sigmas^2) ) 
# We then decode s.
decoded_data_rand <- decoder %>% predict(s)

decoded_data_rand <- as.data.frame(decoded_data_rand)
colnames(decoded_data_rand) <- colnames(x_train)
head(decoded_data_rand)

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
