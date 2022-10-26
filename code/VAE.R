# We try to implement our first VAE for our adult dataset


rm(list = ls())  # make sure to remove previously loaded variables into the Session.

if (tensorflow::tf$executing_eagerly())
 tensorflow::tf$compat$v1$disable_eager_execution()

setwd("/home/ajo/gitRepos/project")
library(keras)
library(caret)
library(dplyr)
source("code/utilities.R")

# We load the original (non-binarized data).
load("data/adult_data_binarized.RData", verbose = T)
#load("data/adult_data_categ.RData", verbose = T)

# List of continuous variables.
cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
# List of categorical variables (used to reverse onehot encode later!)
categ <- setdiff(names(adult.data), cont)
categ <- categ[-length(categ)] # Remove the label "y"!

adult.data.onehot <- make.data.for.ANN(adult.data, cont)

# Make train and test data.
train_and_test_data <- make.train.and.test(data = adult.data.onehot) # The function returns two matrices (x) and two vectors (y). 
# In addition, it returns two dataframes that are the original dataframe split into train and test (containing y's and x's).
summary(train_and_test_data) # Returned list. 
x_train <- train_and_test_data[[1]]
# We do not really need the output variable y when building a VAE.
x_test <- train_and_test_data[[3]]
train_indices <- train_and_test_data[[5]]

# Normalize the data AFTER splitting to avoid data leakage!
x_train.normalization <- normalize.data(data = x_train, continuous_vars = cont, standardscaler = T) # returns list with data, mins and maxs.
x_train <- x_train.normalization[[1]] # we are only interested in the data for now. 

# Make validation data also, in order to not validate using the test data. 
sample.size.valid <- floor(nrow(x_test) * 1/3)
valid.indices <- sample(1:nrow(x_test), size = sample.size.valid)
x_valid <- x_test[valid.indices, ]
x_test <- x_test[-valid.indices, ]

x_test.normalization <- normalize.data(data = x_test, continuous_vars = cont, standardscaler = T) # returns list with data, mins and maxs.
x_test <- x_test.normalization[[1]] # we are only interested in the data for now. 

x_valid.normalization <- normalize.data(data = x_valid, continuous_vars = cont, standardscaler = T) # returns list with data, mins and maxs.
x_valid <- x_valid.normalization[[1]] # we are only interested in the data for now. 

# Scale the adult data as well, such that we can compare the data sets after generation. 
adult.data.normalization <- normalize.data(data = adult.data.onehot, continuous_vars = cont, standardscaler = T)
adult.data.onehot <- adult.data.normalization[[1]]



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
# Ensure that the standard deviations are positive 
# Not necessary here I think though, because these are the log-vars, which could be both positive and negative. 

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
    kl <- vae_kl_loss(enc_mean, enc_log_var)#*input_size # Trying to scale the KL divergence such that it is of a similar scale as the reconstruction loss.
    return(reconstruction + kl) # + reconstruction
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
  validation_data = list(data.matrix(x_valid), data.matrix(x_valid)),
  verbose = 1,
  callbacks = callbacks_list
)

plot(history)

############################ Make some fake data!
generation_method <- 1
K <- 1e5L
variational_parameters <- encoder %>% predict(data.matrix(x_test)) # Eller bruke treningsdataene her?
v_means <- variational_parameters[[1]]
v_log_vars <- variational_parameters[[2]] 
v_zs <- variational_parameters[[3]]

# Could generate data in two different ways now!
# 1) Use the means and vars from the encoder to sample from MVN, then decode the data. Or simply sample from a standard normal?
# 2) Use the z's: Sample from them, add some small noise to the samples, then decode the data. 

if (generation_method == 1){
  library(MASS)
  mu <- colMeans(v_means)
  sigma <- diag(colMeans((exp(v_log_vars/2))^2))
  #s <- mvrnorm(n = K, mu = mu, Sigma = sigma) # Blir det korrekt å sample fra en multivariat normalfordeling basert på mean av alle encoder means og stds?
  # Jeg tror ikke dette fungerer, fordi mye av variansen forsvinner (se illustrasjon av fordelingen til treningsdataene nedenfor!)
  # Dette går ikke bra, for man mister mye av variasjonen i hver fordeling virker det som!
  s <- mvrnorm(n = K, mu = rep(0,ncol(v_means)), Sigma = diag(rep(1,ncol(v_means)))) # Try to sample from the prior we have put on the latent variables, aka N(0,1).
  # This looks like it gives good results as well! Similar to the one with noise below!
} else if (generation_method == 2){
  r <- slice_sample(as.data.frame(v_zs), n = K, replace = T) # Sample K rows from the z's. 
  s <- jitter(data.matrix(r), amount = 0) # Add some uniform noise to r.
}

if (generation_method == 1){
  # CHECK if sampling is "good"
  scheck <- apply(s, MARGIN = 2, FUN = function(x){(x-mean(x))/sd(x)})
  df <- data.frame(s)
  colnames(df) <- c("Z1", "Z2")
  plt <- tibble(df) %>% 
    ggplot(aes(x = Z1, y = Z2)) +
    geom_point() +
    ggtitle("Sampled from 2d normal") +
    theme_minimal()
  print(plt)
  
  # Check if the data is sampled correctly from the multivariate Gaussian. 
  # The sampling seems to be correct!
  colMeans(v_means) - 3*colMeans((exp(v_log_vars/2)))
  colMeans(v_means) + 3*colMeans((exp(v_log_vars/2)))
  colMeans(v_means) - 2*colMeans((exp(v_log_vars/2)))
  colMeans(v_means) + 2*colMeans((exp(v_log_vars/2)))
  quantile(s[,1],c(0.01, 0.05, 0.95, 0.99))
  quantile(s[,2],c(0.01, 0.05, 0.95, 0.99))
}


# Decode our latent sample s.
decoded_data_rand <- decoder %>% predict(s)
decoded_data_rand <- as.data.frame(decoded_data_rand)
colnames(decoded_data_rand) <- colnames(x_train) 
head(decoded_data_rand)

# De normalize the data according to the test data. 
#decoded_data_rand <- de.normalize.data(decoded_data_rand, cont, x_test.normalization$mins, x_test.normalization$maxs)

# Revert the one hot encoding
adult.data.reverse.onehot <- reverse.onehot.encoding(adult.data.onehot, cont, categ, has.label = T)
decoded_data_rand <- reverse.onehot.encoding(decoded_data_rand, cont, categ, has.label = F)

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
  # Should perhaps use a non-linear dim-reduction algorithm instead here!
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
  
  
  # Plot the first 3 components of PCA.
  library(rgl)
  plot3d(
    x = scores[,1], 
    y = scores[,2],
    z = scores[,3],
    col = as.factor(as.numeric(df$Label)+1),
    xlab = "P1",
    ylab = "P2",
    zlab = "P3"
  )
  rglwidget()
  
  # Use t-SNE to try to reduce the dimension to 2 (for visualization).
  library(Rtsne)
  tsne_out <- Rtsne(latent_train_data, pca = F, perplexity = 30, theta = 0.8)
  plot(tsne_out$Y, col = df$Label, asp = 1)
}

# Noen nøkkeltall (sjekker om generation gir mening). Mest for generation_method 1 kanskje?
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

###################################### Center and scale the datasets "back" to the original scales. 

# De-normalize the normalized (min-max data).
#adult.data.reverse.onehot <- de.normalize.data(adult.data.reverse.onehot, cont, adult.data.normalization$mins, adult.data.normalization$maxs)
#decoded_data_rand <- de.normalize.data(decoded_data_rand, cont, x_test.normalization$mins, x_test.normalization$maxs)

# This can be done easily for the adult data, since this was centered and scaled earlier. 
data_orig <- t(apply(adult.data.reverse.onehot[,cont], 1, function(r)r*adult.data.normalization$sds + adult.data.normalization$means))
data_orig <- cbind(data_orig, as.data.frame(adult.data.reverse.onehot[,-which(names(adult.data) %in% c(cont,"y"))]))
adult.data.reverse.onehot <- data_orig

# There is no clear way of doing it for the generated/decoded data, but we will use the center and the scale of the test-data here. 
data_orig2 <- t(apply(decoded_data_rand[,cont], 1, function(r)r*x_test.normalization$sds + x_test.normalization$means))
data_orig2 <- cbind(data_orig2, as.data.frame(decoded_data_rand[,-which(names(decoded_data_rand) %in% cont)]))
decoded_data_rand <- data_orig2

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

