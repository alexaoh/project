# We try to implement our first VAE for our adult dataset


rm(list = ls())  # make sure to remove previously loaded variables into the Session.
setwd("/home/ajo/gitRepos/project")
library(keras)
source("code/utilities.R")

# We load the original (non-binarized data).
load("data/adult_data_categ.RData", verbose = T)

# List of continuous variables.
cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")

adult.data.normalized <- normalize.data(data = adult.data, continuous_vars = cont) # returns list with data, mins and maxs.
summary(adult.data.normalized)
adult.data <- adult.data.normalized[[1]] # we are only interested in the data for now. 

# Probably need to make a design matrix here as well! Dette blir rart når jeg skal encode og deretter decode + sammenligne med originaldataen!
# Hør med Kjersti om jeg bør lage en design-matrise eller ikke!?
adult.data <- as.data.frame(model.matrix(~.,data = adult.data))

# Make train and test data.
train_and_test_data <- make.train.and.test(data = adult.data) # The function returns two matrices (x) and two vectors (y). 
# In addition, it returns two dataframes that are the original dataframe split into train and test (containing y's and x's).
summary(train_and_test_data) # Returned list. 
x_train <- train_and_test_data[[1]]
# We do not really need the output variable y when building a VAE.
x_test <- train_and_test_data[[3]]
# These two are used when I want to make dataframes later, in order to easier keep all correct datatypes in the columns. 
train <- train_and_test_data[[5]]
test <- train_and_test_data[[6]]

# Following this guide: https://www.datatechnotes.com/2020/06/how-to-build-variational-autoencoders-in-R.html

latent_size <- 10 # Not sure how to select this parameter yet. This sets the size of the latent representation (vector), 
# defined by the parameters z_mean and z_log_var. 
input_size <- dim(x_train)[2] # The same as ncol(x_train).

####### First we define the encoder. 
enc_input <- layer_input(shape = c(input_size))
layer_one <- layer_dense(enc_input, units=256, activation = "relu")
z_mean <- layer_dense(layer_one, latent_size)
z_log_var <- layer_dense(layer_one, latent_size)

encoder <- keras_model(enc_input, z_mean)
summary(encoder)

# Sampling function in the latent space. It samples the Gaussian distribution by using the mean and variance
# that will be learned. It returns a sampled latent vector. 
sampling <- function(arg){
  # Here we clearly assume a Gaussian distribution via the method of sampling from the assumed latent distribution as shown below. 
  z_mean <- arg[,1:(latent_size)]
  z_log_var <- arg[,(latent_size + 1):(2 * latent_size)]
  # Should ,latent_size be added in the shape-vector of the k_random_normal below?
  epsilon <- k_random_normal(shape = c(k_shape(z_mean)[[1]]), mean=0, stddev = 1) # Draws samples from standard normal, adds element-wise to a vector. 
  z_mean + k_exp(z_log_var/2)*epsilon # element-wise exponential, in order to transform the standard normal samples to 
  # sampling from our latent variable distribution (the reparametrization trick).
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
  layer_lambda(sampling)



from.DNN.course <- function(){
  # I think this should be equivalent to the code for these two components from directly above but this needs to be tested. 
  # This is taken from VAE_MNIST_1.R from DNN course at UPC. 
  
  # I think this way of defining the function and making z makes more sense!
  sampling <- function(args) {
    c(z_mean, z_log_var) %<-% args
    epsilon <- k_random_normal(shape = list(k_shape(z_mean)[1], latent_size),
                               mean = 0, stddev = 1)
    z_mean + k_exp(z_log_var) * epsilon # Not sure if there should be a 1/2 here as above (check theoretical equations!)?
  }
  
  # Point randomly sampled
  z <- list(z_mean, z_log_var) %>% 
    layer_lambda(sampling)
}

####### Defining the decoder. Det gir ikke mening for meg her nede! Tror det er noe rart med dimensjonen på dataen i hele modellen fra begynnelsen av!
#decoder_input <- layer_input(k_int_shape(z)[-1])
#decoder_layer <- layer_dense(decoder_input, units = 256, activation = "relu") # I have changed the code from the guide a bit here, since it did not make sense to me. 
decoder_layer <- layer_dense(units = 256, activation = "relu")
decoder_mean <- layer_dense(units = input_size, activation = "sigmoid")
h_decoded <- decoder_layer(z)
x_decoded_mean <- decoder_mean(h_decoded)

vae <- keras_model(enc_input, x_decoded_mean)
summary(vae)


vae_loss <- function(input, x_decoded_mean){
  xent_loss=(input_size/1.0)*loss_binary_crossentropy(input, x_decoded_mean)
  kl_loss=-0.5*k_mean(1+z_log_var-k_square(z_mean)-k_exp(z_log_var), axis=-1)
  xent_loss + kl_loss
}

vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
summary(vae)


### Generator — not sure how this fits in to my model yet. 
dec_input <- layer_input(shape = latent_size)
h_decoded_2 <- decoder_layer(dec_input)
x_decoded_mean_2 <- decoder_mean(h_decoded_2)
generator <- keras_model(dec_input, x_decoded_mean_2)
summary(generator)

# Training/fitting the model.
vae %>% fit(
  x = data.matrix(x_train), 
  y = data.matrix(x_train), 
  shuffle = TRUE, 
  epochs = 20, 
  batch_size = 64, 
  validation_data = list(data.matrix(x_test), data.matrix(x_test))
)

# Guiden sier at vi kan bruke denne (nedenfor) for å sette en stopper for en error. Aner ikke hva som er greia med det!?
# Veldig merkelig! Fungerer ikke på den måten uansett!
#tensorflow::tf$compat$v1$disable_eager_execution()

# Tester å genere noe data nedenfor:
n = 10
test =  x_test[0:n,]
x_test_encoded <- predict(encoder, data.matrix(test))

decoded_data = generator %>% predict(x_test_encoded) # Hvorfor bruke "generator" her og ikke "decoderen" fra over?

# De er relativt like!! Ganske kult!!
head(test[,cont])
decoded_data <- as.data.frame(decoded_data)
colnames(decoded_data) <- colnames(test)
head(decoded_data[,cont])

head(test[,1:6])
head(decoded_data[,1:6])
