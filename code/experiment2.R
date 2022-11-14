# Experiment 2: Make generative model for our data set and sample synthetic data. 
# Our goal is for the synthetic (generated) sample to be very similar to our input data. 

rm(list = ls())  # make sure to remove previously loaded variables into the Session.

if (tensorflow::tf$executing_eagerly())
 tensorflow::tf$compat$v1$disable_eager_execution()

setwd("/home/ajo/gitRepos/project")
library(keras)
library(caret)
library(dplyr)
library(MASS)
library(kableExtra)
source("code/utilities.R")

set.seed(1234)

# Parameter for standardscaler (centering and scaling) or not (normalization, i.e. min-max scaling).
standardscaler <- T

# We load the data (binarized or categorical).
#load("data/adult_data_binarized.RData", verbose = T)
load("data/adult_data_categ.RData", verbose = T)

# List of continuous variables.
cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
# List of categorical variables (used to reverse onehot encode later!)
categ <- setdiff(names(adult.data), cont)
categ <- categ[-length(categ)] # Remove the label "y"!

############### Do correct post-processing of the data. We do not need the "usual" labels y when building the VAE. 
adult.data.onehot <- data.frame(adult.data) # make a copy of the dataframe for one hot encoding in ANN.
tracemem(adult.data) == tracemem(adult.data.onehot) # it is a deep copy.
data.table::address(adult.data)
data.table::address(adult.data.onehot)
# The memory addresses are different. 

# Make the design matrix for the DNN.
adult.data.onehot <- make.data.for.ANN(adult.data.onehot, cont, label = T) 

# Make train and test data for our model matrix adult.data.
sample.size <- floor(nrow(adult.data.onehot) * 6/7)
train.indices <- sample(1:nrow(adult.data.onehot), size = sample.size)
train <- adult.data.onehot[train.indices, ]
test <- adult.data.onehot[-train.indices, ]

# Scale training data. 
train.normalization <- normalize.data(data = train, continuous_vars = cont, standardscaler = standardscaler) # returns list with data, mins and maxs.
train <- train.normalization[[1]]
m <- train.normalization[[2]]
M <- train.normalization[[3]]

x_train <- train[,-which(names(train) == "y")]
y_train <- train[, "y"] # Use this for visualization later only.

# Make validation data also. THIS IS NOT NECESSARY WHEN FOR THE VAE, SINCE I WANT TO TRAIN IT ON ALL THE DATA. 
# In cases where validation data is dropped I simply use the testing data as validation data when fitting the VAE.
#sample.size.valid <- floor(nrow(test) * 1/3)
#valid.indices <- sample(1:nrow(test), size = sample.size.valid)
#valid <- test[valid.indices, ]
#test <- test[-valid.indices, ]

# Scaling according to the same values abtained when scaling the training data! This is very important in all applications for generalizability!!
if (standardscaler){
  # Centering and scaling according to scales and centers from training data. 
  d_test <- scale(test[,cont], center = m, scale = M)
  catego <- setdiff(names(test), cont)
  test <- cbind(d_test, test[,catego])[,colnames(test)]
  
  #d_valid <- scale(valid[,cont], center = m, scale = M)
  #catego <- setdiff(names(valid), cont)
  #valid <- cbind(d_valid, valid[,catego])[,colnames(valid)]
} else {
  # min-max normalization according to mins and maxes from training data. 
  for (j in 1:length(cont)){
    cont_var <- cont[j]
    test[,cont_var] <- (test[,cont_var]-m[j])/(M[j]-m[j])
    #valid[,cont_var] <- (valid[,cont_var]-m[j])/(M[j]-m[j])
  }
}

x_test <- test[,-which(names(test) == "y")]
#y_test <- test[,"y"]

#x_valid <- valid[,-which(names(valid) == "y")]
#y_valid <- valid[,"y"]

# Scale the adult data as well, such that we can compare the data sets after generation. 
#adult.data.normalization <- normalize.data(data = adult.data.onehot, continuous_vars = cont, standardscaler = T)
#adult.data.onehot <- adult.data.normalization[[1]]


############################ Build the VAE. 
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
    loss_factor <- 1 # Give weight to the reconstruction in the loss function ("hyperparameter")
    #reconstruction_loss <- loss_mean_squared_error(y_true, y_predict) 
    #reconstruction_loss <- loss_binary_crossentropy(y_true, y_predict) # Or binary cross entropy?
    #reconstruction_loss <- k_mean(k_square(y_true - y_predict))
    reconstruction_loss <- k_sum(k_square(y_true - y_predict))
    # Tror jeg antar Gaussian output når jeg bruker mean squarederror som reconstruction loss!!!!!
    # https://stats.stackexchange.com/questions/464875/mean-square-error-as-reconstruction-loss-in-vae
    
    # Prøver å lage en egen loss function!
    # if (class(y_true) == "factor"){
    #   reconstruction_loss <- metric_categorical_crossentropy(y_true, y_predict)
    # } else {
    #   reconstruction_loss <- k_mean(k_square(y_true - y_predict))
    # }
        
    # Siden y_true er vektor, bruk heller ifelse:
    # Klarer ikke å finne dtype til den OPPRINNELIGE DATAEN PÅ DENNE MÅTEN!
    # reconstruction_loss <- k_switch(k_dtype(y_true), # class(y_true) == "factor" # Fungerer ikke. 
    #                               k_mean(k_square(y_true - y_predict)),
    #                               metric_categorical_crossentropy(y_true, y_predict))
    # Her må man finne ut hvordan man vil vekte faktorene vs de numeriske kovariatene!
    # Hvilken blir størst i loss-funksjonen totalt sett? ANER IKKE!
    
    #reconstruction_loss <- k_mean(k_square(y_true - y_predict)) + metric_categorical_crossentropy(y_true, y_predict)
    
    # https://datascience.stackexchange.com/questions/28440/custom-conditional-loss-function-in-keras
    
    # reconstruction_loss <- function(y_true, y_predict){
    #   l <-
    # }
    
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

############################ Make some synthetic data!
generation_method <- 4
K <- 1e5L
#K <- 3*nrow(x_train)

# Could generate data in four different ways now. Test all of them below and add results to the report. 
# Tried to match these numbers to the numbers used in the project report.
# Here we generate samples in the latent space according to the given method. 
# Then we decode the latent space samples below. 
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


# Decode our latent sample s.
decoded_data_rand <- decoder %>% predict(s)
decoded_data_rand <- as.data.frame(decoded_data_rand)
colnames(decoded_data_rand) <- colnames(x_train) 
head(decoded_data_rand)

# De normalize the data according to the test data. 
#decoded_data_rand <- de.normalize.data(decoded_data_rand, cont, x_test.normalization$mins, x_test.normalization$maxs)

# Revert the one hot encoding
decoded_data_rand <- reverse.onehot.encoding(decoded_data_rand, cont, categ, has.label = F)

change_names <- function(decoded_data_rand){
  # Change the names of the categorical values in the decoded data. 
  decoded_data_rand$workclass[decoded_data_rand$workclass == "workclass..Private"] <- " Private"
  decoded_data_rand$workclass[decoded_data_rand$workclass == "workclass..Other"] <- " Other"
  decoded_data_rand$workclass[decoded_data_rand$workclass == "workclass..Local.gov"] <- " Local-gov"
  decoded_data_rand$workclass[decoded_data_rand$workclass == "workclass..Self.emp.inc"] <- " Self-emp-inc"
  decoded_data_rand$workclass[decoded_data_rand$workclass == "workclass..Self.emp.not.inc"] <- " Self-emp-not-inc"
  decoded_data_rand$workclass[decoded_data_rand$workclass == "workclass..State.gov"] <- " State-gov"
  decoded_data_rand$workclass[decoded_data_rand$workclass == "workclass..Without.pay"] <- " Without-pay"
  decoded_data_rand$marital_status[decoded_data_rand$marital_status == "marital_status..Other"] <- " Other"
  decoded_data_rand$marital_status[decoded_data_rand$marital_status == "marital_status..Married.civ.spouse"] <- " Married-civ-spouse"
  decoded_data_rand$marital_status[decoded_data_rand$marital_status == "marital_status..Divorced"] <- " Divorced"
  decoded_data_rand$marital_status[decoded_data_rand$marital_status == "marital_status..Never.married"] <- " Never-married"
  decoded_data_rand$marital_status[decoded_data_rand$marital_status == "marital_status..Widowed"] <- " Widowed"
  decoded_data_rand$marital_status[decoded_data_rand$marital_status == "marital_status..Separated"] <- " Separated"
  decoded_data_rand$occupation[decoded_data_rand$occupation == "occupation..Other"] <- " Other"
  decoded_data_rand$occupation[decoded_data_rand$occupation == "occupation..Craft.repair"] <- " Craft-repair"
  decoded_data_rand$occupation[decoded_data_rand$occupation == "occupation..Adm.clerical"] <- " Adm-clerical"
  decoded_data_rand$occupation[decoded_data_rand$occupation == "occupation..Exec.managerial"] <- " Exec-managerial"
  decoded_data_rand$occupation[decoded_data_rand$occupation == "occupation..Farming.fishing"] <- " Farming-fishing"
  decoded_data_rand$occupation[decoded_data_rand$occupation == "occupation..Machine.op.inspct"] <- " Machine-op-inspct"
  decoded_data_rand$occupation[decoded_data_rand$occupation == "occupation..Transport.moving"] <- " Transport-moving"
  decoded_data_rand$occupation[decoded_data_rand$occupation == "occupation..Other.service"] <- " Other-service"
  decoded_data_rand$occupation[decoded_data_rand$occupation == "occupation..Prof.specialty"] <- " Prof-specialty"
  decoded_data_rand$occupation[decoded_data_rand$occupation == "occupation..Sales"] <- " Sales"
  decoded_data_rand$relationship[decoded_data_rand$relationship == "relationship..Husband"] <- " Husband"
  decoded_data_rand$relationship[decoded_data_rand$relationship == "relationship..Other"] <- " Other"
  decoded_data_rand$relationship[decoded_data_rand$relationship == "relationship..Not.in.family"] <- " Not-in-family"
  decoded_data_rand$relationship[decoded_data_rand$relationship == "relationship..Own.child"] <- " Own-child"
  decoded_data_rand$relationship[decoded_data_rand$relationship == "relationship..Unmarried"] <- " Unmarried"
  decoded_data_rand$relationship[decoded_data_rand$relationship == "relationship..Wife"] <- " Wife"
  decoded_data_rand$race[decoded_data_rand$race == "race..Other"] <- " Other"
  decoded_data_rand$race[decoded_data_rand$race == "race..White"] <- " White"
  decoded_data_rand$race[decoded_data_rand$race == "race..Black"] <- " Black"
  decoded_data_rand$race[decoded_data_rand$race == "race..Amer.Indian.Eskimo"] <- " Amer-Indian-Eskimo"
  decoded_data_rand$race[decoded_data_rand$race == "race..Asian.Pac.Islander"] <- " Asian-Pac-Islander"
  decoded_data_rand$native_country[decoded_data_rand$native_country == "native_country..Other"] <- " Other"
  decoded_data_rand$native_country[decoded_data_rand$native_country == "native_country..United.States"] <- " United-States"
  decoded_data_rand$native_country[decoded_data_rand$native_country == "native_country..Mexico"] <- " Mexico"
  decoded_data_rand$sex[decoded_data_rand$sex == "sex..Male"] <- " Male"
  decoded_data_rand$sex[decoded_data_rand$sex == "sex..Female"] <- " Female"
  return(decoded_data_rand)
}

decoded_data_rand <- change_names(decoded_data_rand)

# De-normalize the generated data, such that it is on the same scale as the adult data. 
if (standardscaler){
  data_orig2 <- t(apply(decoded_data_rand[,cont], 1, function(r)r*M + m))
  data_orig2 <- cbind(data_orig2, as.data.frame(decoded_data_rand[,-which(names(decoded_data_rand) %in% cont)]))
  decoded_data_rand <- data_orig2  
} else {
  # Har ikke testet denne på lenge! OBS!
  decoded_data_rand <- de.normalize.data(decoded_data_rand, cont, m, M)
}


summary(adult.data)
summary(decoded_data_rand)

# Round these — check later if this is correct but it seems correct to me!
decoded_data_rand[,cont] <- round(decoded_data_rand[,cont]) # Round all the numerical variable predictions to the closest integer. 
head(decoded_data_rand)

summary(adult.data[, cont])
summary(decoded_data_rand[, cont])

cap_gain_real <- (adult.data %>% dplyr::select(capital_gain))[[1]]
cap_gain_gen <- (decoded_data_rand %>% dplyr::select(capital_gain))[[1]]
length(cap_gain_real[cap_gain_real != 0])/length(cap_gain_real)
length(cap_gain_gen[cap_gain_gen != 0])/length(cap_gain_gen) # Almost all data points are != 0 from VAE!

cap_loss_real <- (adult.data %>% dplyr::select(capital_loss))[[1]]
cap_loss_gen <- (decoded_data_rand %>% dplyr::select(capital_loss))[[1]]
length(cap_loss_real[cap_loss_real != 0])/length(cap_loss_real)
length(cap_loss_gen[cap_loss_gen != 0])/length(cap_loss_gen) # Almost all data points are != 0 from VAE!


table(adult.data$workclass)/sum(table(adult.data$workclass))
table(decoded_data_rand$workclass)/sum(table(decoded_data_rand$workclass))

table(adult.data$marital_status)/sum(table(adult.data$marital_status))
table(decoded_data_rand$marital_status)/sum(table(decoded_data_rand$marital_status))

table(adult.data$occupation)/sum(table(adult.data$occupation))
table(decoded_data_rand$occupation)/sum(table(decoded_data_rand$occupation))

table(adult.data$relationship)/sum(table(adult.data$relationship))
table(decoded_data_rand$relationship)/sum(table(decoded_data_rand$relationship))

table(adult.data$race)/sum(table(adult.data$race))
table(decoded_data_rand$race)/sum(table(decoded_data_rand$race))

table(adult.data$sex)/sum(table(adult.data$sex))
table(decoded_data_rand$sex)/sum(table(decoded_data_rand$sex))

table(adult.data$native_country)/sum(table(adult.data$native_country))
table(decoded_data_rand$native_country)/sum(table(decoded_data_rand$native_country))

# Could do things like this also! (compare tables of factors to each other)
table(adult.data[,c("sex")], adult.data[,c("native_country")])/sum(table(adult.data[,c("sex")], adult.data[,c("native_country")]))
table(decoded_data_rand[,c("sex")], decoded_data_rand[,c("native_country")])/sum(table(decoded_data_rand[,c("sex")], decoded_data_rand[,c("native_country")]))

cont.summary <- function(data){
  summary <- data %>%
    dplyr::select(c("sex",cont)) %>%
    tidyr::pivot_longer(-sex) %>%
    group_by(` ` = name) %>% 
    summarize(Min. = min(value),
              "25%" = quantile(value, p = 0.25),
              Median = median(value), 
              Mean = mean(value), 
              "75%" = quantile(value, p = 0.75),
              Max. = max(value))
  summary
}

kbl(cont.summary(adult.data), format = "latex", linesep = "", digits = 1, booktabs = T) %>% 
  kable_styling(latex_options = c("scale_down")) %>% 
  column_spec(1, monospace = T) %>% 
  print()

kbl(cont.summary(decoded_data_rand), format = "latex", linesep = "", digits = 1, booktabs = T) %>% 
  kable_styling(latex_options = c("scale_down")) %>% 
  column_spec(1, monospace = T) %>% 
  print()

make_ggplot_for_categ <- function(data, filename, save){
  data.categ <- data[,categ]
  data.categ.wide <- data.categ %>% tidyr::pivot_longer(categ) %>% count(name, value) %>% mutate(ratio = round(n/nrow(data.categ), 3))
  #adult.data.categ.wide <- adult.data.categ %>% tidyr::pivot_longer(categ)
  #adult.data.categ <- apply(adult.data.categ,FUN = function(x){table(x)/sum(table(x))}, MARGIN = 2)
  categ_plot <- data.categ.wide %>% ggplot(aes(x = name, y = ratio, fill = value)) +
    geom_col(position = "stack")+#), show.legend = F) + # For kategorisk data må legend fjernes
    geom_text(aes(label = ratio), position = position_stack(vjust = 0.5)) +
    theme_minimal() 
  #geom_text(nudge_y = 1)
  if (save) ggsave(paste0("plots/",filename,".pdf"), width = 9, height = 5)
  return(categ_plot)
}

# Vanskeligere å lage disse plottene for den kategoriske dataen!!
make_ggplot_for_categ(adult.data, "ikkenoeenda", F)
make_ggplot_for_categ(decoded_data_rand[,colnames(adult.data)[-14]], "generated_exp2_categ_ratios_bin_data_method4", F) # Legg inn dette senere!


#### For illustration purposes we make a scatter plot of the latent space (when 2 dimensional).
library(ggplot2)
library(dplyr)
latent_train_data <- (encoder %>% predict(data.matrix(x_train)))[[3]] # We only need the latent values for this. 
if (latent_dim == 2){
  df <- data.frame(cbind(latent_train_data, y_train))
  df$y_train<- as.factor(df$y_train)
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
  df <- data.frame(cbind(scores[,1], scores[,2], y_train))
  df$y_train<- as.factor(df$y_train)
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


###################################### Try "deployment phase" in Olsen 2021 (page 10 + 11).
# Dette skal tilsvare min generation_method = 2 oppe!! (det var tanken). Har brukt denne tankegangen i generation_method = 2 oppe!
K <- 1
variational_parameters <- encoder %>% predict(data.matrix(x_train[1,])) # Eller bruke treningsdataene her?
v_means <- variational_parameters[[1]]
v_log_vars <- variational_parameters[[2]] 

sample.after <- function(K, v_means, v_log_vars){
  # Sample from one test point. 
  mu <- as.numeric(v_means)
  sigma <- diag(as.numeric((exp(v_log_vars/2))^2))
  mvrnorm(n = K, mu = mu, Sigma = sigma)
}
z_test <- sample.after(K, v_means, v_log_vars)
dec_test <- decoder %>% predict(matrix(z_test, nrow = 1))
dec_test <- as.data.frame(dec_test)
colnames(dec_test) <- colnames(x_test)

data_orig2 <- t(apply(dec_test[,cont], 1, function(r)r*M + m))
data_orig2 <- cbind(data_orig2, as.data.frame(dec_test[,-which(names(dec_test) %in% cont)]))
decoded_data_rand <- data_orig2  
decoded_data_rand <- reverse.onehot.encoding(decoded_data_rand, cont, categ, has.label = F)

# De normalize the data according to the test data. 
#decoded_data_rand <- de.normalize.data(decoded_data_rand, cont, x_test.normalization$mins, x_test.normalization$maxs)

# Revert the one hot encoding
#adult.data.reverse.onehot <- reverse.onehot.encoding(adult.data.onehot, cont, categ, has.label = T)
decoded_data_rand <- reverse.onehot.encoding(decoded_data_rand, cont, categ, has.label = F)
