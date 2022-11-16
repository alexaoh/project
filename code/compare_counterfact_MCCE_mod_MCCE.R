# We print some tables of counterfactuals for the same h \in H
# generated with MCCE and Modified MCCE respectively.
library(kableExtra)
setwd("/home/ajo/gitRepos/project")

# First we compare exp3 and exp5 (with binarized data).
load("results/Hs/H_ANN_H100_K10000_binTRUE.RData", verbose = T)
load("results/final_counterfactuals_ANN_H100_K10000_binTRUE.RData", verbose = T)
final_counterfactuals_exp3 <- final_counterfactuals # Rename in case. 
load("resultsVAE/final_counterfactuals_ANN_H100_K10000_binTRUE.RData", verbose = T) # Etc

fact_num <- 10 # Select a factual we wanted explain.
h <- H[fact_num,] 
exp3 <- final_counterfactuals_exp3[[fact_num]][,-c(14,15)] # Remove "sparsity" and "Gower" from the row. 
exp5 <- final_counterfactuals_exp5[[fact_num]][,-c(14,15)]
col_names <- colnames(h)
df <- data.frame(cbind(as.character(h %>% mutate_if(is.factor, as.character)),
                       as.character(exp3 %>% mutate_if(is.factor, as.character)), 
                       as.character(exp5 %>% mutate_if(is.factor, as.character))))
colnames(df) <- c("h","exp3","exp5")
rownames(df) <- col_names
knitr::kable(df)

# Just for checking the predictions (that they actually are different).
# Code copied from experiment 3 (running this code assumes the correct parts from exp3 loaded in session).
library(keras)
ANN <- load_model_hdf5("classifiers/ANN_experiment3.h5") # Load the classifier for step 1. 
col_names <- colnames(x_test_ANN)
col_names_categ <- setdiff(col_names,cont)
df2 <- data.frame(rbind(h,exp3,exp5))
onehot_test_dat <- data.frame(df2[,cont])
col_names_D_h <- colnames(df2)
col_names_D_h <- setdiff(col_names_D_h,cont)
for (n in col_names_D_h){
  true_factors <- levels(adult.data[,n]) # Find the factors we want from adult.data
  for (new_col in 1:length(true_factors)){ # Make one new column per factor in the design matrix.
    column_name_new <- paste0(n,"..",substring(true_factors[new_col], 2, nchar(true_factors[new_col])))
    onehot_test_dat[,column_name_new] <- ifelse(df2[n] == true_factors[new_col], 1,0)
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

preds <- as.numeric(ANN %>% predict(as.matrix(onehot_test_dat)))
preds

# We add the preds to the table for visualization purposes. 
df <- rbind(df, "f()" = preds)
df

# Make fancy latex table for report. 
knitr::kable(df, format = "latex", linesep = "", digits = 3, booktabs = T) %>% 
  kable_styling(latex_options = c("scale_down")) %>% 
  column_spec(1, monospace = T) %>% 
  row_spec(c(1,9), background = "lgrey") %>% # Have defined a color lgrey in latex document.
  print()


############## Then we look at exp4 and exp6 (with the categorical data).
load("results/Hs/H_ANN_H100_K10000_binFALSE.RData", verbose = T)
load("results/final_counterfactuals_ANN_H100_K10000_binFALSE.RData", verbose = T)
final_counterfactuals_exp4 <- final_counterfactuals # Rename in case. 
load("resultsVAE/final_counterfactuals_ANN_H100_K10000_binFALSE.RData", verbose = T) # Etc

fact_num <- 10 # Select a factual we wanted explain.
h <- H[fact_num,] 
exp4 <- final_counterfactuals_exp4[[fact_num]][,-c(14,15)] # Remove "sparsity" and "Gower" from the row. 
exp6 <- final_counterfactuals_exp6[[fact_num]][,-c(14,15)]
col_names <- colnames(h)
df <- data.frame(cbind(as.character(h %>% mutate_if(is.factor, as.character)),
                       as.character(exp4 %>% mutate_if(is.factor, as.character)), 
                       as.character(exp6 %>% mutate_if(is.factor, as.character))))
colnames(df) <- c("h","exp4","exp6")
rownames(df) <- col_names
knitr::kable(df)

# Just for checking the predictions (that they actually are different).
# Code copied from experiment 3 (running this code assumes the correct parts from exp3 loaded in session).
library(keras)
ANN <- load_model_hdf5("classifiers/ANN_experiment4.h5") # Load the classifier for step 1. 
col_names <- colnames(x_test_ANN)
col_names_categ <- setdiff(col_names,cont)
df2 <- data.frame(rbind(h,exp4,exp6))
onehot_test_dat <- data.frame(df2[,cont])
col_names_D_h <- colnames(df2)
col_names_D_h <- setdiff(col_names_D_h,cont)
for (n in col_names_D_h){
  true_factors <- levels(adult.data[,n]) # Find the factors we want from adult.data
  for (new_col in 1:length(true_factors)){ # Make one new column per factor in the design matrix.
    column_name_new <- paste0(n,"..",substring(true_factors[new_col], 2, nchar(true_factors[new_col])))
    onehot_test_dat[,column_name_new] <- ifelse(df2[n] == true_factors[new_col], 1,0)
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

preds <- as.numeric(ANN %>% predict(as.matrix(onehot_test_dat)))
preds

# We add the preds to the table for visualization purposes. 
df <- rbind(df, "f()" = preds)
df

knitr::kable(df, format = "latex", linesep = "", digits = 3, booktabs = T) %>% 
  kable_styling(latex_options = c("scale_down")) %>% 
  column_spec(1, monospace = T) %>% 
  row_spec(c(1,9), background = "lgrey") %>% # Have defined a color lgrey in latex document.
  print()
