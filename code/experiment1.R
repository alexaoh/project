# Code for producing results in Experiment 1. 

library(rpart)
library(dplyr)
library(ggplot2)

setwd("/home/ajo/gitRepos/project")

args <- commandArgs(trailingOnly = T)

set.seed(1234)
# Load the data.
if (args[1] == "bin"){
  load("data/adult_data_binarized.RData", verbose = T)  
} else if (args[1] == "cat"){
  load("data/adult_data_categ.RData", verbose = T)
} else {
  stop("Wrong argument!")
}

cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
categ <- setdiff(names(adult.data), cont)
categ <- categ[-length(categ)] # Remove the label "y"!

# I build new trees, such that I can build a first tree for sex ~ age. 
data_min_response <- adult.data[,-which(names(adult.data) == "y")] # All covariates (removed the response from the data frame).

fixed_features <- c("age") # Now we do not have any fixed values but I keep this here for easier implementation. 
mut_features <- base::setdiff(colnames(data_min_response), fixed_features)[c(8,1,2,3,4,5,6,7,9,10,11,12)] 
# We rearrange the "mut_features", such that sex is the first on (such that we get age, sex, etc in the list of features).
mut_datatypes <- sapply(data_min_response[mut_features], class)
u <- length(fixed_features) # Number of fixed features. 
q <- length(mut_features) # Number of mutable features. 
p <- q+u # Total number of features.
all.equal(ncol(data_min_response), p) # We want to check that p is correctly defined. Looks good!

fit.trees <- function(){
  # Fit the regression trees and add all these objects to a list.
  T_j <- list() # Vector of fitted trees!
  fixed_form <- paste(fixed_features, collapse = "+") # Fixed features, for making the formula. 
  total_formulas <- list()
  for (i in 1:q){
    covariates <- paste(c(fixed_features,mut_features[1:i-1]), collapse = "+")
    tot_form <- as.formula(paste(mut_features[i]," ~ ", covariates, sep= ""))
    total_formulas[[i]] <- tot_form

    if (mut_datatypes[[i]] == "factor"){ 
      T_j[[i]] <- rpart(tot_form, data = adult.data, method = "class", control = rpart.control(minbucket = 5, cp = 1e-6)) 
      # Method = "class": Uses Gini index by default. 
    } else if (mut_datatypes[[i]] == "integer" || mut_datatypes[[i]] == "numeric"){ # mean squared error.
      T_j[[i]] <- rpart(tot_form, data = adult.data, method = "anova", control = rpart.control(minbucket = 5, cp = 1e-10)) 
      # Method = "anova": SST-(SSL+SSR). 
    } else { 
      stop("Error: Datatypes need to be either factor or integer/numeric.") # We need to use "numeric" if we have normalized the data!
    } 
  }
  return(list("trees" = T_j, "formulas" = total_formulas))
}

fitted.tree.list <- fit.trees()
T_j <- fitted.tree.list[[1]]
total_formulas <- fitted.tree.list[[2]]
length(T_j) # We have 12 trees now. 
total_formulas

# We simply sample from our original data and make a similar generation algorithm to before. 
K <- 1e5L
D <- as.data.frame(matrix(data = rep(NA, K*p), nrow = K, ncol = p))
colnames(D) <- c(fixed_features, mut_features)

# Fill the matrix D with different samples from the true data set. 
# All columns except sex and age (for now) will be replaced with generated sample values below.
s <- slice_sample(adult.data, n = K, replace = T) # Sample K rows from the original data set. 
# The two lines below are split up like this in order to keep the order as in my other experiments (with "fixed" first).
D[, fixed_features] <- s %>% dplyr::select(all_of(fixed_features))
D[, mut_features] <- s %>% dplyr::select(all_of(mut_features)) # Add this to get the correct datatypes (these are not used when predicting though!)


# Now setup of D is complete. We move on to the second part, where we append columns to D.
generate_unconditional <- function(D_h){
  # Since we had some problems generating from the first tree (the value of age has to be a data.frame) 
  # but is only returned as a number when I select one cell at a time, we simply generate quickly from the tree first. 
  for (i in 1:K){
    # For age!
    end_node_distr <- predict(T_j[[1]], newdata = data.frame("age" = D_h[i,1]))
    sorted <- sort(end_node_distr, decreasing = T, index.return = T)
    largest_class <- sorted$x
    largest_index <- sorted$ix
    # Since sex is a factor, we do the following.
    D_h[i,2] <- sample(x = levels(adult.data[,"sex"])[largest_index], size = 1, prob = largest_class)
  }
  
  cat("Generated for feature: ", 1, "\n")
  
  # Then we generate as normal for the other features. 
  for (j in 2:q){
    feature_regressed <- mut_features[j]
    feature_regressed_dtype <- mut_datatypes[[j]]
    
    d <- rep(NA, K) # Empty vector of length K. 
    # Will be inserted into D later (could col-bind also, but chose to instantiate entire D from the beginning).
    
    for (i in 1:K){
      # Add a single sample from the end node of tree T_j[j] based on data D[i,u+j] to d[i].
      end_node_distr <- predict(T_j[[j]], newdata = D_h[i,1:(u+j-1)]) # Usikker på om "predict" blir korrekt her? Burde det vært en "where" for å finne indeks først?
      sorted <- sort(end_node_distr, decreasing = T, index.return = T)
      largest_class <- sorted$x
      largest_index <- sorted$ix
      if (feature_regressed_dtype == "factor"){
        d[i] <- sample(x = levels(adult.data[,feature_regressed])[largest_index], size = 1, prob = largest_class) 
      } else { # Numeric
        d[i] <- end_node_distr
      }
    }
    D_h[,u+j] <- d # Add all the tree samples based on the jth mutable feature to the next column. 
    cat("Generating for feature: ", j, "\n")
  }
  return(D_h[,colnames(adult.data)[-length(colnames(adult.data))]] %>% mutate_if(is.character,as.factor))
}

D2 <- generate_unconditional(D) # We generate samples from the trees.
if (args[1] == "bin"){
  save(D2, file = "results/unconditional_generated_trees_bin.RData") # Save the generated D_h per point.  
} else if (args[1] == "cat"){
  save(D2, file = "results/unconditional_generated_trees_cat.RData")
}

# Next we compare this generated set of data from the trees to the real data!

load("results/unconditional_generated_trees_bin.RData", verbose = T)
load("results/unconditional_generated_trees_cat.RData", verbose = T)

table(D2$sex)/sum(table(D2$sex))
table(adult.data$sex)/sum(table(adult.data$sex))

table(D2$workclass)/sum(table(D2$workclass))
table(adult.data$workclass)/sum(table(adult.data$workclass))

table(D2$marital_status)/sum(table(D2$marital_status))
table(adult.data$marital_status)/sum(table(adult.data$marital_status))

table(D2$occupation)/sum(table(D2$occupation))
table(adult.data$occupation)/sum(table(adult.data$occupation))

table(D2$relationship)/sum(table(D2$relationship))
table(adult.data$relationship)/sum(table(adult.data$relationship))

table(D2$race)/sum(table(D2$race))
table(adult.data$race)/sum(table(adult.data$race))

table(D2$native_country)/sum(table(D2$native_country))
table(adult.data$native_country)/sum(table(adult.data$native_country))

summary(D2 %>% dplyr::select(cont))
summary(adult.data %>% dplyr::select(cont))

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

knitr::kable(cont.summary(adult.data), format = "latex", linesep = "", digits = 1, booktabs = T) %>% print()
knitr::kable(cont.summary(D2), format = "latex", linesep = "", digits = 1, booktabs = T) %>% print()

# Looking at bit more closely at some of the continuous features.
cap_gain_OG <- (adult.data %>% dplyr::select(capital_gain))[[1]]
cap_gain_gen <- (D2 %>% dplyr::select(capital_gain))[[1]]
summary(cap_gain_OG)
summary(cap_gain_gen)
length(cap_gain_OG[cap_gain_OG != 0])/length(cap_gain_OG)
length(cap_gain_gen[cap_gain_gen != 0])/length(cap_gain_gen)

cap_loss_OG <- (adult.data %>% dplyr::select(capital_loss))[[1]]
cap_loss_gen <- (D2 %>% dplyr::select(capital_loss))[[1]]
summary(cap_loss_OG)
summary(cap_loss_gen)
length(cap_loss_OG[cap_loss_OG != 0])/length(cap_loss_OG)
length(cap_loss_gen[cap_loss_gen != 0])/length(cap_loss_gen)

# Make plots for showing ratios between levels in categorical features. 
# First iteration of plots. The final type of plot that was used can be found in "investigate_data.R"
make_ggplot_for_categ <- function(data, filename, save){
  data.categ <- data[,categ]
  data.categ.wide <- data.categ %>% tidyr::pivot_longer(categ) %>% count(name, value) %>% mutate(ratio = round(n/nrow(data.categ), 3))
  categ_plot <- data.categ.wide %>% ggplot(aes(x = name, y = ratio, fill = value)) +
    geom_col(position = "stack", show.legend = F) + # Need to remove legend for categorical data (for readibility).
    geom_text(aes(label = ratio), position = position_stack(vjust = 0.5)) +
    theme_minimal() 
  if (save) ggsave(paste0("plots/",filename,".pdf"), width = 10, height = 5)
  return(categ_plot)
}

# These are not very nice for the categorical data. 
make_ggplot_for_categ(adult.data, "adult_data_categ_ratios_cat_data", T)
make_ggplot_for_categ(D2, "generated_exp1_categ_ratios_cat_data", T)
