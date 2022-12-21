# Some of the functions in our main algorithm code are tested as we go. 
# This is good practice and makes it easier for me to be certain that my utility-
# functions are working as they should.

rm(list = ls()) # make sure to remove previously loaded variables into the Session.
setwd("/home/ajo/gitRepos/project")

# Source the utility functions we are testing.
source("code/utilities.R")

# Load the data we need for the testing.
load("adult_data_binarized.RData", verbose = T)
#load("adult_data_categ.RData", verbose = T) # For when I want to do experiment with all categories intact. 

# List of continuous variables.
cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")

##################### Tests. 
test_normalization_and_denormalization <- function(){
  # Test that normalizing and de-normalizing gives the same data back. 
  norm.data <- normalize.data(data = adult.data, continuous_vars = cont)
  
  summary(norm.data[[1]]) # Now the data has been normalized. 
  norm.data[[2]]
  norm.data[[3]]
  
  OG.data <- de.normalize.data(data = norm.data[[1]], continuous_vars = cont, m.list = norm.data[[2]], M.list = norm.data[[3]])
  
  summary(OG.data)
  all.equal(OG.data, adult.data) # Test that normalizing and de-normalizing gives the same data back. 
}


test_gower_distance <- function(){
  norm.data <- normalize.data(data = adult.data, continuous_vars = cont)[[1]]
  x <- norm.data[1,-ncol(norm.data)]
  y <- norm.data[2,-ncol(norm.data)]
  g1 <- gower_D_h(x,y)["gower"] # The correct answer is 0.1751324 --> Good!
  g2 <- gower::gower_dist(x,y) # Not sure why this gives a difference answer? What features is it skipping all the time?
}


##################### Run the tests.
cat("Normalization and de-normalization returns the same data: ", test_normalization_and_denormalization(), "\n")

test_gower_distance()
