# Writing the outline for algorithm 1 in MCCE by Aas et al.

# Next need to add some data to test this out. 
library(tree)

# How to use the library tree with some example data (for future reference)
# library(ISLR)
# data("Carseats")
# set.seed(4268)
# n = nrow(Carseats)
# train = sample(1:n, 0.7 * n, replace = F)
# test = (1:n)[-train]
# Carseats.train = Carseats[train, ]
# Carseats.test = Carseats[-train, ]
# tree.mod = tree(Sales ~ ., data = Carseats.train)
# summary(tree.mod)
# plot(tree.mod)
# text(tree.mod, pretty = 0)

H <- 1:10 # Points we want to explain.
K <- 100 # Number of returned possible counterfactuals before pre-processing.
u <- 10 # Number of fixed features. 
q <- 10 # Number of mutable features. 

T_j <- 1:10 # Vector of fitted trees!

for (h in H){
  D_h <- matrix(data = rep(NA, 1:K*u), nrow = K, ncol = u)
  
  # Fill the matrix D_h with copies of the vectors of fixed features. 
  for (i in 1:u){
    D_h[i,] <- 1 # Insert fixed feauture vector i.
  }
  
  # Now setup of D_h is complete. We move on to the second part, where we append columns to D_h. 
  
  for (j in 1:q){
    d <- rep(NA, K) # Empty vector of length K. Will be col-binded with D_h. 
    
    for (i in 1:K){
      # Add a single sample from the end node of tree T_j[j] based on data D_h[i,] to d[i] (not sure if this is the correct way).
      d[i] <- predict(T_j[j], data = D_h[i,]) 
    }
    D_h <- cbind(D_h,d)
  }
  # return D_h
}

# Post-processing.
# fulfilling criterion 3.
c <- 1e-10 # Threshold for removal.
D_h <- D_h[f(D_h) >= c,] # f(*) is the R function that predicts according to the model we want to make explanations for. 

# fulfilling criterion 4.
# Sparsity and Gowers distance.
