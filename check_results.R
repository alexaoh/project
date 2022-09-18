# Open the data resulting from the simulations done on Markov.

load("../results/D_hs/logreg_H100_K10000_binT.RData", verbose = T)

length(D_h_per_point)
dim(D_h_per_point[[1]])


rm(list = ls())
load("../results/final_counterfactuals_logreg_H100_K10000_binT.RData", verbose = T)
length(D_h_per_point)
dim(D_h_per_point[[5]])


rm(list = ls())
load("../results/final_counterfactuals.RData", verbose = T)
length(D_h_per_point)
dim(D_h_per_point[[5]])
