# Open the data resulting from the simulations done on Markov.

setwd("/home/ajo/gitRepos/project")

load("results/D_hs/logreg_H100_K10000_binT.RData", verbose = T)

length(D_h_per_point)
dim(D_h_per_point[[1]])


rm(list = ls())
load("results/final_counterfactuals_logreg_H100_K10000_binT.RData", verbose = T)
length(final_counterfactuals)
dim(final_counterfactuals[[5]])
final_counterfactuals[[1]]

rm(list = ls())
load("results/final_counterfactuals_ANN_H100_K10000_binT.RData", verbose = T)
length(final_counterfactuals)
dim(final_counterfactuals[[5]])
final_counterfactuals[[1]]
