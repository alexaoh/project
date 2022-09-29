# In this file we test how the generated distributions from the trees are, 
# when we do not fix on any ages in the steps of the generation.

# I assume that the data and the trees are loaded into the global scope already, 
# so no need to source or import anything. 

# We simply sample from our original data and make a similar generation algorithm to before. 
set.seed(1234)
K <- 100
p <- ncol(adult.data)
D <- as.data.frame(matrix(data = rep(NA, K*p), nrow = K, ncol = p))
colnames(D) <- c(fixed_features, mut_features)

# Fill the matrix D with different samples from the true data set. 
# All columns except sex and age (for now) will be replaced with generated sample values below.
s <- slice_sample(adult.data, n = K)
D[, fixed_features] <- s %>% dplyr::select(all_of(fixed_features))
D[, mut_features] <- s %>% dplyr::select(all_of(mut_features)) # Add this to get the correct datatypes (these are not used when predicting though!)

# Now setup of D is complete. We move on to the second part, where we append columns to D.

for (j in 1:q){
  feature_regressed <- mut_features[j]
  feature_regressed_dtype <- mut_datatypes[[j]]
  
  d <- rep(NA, K) # Empty vector of length K. 
  # Will be inserted into D later (could col-bind also, but chose to instantiate entire D from the beginning).
  
  for (i in 1:K){
    # Add a single sample from the end node of tree T_j[j] based on data D[i,u+j] to d[i].
    end_node_distr <- predict(T_j[[j]], newdata = D[i,1:(u+j-1)]) # Usikker på om "predict" blir korrekt her? Burde det vært en "where" for å finne indeks først?
    sorted <- sort(end_node_distr, decreasing = T, index.return = T)
    largest_class <- sorted$x
    largest_index <- sorted$ix
    if (feature_regressed_dtype == "factor"){
      # s <- runif(1)
      # if (s >= largest_class[1]){ # This only works for two classes at this point! Perhaps I can simply use the sample function with the list of probabilities?
      #   d[i] <- levels(adult.data[,feature_regressed])[largest_index[2]]
      # } else {
      #   d[i] <- levels(adult.data[,feature_regressed])[largest_index[1]]
      # }
      # I think the following is a better solution. This works for the categorical data as well!
      d[i] <- sample(x = levels(adult.data[,feature_regressed])[largest_index], size = 1, prob = largest_class) 
      d[i] <- end_node_distr
    }
  }
  D[,u+j] <- d # Add all the tree samples based on the jth mutable feature to the next column. 
}
return(D[,colnames(adult.data)[-length(colnames(adult.data))]] %>% mutate_if(is.character,as.factor))
# Put this inside a function!! The algorithm should be correct like this I think, for generating non-conditional / 
# not based on any fixed covariates data!