# Utility functions are placed here. 

normalize <- function(x){
  return((x- min(x))/(max(x)-min(x)))
}

de.normalize <- function(x, M, m){
  return(x*(M-m)+m)
}

# Save the minimums and maximums such that I can transform back later!

normalize.data <- function(data = adult.data, continuous_vars = cont){
  # Normalizes our data and returns the mins and maxs of all continuous variables, such that we can de-normalize later. 
  mins <- c()
  maxs <- c()
  
  for (j in continuous_vars){
    mins <- c(mins, min(data[,j]))
    maxs <- c(maxs, max(data[,j]))
    data[,j] <- normalize(data[,j])
  }
  return(list("d" = data, "mins" = mins, "maxs" = maxs))
}

de.normalize.data <- function(data, continuous_vars, m.list, M.list){
  # De-normalizes the data that has been normalized previously, using the mins and maxs returned under normalization.
  for (j in 1:length(continuous_vars)){
    v <- continuous_vars[j]
    data[,v] <- de.normalize(data[,v], m = m.list[j], M = M.list[j])
  }
  return(data)
}

