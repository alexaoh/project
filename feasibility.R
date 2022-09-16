# Feasibility: distance between the counterfactual and the training data
# As in the article, we choose Euclidean distance, k = 1/5 and w^[i] = 1/k = 1/5.
feasibility <- function(){
  # We skip the feasibility for now!
  k <- 5
  w <- 1/k
  p <- ncol(x_h)
  euclidean <- function(x1,x2) sqrt(sum(x1-x2)^2)
  f <- 0
  e <- unique_D_h[1,]
  
  # The line below is not feasible! Perhaps need to loop over each row and save each answer. Then sort after.
  #first_k_distances <- order(as.matrix(dist(rbind(e[,-which(names(e) %in% c("sparsity","gower","violation"))],adult.data)))[1,-1],decreasing = F)[1:k]
  
  # Find k nearest neighbors in dataset. 
  k_nearest <- function(e, k = 5,data = adult.data){
    n <- nrow(adult.data)
    distances <- rep(NA, n)
    for (r in 1:n){
      distances[r] <- euclidean(e,adult.data[r,]) # Hva kan jeg gjøre med factors??
    }
    distances.ordered <- order(distances, decreasing = F)
    return(distances.ordered[1:5])
  }
  
  k_nearest_to_e <- adult.data[k_nearest(e[,-which(names(e) %in% c("sparsity","gower","violation"))], k = 5, data = adult.data),]
  all.equal(k, length(k_nearest_to_e))
  
  for (i in 1:k){
    f <- f + w/p*euclidean(e,k_nearest_to_e[i]) # Kunne sikkert bare brukt distances.ordered her, i stedet for å beregne dette på nytt her!!
  }
  
  # Feasibility SPM: "K nearest observed data points" står det i artikkelen. Mener de da mellom e og dataen?
}
