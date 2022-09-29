p1 <- c("large", "small")
p <- c(0.8,0.2)

set.seed(1)

samp <- function(){
  s <- runif(1)
  if (s >= p[1]){ # This only works for two classes at this point! Perhaps I can simply use the sample function with the list of probabilities?
    d <- p1[2]
  } else {
    d <- p1[1]
  }
  return(d)
}

samp2 <- function(){
  return(sample(p1, 1, prob = p))
}

K <- 1000000

d <- rep(NA,K)
d2 <- rep(NA,K)
for (i in 1:K){
  d[i] <- samp()
  d2[i] <- samp2()
}

table(d)
table(d2)
