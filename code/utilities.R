# Utility functions are placed here. Then they are source into the main algorithm file. 

take.arguments <- function(){
  # Take command line arguments into the main algorithm file. This is done for easier simulation on remote machine.
  args <- commandArgs(trailingOnly = T)
  if (length(args) != 5){
    print("Default arguments are used")
            # method, length(H), K, generate (TRUE) or load (FALSE), binarized data (TRUE) or not (FALSE)
    args <- c("ANN",100,10000,FALSE,TRUE)
  } 
  return(args)
}


########################## Data processing tools. 
normalize <- function(x){
  # Normalize the vector x. 
  return((x- min(x))/(max(x)-min(x)))
}

de.normalize <- function(x, M, m){
  # De-normalize the vector x. 
  return(x*(M-m)+m)
}

normalize.data <- function(data, continuous_vars){
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

make.train.and.test <- function(data, train.ratio = 2/3){
  # Make the train and test data. Return all the four parts as a list.
  sample.size <- floor(nrow(data) * train.ratio)
  train.indices <- sample(1:nrow(data), size = sample.size)
  train <- data[train.indices, ]
  test <- data[-train.indices, ]
  
  x_train <- train[,-which(names(train) == "y")] # Training covariates. 
  y_train <- train[,c("y")] # Training label.
  x_test <- test[,-which(names(test) == "y")] # Testing covariates. 
  y_test <- test[,c("y")] # Testing label.
  return(list("x_train" = x_train, "y_train" = y_train, "x_test" = x_test, "y_test" = y_test, 
              "train" = train, "test" = test))
}


######################## Fit prediction models.
fit.ANN <- function(x_train, y_train, x_test, y_test){
  # Fit the ANN and return the keras object for the ANN.
  
  ANN <- keras_model_sequential() %>%
    layer_dense(units = 18, activation = 'relu', input_shape = c(ncol(x_train))) %>%
    layer_dense(units = 9, activation = 'relu') %>%
    layer_dense(units = 3, activation = 'relu') %>% 
    layer_dense(units = 1, activation = 'sigmoid')
  
  # compile (define loss and optimizer)
  ANN %>% compile(loss = 'binary_crossentropy',
                  optimizer = optimizer_rmsprop(),
                  metrics = c('accuracy'))
  
  # train (fit)
  history <- ANN %>% fit(x_train, y_train, epochs = 40, 
                         batch_size = 1024, validation_split = 0.2)
  # plot
  plot(history)
  
  print(summary(ANN))
  
  # evaluate on training data. 
  ANN %>% evaluate(x_train, y_train)
  
  # evaluate on test data. 
  ANN %>% evaluate(x_test, y_test)
  
  y_pred <- ANN %>% predict(x_test) %>% `>`(0.5) %>% k_cast("int32")
  y_pred <- as.array(y_pred)
  tab <- table("Predictions" = y_pred, "Labels" = y_test)
  print(confusionMatrix(factor(y_pred), factor(y_test)))
  print(roc(response = y_test, predictor = as.numeric(y_pred), plot = T))
  return(ANN)
}

fit.logreg <- function(x_train, y_train, x_test, y_test){
  # Fit the logreg and return the logreg-object. 
  lin_mod <- glm(y ~ ., family=binomial(link='logit'), data=data.frame(cbind(x_train, "y" = y_train)))
  print(summary(lin_mod))
  y_pred_logreg <- predict(lin_mod, data.frame(x_test), type = "response")
  y_pred_logreg[y_pred_logreg >= 0.5] <- 1
  y_pred_logreg[y_pred_logreg < 0.5] <- 0
  print(confusionMatrix(factor(y_pred_logreg), factor(y_test)))
  print(roc(response = y_test, predictor = as.numeric(y_pred_logreg), plot = T))
  return(lin_mod)
}

############################ Tools for post-processing of possible counterfactuals.
sparsity_D_h <- function(x_h,D_h){
  # Calculates sparsity for one counterfactual x_h; "Number of features changed between x_h and the counterfactual".
  D_h$sparsity <- rep(NA, nrow(D_h))
  if (nrow(D_h) >= 1){
    for (i in 1:nrow(D_h)){
      D_h[i,"sparsity"] <- sum(x_h != D_h[i,-which(names(D_h) %in% c("sparsity","gower","gowerpack"))]) # We remove newly added columns.
    }
  }
  return(D_h)
}

gower_D_h <- function(x_h, D_h){
  # Calculates Gower's distance for one counterfactual x_h.
  library(gower) # Could try to use this package instead of calculating everything by hand below!
  D_h$gower <- rep(NA, nrow(D_h))
  D_h$gowerpack <- rep(NA, nrow(D_h)) # Result from package.
  dtypes <- sapply(x_h[colnames(x_h)], class)
  
  if (nrow(D_h) >= 1){
    for (i in 1:nrow(D_h)){
      g <- 0 # Sum for Gower distance.
      p <- ncol(x_h)
      
      for (j in 1:p){ # Assuming that the features are already normalized! Perhaps they need to be normalized again!?
        d_j <- D_h[i,j]
        if (dtypes[j] == "numeric"){
          R_j <- 1 # normalization factor, see note in line above.
          g <- g + 1/R_j*abs(d_j-x_h[,j])
        } else if (dtypes[j] == "factor"){
          if (x_h[,j] != d_j){
            g <- g + 1
          }
        }
      }
      # Disse to er ulike!! Finn ut hvorfor!? Noe med normaliseringen jeg nevner ovenfor å gjøre?
      # Det kommer en warning om "zero or non-finite range" ved bruke av pakken!
      # Kanskje jeg bare skal sjekke at det gir mening det jeg har gjort manuelt først!
      D_h[i,"gower"] <- g/p
      #D_h[i,"gowerpack"] <- gower_dist(x_h,D_h[i,colnames(x_h)])
    }
  }
  return(D_h)
}
