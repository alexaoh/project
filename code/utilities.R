# Utility functions are placed here. Then they are sourced into the main algorithm file. 

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
  # Normalize the vector x. This is not normalization but min-max scaling. 
  return((x- min(x))/(max(x)-min(x)))  
}

de.normalize <- function(x, M, m){
  # De-normalize the vector x. De-min-max transform the data. 
  return(x*(M-m)+m)
}

normalize.data <- function(data, continuous_vars, standardscaler){
  # Normalizes our data and returns the mins and maxs of all continuous variables, such that we can de-normalize later. 
  if (standardscaler){
    d <- scale(data[,cont])
    categ <- setdiff(names(data), cont)
    full_data <- cbind(d, data[,categ])[,colnames(data)]
    return(list("d" = full_data, "means" = attr(d, "scaled:center"), "sds" = attr(d, "scaled:scale")))
  } else {
    mins <- c()
    maxs <- c()
    
    for (j in continuous_vars){
      mins <- c(mins, min(data[,j]))
      maxs <- c(maxs, max(data[,j]))
      data[,j] <- normalize(data[,j])
    }
    return(list("d" = data, "mins" = mins, "maxs" = maxs))
  }
}

de.normalize.data <- function(data, continuous_vars, m.list, M.list){
  # De-normalizes the data that has been normalized previously, using the mins and maxs returned under normalization.
  for (j in 1:length(continuous_vars)){
    v <- continuous_vars[j]
    data[,v] <- de.normalize(data[,v], m = m.list[j], M = M.list[j])
  }
  return(data)
  # Should/could also be extended to the standardscaler option eventually.
}

make.train.and.test <- function(data, train.ratio = 2/3){
  # Make the train and test data. Return all the four parts as a list. We also return the train indices.
  sample.size <- floor(nrow(data) * train.ratio)
  train.indices <- sample(1:nrow(data), size = sample.size)
  train <- data[train.indices, ]
  test <- data[-train.indices, ]
  
  x_train <- train[,-which(names(train) == "y")] # Training covariates. 
  y_train <- train[,c("y")] # Training label.
  x_test <- test[,-which(names(test) == "y")] # Testing covariates. 
  y_test <- test[,c("y")] # Testing label.
  return(list("x_train" = x_train, "y_train" = y_train, "x_test" = x_test, "y_test" = y_test, 
              "train_indices" = train.indices))
}

plot_tree <- function(index){
  # Helper function to plot each tree nicely. Also prints the formula that was used to construct the tree. 
  par(mar = c(1,1,1,1))
  cat("Formula fitted: ")
  print(total_formulas[[index]])
  cat("\n")
  tree.mod <- T_j[[index]]
  print(summary(tree.mod))
  if (tree.mod$method == "class"){
    rpart.plot::prp(tree.mod, extra = 4)  
  } else {
    rpart.plot::prp(tree.mod)
  }
}

make.data.for.ANN <- function(data, cont, label){
  # Make design matrix via one-hot encoding of the categorical variables. 
  text <- data[,-which(names(data) %in% cont)]
  if (label){
    text <- text[,-ncol(text)] # Remove the label! 
  }
  numbers <- data[,which(names(data) %in% cont)]
  encoded <- caret::dummyVars(" ~ .", data = text, fullRank = F)
  data_encoded <- data.frame(predict(encoded, newdata = text))
  
  if (label){
    final_data <- cbind(numbers, data_encoded, data["y"])  
  } else {
    final_data <- cbind(numbers, data_encoded)  
  }
  
  return(final_data)
}

reverse.onehot.encoding <- function(data, cont, categ, has.label){
  # Reverse one-hot encoded design matrices. The function uses the design matrix (data) and two lists of names (continuous and categorical).
  text <- data[,-which(names(data) %in% cont)]
  if (has.label){
    text <- text[,-ncol(text)] # Remove the label.
  }
  numbers <- data[,which(names(data) %in% cont)]
  new_text <- c()
  for (name in categ){
    d <- data %>% dplyr::select(starts_with(name))
    categorical_value <- names(d)[max.col(d)]
    new_text <- cbind(new_text, categorical_value)
  }
  new_text <- as.data.frame(new_text)
  colnames(new_text) <- categ
  
  if (has.label){
    r <- cbind(numbers, new_text, data["y"])
  } else {
    r <- cbind(numbers, new_text)
  }
  return(r)
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
                  optimizer = optimizer_adam(), 
                  metrics = c('accuracy'))
  
  # train (fit)
  history <- ANN %>% fit(x_train, y_train, epochs = 30, 
                         batch_size = 1024, validation_split = 0.2)
  # plot
  plot(history)
  
  print(summary(ANN))
  
  # evaluate on training data. 
  ANN %>% evaluate(x_train, y_train)
  
  # evaluate on test data. 
  ANN %>% evaluate(x_test, y_test)
  
  y_pred <- ANN %>% predict(x_test) 
  print(confusionMatrix(factor(as.numeric(y_pred %>% `>=`(0.5))), factor(y_test)))
  print(roc(response = y_test, predictor = as.numeric(y_pred), plot = T))
  results <- HMeasure(y_test,as.numeric(y_pred),threshold=0.5)
  print(results$metrics$AUC)
  return(ANN)
}

fit.logreg <- function(x_train, y_train, x_test, y_test){
  # Fit the logreg and return the logreg-object. 
  lin_mod <- glm(y ~ ., family=binomial(link='logit'), data=data.frame(cbind(x_train, "y" = y_train)))
  print(summary(lin_mod))
  y_pred_logreg <- predict(lin_mod, data.frame(x_test), type = "response")
  print(confusionMatrix(factor(as.numeric(y_pred_logreg %>% `>=`(0.5))), factor(y_test)))
  print(roc(response = y_test, predictor = as.numeric(y_pred_logreg), plot = T))
  results <- HMeasure(y_test,as.numeric(y_pred_logreg),threshold=0.5)
  print(results$metrics$AUC)
  return(lin_mod)
}

fit.random.forest <- function(x_train, y_train, x_test, y_test){
  # Fit the random forest and return the random forest object.
  model <- ranger(as.factor(y_train) ~ ., data = x_train, num.trees = 500, num.threads = 6,
                  verbose = TRUE,
                  probability = TRUE,
                  importance = "impurity",
                  mtry = sqrt(13))
  pred.rf <- predict(model, data = x_test)$predictions[,2]
  print(confusionMatrix(factor(as.numeric(pred.rf %>% `>=`(0.5))), factor(y_test)))
  results <- HMeasure(y_test,pred.rf,threshold=0.5)
  print(results$metrics$AUC)
  print(roc(response = y_test, predictor = as.numeric(pred.rf), plot = T))
  return(model)
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

gower_D_h <- function(x_h, D_h, norm.factors){
  # Calculates Gower's distance for one counterfactual x_h.
  D_h$gower <- rep(NA, nrow(D_h))
  dtypes <- sapply(x_h[colnames(x_h)], class)
  
  if (nrow(D_h) >= 1){
    for (i in 1:nrow(D_h)){
      g <- 0 # Sum for Gower distance.
      p <- ncol(x_h)
  
      for (j in 1:p){ 
        d_j <- D_h[i,j]
        if (dtypes[j] == "integer" || dtypes[j] == "numeric"){ # If we normalize we need to have "numeric" here.
          m_j <- norm.factors[[j]][1]
          M_j <- norm.factors[[j]][2]
          z <- abs(d_j-x_h[,j])
          R_j <- M_j-m_j # I think this solution makes more sense! We assume that the data is well modeled earlier, such that 
          # z does not become larger than R_j and the total factor delta_G will stay between 0 and 1. 
          g <- g + 1/R_j*z
          
        } else if (dtypes[j] == "factor"){
          if (x_h[,j] != d_j){
            g <- g + 1
          }
        }
      }
      D_h[i,"gower"] <- g/p
    }
  }
  return(D_h)
}
