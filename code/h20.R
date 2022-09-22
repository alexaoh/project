setwd("/home/ajo/gitRepos/project")
load("data/adult_data_binarized.RData", verbose = T) # Binarized factors in the data. 

h2o.init() # Need to activate an H2o cluster.
data <- as.h2o(adult.data)

library(h2o)

splitData <- h2o.splitFrame(data, ratios = c(0.75,0.15), seed = 100)

train0 <- splitData[[1]]
valid0 <- splitData[[2]]
test0  <- splitData[[3]]

categorical = c(2,5,6,7,8,9,13)

y  <- "y"
x  <- setdiff(names(train0),y)


train0[,y] <- as.factor(train0[,y])
for(i in categorical)
  train0[,i] <- as.factor(train0[,i])

valid0[,y] <- as.factor(valid0[,y])
for(i in categorical)
  valid0[,i] <- as.factor(valid0[,i])

test0[,y] <- as.factor(test0[,y])
for(i in categorical)
  test0[,i] <- as.factor(test0[,i])


#Tar ut "Education" og "Relationship" fordi de er veldig korrelert med hhv
#"Education-Num" og  "Marital Status"
#x <- x[c(1:2,5:7,9:13)]

#Estimerer modellen. Her antar vi at vi har gjort en tuning av
#hyperparametere foerst og funnet ut at ett skjult lag er best.
# model.DL <-
# h2o.deeplearning(x=x,y=y,training_frame=train0,validation_frame=valid0,distribution="bernoulli",loss="CrossEntropy",seed=100,
# activation="RectifierWithDropout",
# hidden=c(300),input_dropout_ratio=0.2,hidden_dropout_ratio=c(0.7),l1=1e-5,epochs=10,
# variable_importances=TRUE,use_all_factor_levels=TRUE,balance_classes=F)

# Prøver å finne ut hvordan jeg kan gjøre den dårlig!?
model.DL <-
  h2o.deeplearning(x=x,y=y,training_frame=train0,validation_frame=valid0,distribution="bernoulli",loss="CrossEntropy",seed=123,
  activation="Rectifier",
  hidden=c(18),epochs=1, categorical_encoding = "OneHotExplicit")

pred  <- h2o.predict(model.DL,newdata=test0)
perf <- h2o.performance(model.DL,newdata=test0)
h2o.auc(perf)
plot(perf, type = "roc")

(tab <- table("preds" = as.numeric(as.data.frame(pred)["predict"][[1]]), "labs" = as.numeric(as.data.frame(test0)["y"][[1]])))
tab[4]/sum(tab[,2])
tab[4]/sum(tab[2,])
tab[1]/sum(tab[,1])
tab[1]/sum(tab[1,])
