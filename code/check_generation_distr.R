# Here we check the distribution of the generated D_hs per h. 
# They should be very similar to the original data (given the fixed covariates)
# when generating with K = 10000.

#load("results/D_hs/ANN_H100_K10000_binTRUE.RData", verbose = T)
load("data/adult_data_binarized.RData", verbose = T) 
#load("data/adult_data_categ.RData", verbose = T) 
#load("results/D_hs/logreg_H100_K10000_binFALSE.RData", verbose = T)
load("results/D_hs/randomForest_H100_K10000_binTRUE.RData", verbose = T)

str(D_h_per_point[[1]])

cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")

adult.data.normalized <- normalize.data(data = adult.data, continuous_vars = cont) # returns list with data, mins and maxs.
# summary(adult.data.normalized[[1]])
# str(adult.data.normalized[[1]])

# We de-normalize the data seen above. 
D_h_first <- D_h_per_point[[1]]#de.normalize.data(D_h_per_point[[1]], cont, adult.data.normalized[[2]], adult.data.normalized[[3]])
str(D_h_first)
table(D_h_first$age)
table(D_h_first$sex)
summary(D_h_first) # We have a closer look at the distribution of the generated data. 
OG.data <-adult.data %>% filter(sex == " Male" & age == 38)
summary(OG.data)

# A closer look: 
(tab1 <- table(D_h_first$workclass))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)
(tab1 <- table(OG.data$workclass))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)

(tab1 <- table(D_h_first$marital_status))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)
(tab1 <- table(OG.data$marital_status))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)

(tab1 <- table(D_h_first$occupation))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)
(tab1 <- table(OG.data$occupation))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)

(tab1 <- table(D_h_first$relationship))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)
(tab1 <- table(OG.data$relationship))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)

(tab1 <- table(D_h_first$race))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)
(tab1 <- table(OG.data$race))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)

(tab1 <- table(D_h_first$native_country))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)
(tab1 <- table(OG.data$native_country))
tab1[1]/sum(tab1)
tab1[2]/sum(tab1)

summary(D_h_first %>% select(cont))
summary(OG.data %>% select(cont))
