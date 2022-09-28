# Here we check the distribution of the generated D_hs per h. 
# They should be very similar to the original data (given the fixed covariates)
# when generating with K = 10000.

setwd("/home/ajo/gitRepos/project")

load("data/adult_data_binarized.RData", verbose = T) 
#load("data/adult_data_categ.RData", verbose = T) 
load("results/D_hs/randomForest_H100_K10000_binTRUE.RData", verbose = T)
#load("results/D_hs/randomForest_H100_K10000_binFALSE.RData", verbose = T)

str(D_h_per_point[[1]])

#cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")

#adult.data.normalized <- normalize.data(data = adult.data, continuous_vars = cont) # returns list with data, mins and maxs.
# summary(adult.data.normalized[[1]])
# str(adult.data.normalized[[1]])

# We de-normalize the data seen above. 
D_h_first <- D_h_per_point[[10]]#de.normalize.data(D_h_per_point[[1]], cont, adult.data.normalized[[2]], adult.data.normalized[[3]])
str(D_h_first)
table(D_h_first$age)
table(D_h_first$sex)
summary(D_h_first) # We have a closer look at the distribution of the generated data. 
OG.data <-adult.data %>% filter(sex == " Male" & age == 53)
str(OG.data)
summary(OG.data)

# A closer look: 
table(D_h_first$workclass)/sum(table(D_h_first$workclass))
table(OG.data$workclass)/sum(table(OG.data$workclass))

table(D_h_first$marital_status)/sum(table(D_h_first$marital_status))
table(OG.data$marital_status)/sum(table(OG.data$marital_status))

table(D_h_first$occupation)/sum(table(D_h_first$occupation))
table(OG.data$occupation)/sum(table(OG.data$occupation))

table(D_h_first$relationship)/sum(table(D_h_first$relationship))
table(OG.data$relationship)/sum(table(OG.data$relationship))

table(D_h_first$race)/sum(table(D_h_first$race))
table(OG.data$race)/sum(table(OG.data$race))

table(D_h_first$native_country)/sum(table(D_h_first$native_country))
table(OG.data$native_country)/sum(table(OG.data$native_country))


summary(D_h_first %>% select(cont))
summary(OG.data %>% select(cont))
