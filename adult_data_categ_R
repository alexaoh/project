library(dplyr)

setwd("/home/ajo/gitRepos/project")
set.seed(42) # Set seed to begin with!

###################################### Loading and cleaning the Adult data.
data1 <- read.csv("adult.data", header = F) 
data2 <- read.csv("adult.test", header = F) 
colnames(data1) <- colnames(data2) <- c("age","workclass","fnlwgt","education","education_num",
                     "marital_status","occupation","relationship","race","sex",
                     "capital_gain","capital_loss","hours_per_week","native_country", "y")
dim(data1)[1] + dim(data2)[1] # Need to concat the test data and the other data given on the website to get all the data used in article. 
adult.data <- rbind(data1, data2) # This is the full dataset.
any(is.na(adult.data))

# Make corrections to the variables (data types, binarization, etc).
adult.data$y[adult.data$y == " <=50K."] <- "<=50K"
adult.data$y[adult.data$y == " <=50K"] <- "<=50K"
adult.data$y[adult.data$y == " >50K."] <- ">50K"
adult.data$y[adult.data$y == " >50K"] <- ">50K"
adult.data$y[adult.data$y == ">50K"] <- 1
adult.data$y[adult.data$y == "<=50K"] <- 0
adult.data$y <- as.numeric(adult.data$y)
adult.data$sex <- as.factor(adult.data$sex)

# binarize <- function(vector){
#   most_frequent <- names(sort(table(vector), decreasing = T))[1]
#   if (vector == most_frequent) vector = most.frequent else vector = "Other"
#   vector
# }

# Find most frequent value.
most_frequent <- function(vec){
  names(sort(table(vec), decreasing = T))[1]
}


binarize <- function(vec){
  vec[which(vec != most_frequent(vec))] <- "Other"
  vec <- as.factor(vec)
  vec
}

# Binarize all necessary columns
adult.data$workclass <- binarize(adult.data$workclass)
adult.data$education <- binarize(adult.data$education)
adult.data$marital_status <- binarize(adult.data$marital_status)
adult.data$occupation <- binarize(adult.data$occupation)
adult.data$relationship <- binarize(adult.data$relationship)
adult.data$race <- binarize(adult.data$race)
adult.data$native_country <- binarize(adult.data$native_country)

# write.csv(adult.data, file = "adult_data_binarized.csv", row.names = F)
save(adult.data, file = "adult_data_binarized.RData") # Save the dataset including all factors etc.

summary(adult.data)
