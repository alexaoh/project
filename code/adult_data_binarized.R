library(dplyr)

setwd("/home/ajo/gitRepos/project")
set.seed(42) # Set seed to begin with!

###################################### Loading and cleaning the Adult data.
#data1 <- read.csv("original_data/adult.data", header = F) 
data1 <- read.table("original_data/adult.data", sep = ",", header = F, na.strings = " ?") # Bedre!!
#data2 <- read.csv("original_data/adult.test", header = F) 
data2 <- read.table("original_data/adult.test", sep = ",", header = F, na.strings = " ?") # Bedre!!
# adult.data <- read.csv("data_from_mcce_github.csv")
# colnames(adult.data) <- c("age","workclass","fnlwgt","education_num",
#                                           "marital_status","occupation","relationship","race","sex",
#                                           "capital_gain","capital_loss","hours_per_week","native_country", "y")
colnames(data1) <- colnames(data2) <- c("age","workclass","fnlwgt","education","education_num",
                      "marital_status","occupation","relationship","race","sex",
                      "capital_gain","capital_loss","hours_per_week","native_country", "y")

dim(data1)[1] + dim(data2)[1] # Need to concat the test data and the other data given on the website to get all the data used in article. 
adult.data <- rbind(data1, data2) # This is the full dataset.
any(is.na(adult.data))
dim(adult.data)
adult.data <- na.omit(adult.data)
dim(adult.data)

# Make corrections to the variables (data types, binarization, etc).
adult.data$y[adult.data$y == " <=50K."] <- " <=50K"
adult.data$y[adult.data$y == " <=50K"] <- " <=50K"
adult.data$y[adult.data$y == " >50K."] <- " >50K"
adult.data$y[adult.data$y == " >50K"] <- " >50K"
# adult.data$y[adult.data$y == ">50K"] <- 1
# adult.data$y[adult.data$y == "<=50K"] <- 0
#adult.data$y <- as.numeric(adult.data$y)

# Fix binarization into 0 and 1 for y. 
response <- array(0,dim(adult.data)[1])
response[which(as.character(adult.data[,15])==" >50K")] <- 1
adult.data[,15] <- as.numeric(response)

# Find most frequent value.
most_frequent <- function(vec){
  names(sort(table(vec), decreasing = T))[1]
}


binarize <- function(vec){
  vec[which(vec != most_frequent(vec))] <- "Other"
  vec <- as.factor(vec)
  vec
}

adult.data$sex <- as.factor(adult.data$sex)
# Binarize all necessary columns
adult.data$workclass <- binarize(adult.data$workclass)
adult.data$education <- binarize(adult.data$education)
adult.data$marital_status <- binarize(adult.data$marital_status)
adult.data$occupation <- binarize(adult.data$occupation)
adult.data$relationship <- binarize(adult.data$relationship)
adult.data$race <- binarize(adult.data$race)
adult.data$native_country <- binarize(adult.data$native_country)

summary(adult.data)
dim(adult.data)

# Before we save our dataset, we remove the columns that have not been used in the article.
adult.data <- adult.data[,-which(names(adult.data) %in% c("education"))] # We only remove "education".

summary(adult.data)
dim(adult.data)

# write.csv(adult.data, file = "adult_data_binarized.csv", row.names = F)
save(adult.data, file = "data/adult_data_binarized.RData") # Save the dataset including all factors etc.
