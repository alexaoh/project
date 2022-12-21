library(dplyr)

setwd("/home/ajo/gitRepos/project")

###################################### Loading and cleaning the Adult data.
data1 <- read.table("original_data/adult.data", sep = ",", header = F, na.strings = " ?") 
data2 <- read.table("original_data/adult.test", sep = ",", header = F, na.strings = " ?") 
colnames(data1) <- colnames(data2) <- c("age","workclass","fnlwgt","education","education_num",
                    "marital_status","occupation","relationship","race","sex",
                    "capital_gain","capital_loss","hours_per_week","native_country", "y")

dim(data1)[1] + dim(data2)[1] # Need to concat the test data and the other data given on the website to get all the data used in article. 
adult.data <- rbind(data1, data2) # This is the full dataset.
any(is.na(adult.data))
dim(adult.data)
adult.data <- na.omit(adult.data)
dim(adult.data)

# Make corrections to the variables (data types, etc).
adult.data$y[adult.data$y == " <=50K."] <- " <=50K"
adult.data$y[adult.data$y == " <=50K"] <- " <=50K"
adult.data$y[adult.data$y == " >50K."] <- " >50K"
adult.data$y[adult.data$y == " >50K"] <- " >50K"

# Fix binarization into 0 and 1 for y. 
response <- array(0,dim(adult.data)[1])
response[which(as.character(adult.data[,15])==" >50K")] <- 1
adult.data[,15] <- as.numeric(response)

adult.data$sex <- as.factor(adult.data$sex)
adult.data$workclass <- as.factor(adult.data$workclass)
adult.data$education <- as.factor(adult.data$education)
adult.data$marital_status <- as.factor(adult.data$marital_status)
adult.data$occupation <- as.factor(adult.data$occupation)
adult.data$relationship <- as.factor(adult.data$relationship)
adult.data$race <- as.factor(adult.data$race)
adult.data$native_country <- as.factor(adult.data$native_country)

summary(adult.data)
dim(adult.data)

# Before we save our dataset, we remove the columns that have not been used in the article.
adult.data <- adult.data[,-which(names(adult.data) %in% c("education"))] # We only remove "education".

summary(adult.data)
dim(adult.data)

save(adult.data, file = "data/adult_data_categ.RData") # Save the dataset including all factors etc.
