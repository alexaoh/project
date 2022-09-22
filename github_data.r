# Load the data I got from the repo on GitHub!!
github_data <- read.csv("data_from_mcce_github.csv")
summary(github_data)


github_data$sex <- as.factor(github_data$sex)
github_data$workclass <- as.factor(github_data$workclass)
github_data$marital.status <- as.factor(github_data$marital.status)
github_data$occupation <- as.factor(github_data$occupation)
github_data$relationship <- as.factor(github_data$relationship)
github_data$race <- as.factor(github_data$race)
github_data$native.country <- as.factor(github_data$native.country)


##### Tester fra https://authoritypartners.com/deep-neural-networks-with-r-tensorflow-and-keras/ her
data <- read.table("original_data/adult.data", sep = ",", header = F, na.strings = " ?")
summary(data)
dim(data)
colnames(data) <- c("age","workclass","fnlwgt","education","education_num",
  "marital_status","occupation","relationship","race","sex",
  "capital_gain","capital_loss","hours_per_week","native_country", "y")
summary(data)
any(is.na(data))
data <- na.omit(data)
data$sex <- as.factor(data$sex)
data$workclass <- as.factor(data$workclass)
data$marital_status <- as.factor(data$marital_status)
data$occupation <- as.factor(data$occupation)
data$relationship <- as.factor(data$relationship)
data$race <- as.factor(data$race)
data$native_country <- as.factor(data$native_country)
data$y <- as.factor(data$y)
summary(data)

cont <- c("age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week")
train_text <- data[,-which(names(data) %in% cont)]
train_text <- train_text[,-ncol(train_text)] # Remove the label!
train_numbers <- data[,which(names(data) %in% cont)]
encoded <- caret::dummyVars(" ~ .", data = train_text, fullRank = F)
train_encoded <- data.frame(predict(encoded, newdata = train_text))

data <- cbind(train_numbers, train_encoded, data["y"])

sample.size <- floor(0.8*nrow(data))
set.seed(123)
train_idx <- sample(nrow(data), size = sample.size)
training_set <- data[train_idx, ]
test_set <- data[-train_idx, ]

training_labels <- as.numeric(training_set[,15])
training_set <- as.matrix(training_set %>% select(-y))
test_labels <- as.numeric(test_set[,15])
test_set <- as.matrix(test_set %>% select(-y))

model <-  keras_model_sequential()
model %>% 
  layer_dense(units = 64, activation = "relu", input_shape = ncol(training_set)) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

history <- model %>% fit(
  training_set, 
  training_labels,
  epochs = 20,
  batch = 16,
  validation_split = 0.15
)

plot(history)

y_pred <- model %>% predict(test_set) %>% `>=`(0.5) #%>% k_cast("int32")
y_pred <- as.array(y_pred)
print(confusionMatrix(factor(y_pred), factor(test_labels)))
print(roc(response = test_labels, predictor = as.numeric(y_pred), plot = T))
# I artikkelen predikerer de alltid 0!! juks!!

ANN <- fit.ANN(training_set, training_labels, test_set, test_labels)
