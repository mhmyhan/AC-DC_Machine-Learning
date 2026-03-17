library(randomForest)
library(caret)
library(readr)
library(ggplot2)

set.seed(42)

# Confirm working directory
getwd()
list.files()

# Load preprocessed data
data_dir <- "Data/Training"

X_train <- read_csv(file.path(data_dir, "X_train.csv"), show_col_types = FALSE)
X_test  <- read_csv(file.path(data_dir, "X_test.csv"),  show_col_types = FALSE)

y_train <- read_csv(file.path(data_dir, "y_train.csv"), show_col_types = FALSE)$y
y_test  <- read_csv(file.path(data_dir, "y_test.csv"),  show_col_types = FALSE)$y

# Clean column names
colnames(X_train) <- make.names(colnames(X_train))
colnames(X_test)  <- make.names(colnames(X_test))

# Convert target to factor
y_train <- as.factor(y_train)
y_test  <- as.factor(y_test)

# Combine features and target
train_data <- X_train
train_data$target <- y_train

test_data <- X_test
test_data$target <- y_test

# Train Random Forest
rf_model <- randomForest(
  target ~ .,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

print(rf_model)

# Predictions
predictions <- predict(rf_model, newdata = test_data)

# Confusion matrix
confusionMatrix(predictions, test_data$target)

# ----- Visualisation -----

# Feature importance
varImpPlot(rf_model)

# ROC curve
library(pROC)
probs <- predict(rf_model, newdata = test_data, type = "prob")[,2]
roc_obj <- roc(test_data$target, probs)
plot(roc_obj, main = "ROC Curve")
auc(roc_obj)