
# Libraries
#install.packages("xgboost")
#install.packages("pROC")
#install.packages("readr")

library(pROC)
library(xgboost)
library(readr)

set.seed(42)
# Confirm working directory
getwd()
list.files("data/Training")

# table(y_train, useNA = "ifany") # To help debug data

#  Load preprocessed data

data_dir <- "data/Training"

X_train <- read_csv(file.path(data_dir, "X_train.csv"), show_col_types = FALSE)
X_test  <- read_csv(file.path(data_dir, "X_test.csv"),  show_col_types = FALSE)

y_train <- read_csv(file.path(data_dir, "y_train.csv"), show_col_types = FALSE)$y
y_test  <- read_csv(file.path(data_dir, "y_test.csv"),  show_col_types = FALSE)$y

# Ensure consistent encoding
y_train <- as.numeric(as.factor(y_train)) - 1
y_test  <- as.numeric(as.factor(y_test)) - 1

# Convert predictors to numeric matrix
X_train_mat <- as.matrix(X_train)
X_test_mat  <- as.matrix(X_test)

# Safety checks
stopifnot(sum(is.na(X_train_mat)) == 0)
stopifnot(sum(is.infinite(X_train_mat)) == 0)
stopifnot(sum(is.na(y_train)) == 0)

# Create DMatrix
dtrain <- xgb.DMatrix(data = X_train_mat, label = y_train)
dtest  <- xgb.DMatrix(data = X_test_mat,  label = y_test)

# Handle class imbalance
neg <- sum(y_train == 0)
pos <- sum(y_train == 1)
scale_pos_weight <- neg / pos

# Train XGBoost
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 3,
  eta = 0.05,
  scale_pos_weight = scale_pos_weight
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 600,
  verbose = 1
)

# Evaluate

y_prob <- predict(xgb_model, dtest)
y_pred <- ifelse(y_prob >= 0.5, 1, 0)

accuracy  <- mean(y_pred == y_test)
precision <- ifelse(sum(y_pred) == 0, 0,
                    sum(y_pred == 1 & y_test == 1) / sum(y_pred == 1))
recall    <- sum(y_pred == 1 & y_test == 1) / sum(y_test == 1)
f1_score  <- ifelse((precision + recall) == 0, 0,
                    2 * precision * recall / (precision + recall))
roc_auc   <- as.numeric(roc(y_test, y_prob)$auc)

metrics <- c(
  Accuracy  = accuracy,
  Precision = precision,
  Recall    = recall,
  F1_score  = f1_score,
  ROC_AUC   = roc_auc
)

cat("\nXGBoost Performance on TEST set:\n")
print(metrics)

# -------------------------------
# Optional: Save model
# -------------------------------
# xgb.save(xgb_model, "models/xgb_model.model")
