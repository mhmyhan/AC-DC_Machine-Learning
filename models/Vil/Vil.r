
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

