#########################################################
# Author: Max Myhan - C2268310
# Email: c2268310@live.tees.ac.uk
#########################################################

## LIBRARIES
library(lightgbm)
library(readr)
library(pROC)
library(caret)
library(stringr)

## FUNCTIONS (for readability)

load_ml_data <- function(data_dir) {
  # Check that the directory exists
  if (!dir.exists(data_dir)) {
    stop(paste("Directory not found: ", data_dir))
  }
  
  # label expected files
  files <- c(X_train = "X_train.csv",
             X_test = "X_test.csv",
             y_train = "y_train.csv",
             y_test = "y_test.csv")
  
  # verify existence of files before loading
  full_paths <- file.path(data_dir, files)
  if (!all(file.exists(full_paths))) {
    stop("One or more required files are missing from the dir.")
  }
  
  # Load data into a list (more organised)
  data <- setNames(
    lapply(full_paths, read_csv, show_col_types = FALSE),
    names(files))
  
  output <- list(
    X_train = data.matrix(data$X_train),
    X_test  = data.matrix(data$X_test),
    y_train = data$y_train$y,
    y_test = data$y_test$y
  )
  
  message("Successfully loaded data into a list.")
  return(output)
}

table(DATA$y_train)
str(DATA$y_train)

###

## MODELING

# Load pre-processed data
DATA <- load_ml_data("Data/Training")

# Additional pre-processing
# Convert to binary numerals
DATA$y_train <- ifelse(DATA$y_train == "Recurred", 1, 0)
DATA$y_test  <- ifelse(DATA$y_test  == "Recurred", 1, 0)

str(DATA)
names(DATA)

# Create dataset
dtrain <- lgb.Dataset(data = DATA$X_train, label = DATA$y_train)

# compute class weightings
neg <- sum(DATA$y_train == 0)
pos <- sum(DATA$y_train == 1)

scale_pos_weight <- neg/pos

str(scale_pos_weight)

# train model
default_params <- list(
  objective = "binary",
  metric = "auc",
  learning_rate = 0.05,
  num_leaves = 31,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 5,
  scale_pos_weight = scale_pos_weight
)

model <- lgb.train(
  params = default_params,
  data = dtrain,
  nrounds = 400,
  verbose = 1
)

## EVALUATION

pred_probs <- predict(model, DATA$X_test)

pred_labels <- ifelse(pred_probs > 0.5, 1, 0)

confusionMatrix(
  factor(pred_labels, levels = c(0,1)),
  factor(DATA$y_test, levels = c(0,1))
)

roc_obj <- roc(DATA$y_test, pred_probs)
auc(roc_obj)
plot(roc_obj)

## Importance
# for clinical interpretation / ethical implications

importance <- lgb.importance(model)
print(importance)

lgb.plot.importance(importance, top_n = 20)



