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

# Load pre-processed data

DATA <- load_ml_data("Data/Training")

dtrain <- lgb.Dataset(data = )




## FUNCTIONS

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
  
  # Load data into a list (organised)
  data <- lapply(full_paths, read_csv, show_col_types = FALSE)
  
  output <- list(
    X_train = as.matrix(data$X_train),
    X_test  = as.matrix(data$X_test),
    y_train = data$y_train$y,
    y_test = data$y_test$y
  )
  
  message("Successfully loaded data into a list.")
  return(output)
}





