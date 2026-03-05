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
library(ggplot2)

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




###

## MODELING

# set seed for consistency
set.seed(42)

# Load pre-processed data
DATA <- load_ml_data("Data/Training")

table(DATA$y_train)
str(DATA$y_train)

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

print("\nScale Position Weight:")
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

# CONFUSION MATRIX
cm <- confusionMatrix(
  factor(pred_labels, levels = c(0,1)),
  factor(DATA$y_test, levels = c(0,1)),
  positive = "1"
)

print(cm)

cm_table <- as.data.frame(cm$table)


ggplot(cm_table, aes(Prediction, Reference, fill=Freq)) +
  geom_tile() +
  geom_text(aes(label=Freq), color="white", size=6) +
  scale_fill_gradient(low="steelblue", high="darkred") +
  labs(
    title="Confusion Matrix Heatmap",
    x="Predicted",
    y="Actual"
  )


## ROC METRICS
roc_obj <- roc(DATA$y_test, pred_probs)
auc(roc_obj)
plot(roc_obj)

# ROC plot
plot(
  roc_obj,
  col = "blue",
  lwd = 3,
  main = "ROC Curve for Breast Cancer Relapse Prediction"
)

abline(a=0, b=1, lty=2, col="grey")
text(0.6,0.2,paste("AUC =", round(auc(roc_obj),3)))


## Importance
# for clinical interpretation / ethical implications

importance <- lgb.importance(model)
print(importance)

lgb.plot.importance(importance,
                    top_n = 20,
                    measure = "Gain")

## SHAP
shap_values <- predict(
  model,
  DATA$X_train,
  type= "contrib"
)

shap_df <- as.data.frame(shap_values)

shap_long <- stack(shap_df)

ggplot(shap_long, aes(x = values)) +
  geom_histogram(bins = 50, fill = "steelblue") +
  facet_wrap(~ ind, scales = "free") +
  theme_minimal() +
  labs(title = "SHAP Value Distribution per Feature")



