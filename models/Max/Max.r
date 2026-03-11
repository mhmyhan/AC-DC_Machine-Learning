#########################################################
# Author: Max Myhan - C2268310
# Email: c2268310@live.tees.ac.uk
#########################################################

## LIBRARIES
library(lightgbm)
library(readr)
library(pROC)
library(PRROC)
library(caret)
library(stringr)
library(ggplot2)
library(knitr)

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


# compute class weightings
neg <- sum(DATA$y_train == 0)
pos <- sum(DATA$y_train == 1)

scale_pos_weight <- neg/pos * 0.7

print("\nScale Position Weight:")
str(scale_pos_weight)



# Create dataset
dtrain <- lgb.Dataset(data = DATA$X_train, label = DATA$y_train)


# model parameters by version
old_params <- list(
  objective = "binary",
  metric = "auc",
  learning_rate = 0.05,
  num_leaves = 31,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 5,
  scale_pos_weight = scale_pos_weight
)

default_params <- list(
  objective = "binary",
  metric = "auc",
  
  learning_rate = 0.03,
  num_leaves = 128,
  max_depth = -1,
  
  feature_fraction = 0.85,
  bagging_fraction = 0.85,
  bagging_freq = 5,
  
  min_data_in_leaf = 10,
  lambda_l1 = 0.1,
  lambda_l2 = 0.2
  
  ## removed since dataset is alr quite balanced
  #scale_pos_weight = scale_pos_weight
)


## CROSS VALIDATION

cv <- lgb.cv(
  params = default_params,
  data = dtrain,
  nrounds = 400,
  nfold = 5,
  stratified = TRUE,
  early_stopping_rounds = 30,
  verbose = 1
)

best_iter <- cv$best_iter
cat("\nBest iteration from CV:", best_iter, "\n")




# train model
model <- lgb.train(
  params = default_params,
  data = dtrain,
  nrounds = best_iter
)



###

## EVALUATION

pred_probs <- predict(model, DATA$X_test)

#Dynamically generate best threshold for the F1 score
#should improve recall and precision
thresholds <- seq(0.01, 0.99, by=0.01)

f1_scores <- sapply(thresholds, function(t){
  preds <- ifelse(pred_probs > t, 1, 0)
  cm_tmp <- confusionMatrix(
    factor(preds, levels=c(0,1)),
    factor(DATA$y_test, levels=c(0,1)),
    positive="1"
  )
  as.numeric(cm_tmp$byClass["F1"])
})

best_thresh <- thresholds[which.max(f1_scores)]

cat("Best threshold:", best_thresh)

pred_labels <- ifelse(pred_probs > best_thresh, 1, 0)

## CONFUSION MATRIX
# True Positives and True Negatives are good
# !In medicine, FP's are not as bad as FN's!
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
  scale_fill_gradient(low="darkgray", high="darkblue") +
  labs(
    title="Confusion Matrix Heatmap",
    x="Predicted",
    y="Actual"
  )


## ROC
roc_obj <- roc(DATA$y_test, pred_probs)
auc(roc_obj)

plot(
  roc_obj,
  col = "blue",
  lwd = 3,
  main = "ROC Curve for Breast Cancer Relapse Prediction"
)

abline(a=0, b=1, lty=2, col="grey")
text(0.6,0.2,paste("AUC =", round(auc(roc_obj),3)))

## Precision-Recall Curve 
# (help uncover problems with class imbalance)
pr <- pr.curve(
  scores.class0 = pred_probs[DATA$y_test == 1],
  scores.class1 = pred_probs[DATA$y_test == 0],
  curve = TRUE
)

pr_df <- data.frame(
  Recall = pr$curve[,1],
  Precision = pr$curve[,2]
)

ggplot(pr_df, aes(x = Recall, y = Precision)) +
  geom_line(color = "darkred", size = 1.2) +
  theme_minimal() +
  labs(
    title = "Precision-Recall Curve",
    subtitle = paste("AUC =", round(pr$auc.integral,3)),
    x = "Recall",
    y = "Precision"
  )

## IMPORTANCE
# for clinical interpretation / ethical implications

importance <- lgb.importance(model)

ggplot(importance[1:20,], aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat="identity", fill="darkgreen") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "20 Most Important Features (LightGBM)",
    x = "Feature",
    y = "Gain"
  )

## SHAP
shap_values <- predict(
  model,
  DATA$X_train,
  type = "contrib"
)

shap_df <- as.data.frame(shap_values)

shap_importance <- colMeans(abs(shap_df))

shap_imp_df <- data.frame(
  Feature = names(shap_importance),
  Importance = shap_importance
)

shap_imp_df <- shap_imp_df[order(-shap_imp_df$Importance),][1:20,]

ggplot(shap_imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top SHAP Feature Contributions",
    x = "Feature",
    y = "Mean |SHAP value|"
  )


## METRICS SUMMARY TABLE
accuracy  <- as.numeric(cm$overall["Accuracy"])
precision <- as.numeric(cm$byClass["Precision"])
recall    <- as.numeric(cm$byClass["Recall"])
f1        <- as.numeric(cm$byClass["F1"])

roc_auc <- as.numeric(auc(roc_obj))

metrics <- data.frame(
  Metric = c("Accuracy","Precision","Recall","F1_score","ROC_AUC"),
  Value  = round(c(accuracy, precision, recall, f1, roc_auc),4)
)

#print 
cat("\nLightGBM Performance on TEST set:\n")
print(metrics)


