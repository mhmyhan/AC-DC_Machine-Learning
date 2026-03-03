
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
y_train <- as.factor(y_train)
levels_ref <- levels(y_train)

y_test  <- factor(y_test, levels = levels_ref)

y_train <- as.numeric(y_train) - 1
y_test  <- as.numeric(y_test) - 1

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
set.seed(42)
train_idx <- sample(seq_len(nrow(X_train_mat)), size = 0.8 * nrow(X_train_mat))

dtrain_inner <- xgb.DMatrix(data = X_train_mat[train_idx, ], 
                            label = y_train[train_idx])

dval <- xgb.DMatrix(data = X_train_mat[-train_idx, ], 
                    label = y_train[-train_idx])

watchlist <- list(train = dtrain_inner, val = dval)

xgb_model <- xgb.train(
  params = params,
  data = dtrain_inner,
  nrounds = 1000,
  watchlist = watchlist,
  early_stopping_rounds = 50,
  verbose = 1
)

# Evaluate
thresholds <- seq(0.2, 0.7, by = 0.01)

results <- sapply(thresholds, function(t) {
  pred <- ifelse(y_prob >= t, 1, 0)
  precision <- sum(pred == 1 & y_test == 1) / sum(pred == 1)
  recall <- sum(pred == 1 & y_test == 1) / sum(y_test == 1)
  if ((precision + recall) == 0) return(0)
  2 * precision * recall / (precision + recall)
})

best_t <- thresholds[which.max(results)]
best_t

y_prob <- predict(xgb_model, dtest)
y_pred <- ifelse(y_prob >= best_t, 1, 0)

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

#  Visualizations

library(ggplot2)
library(xgboost)
library(pROC)

#  Confusion Matrix
conf_mat <- table(Predicted = y_pred, Actual = y_test)
cat("\nConfusion Matrix:\n")
print(conf_mat)

# Optionally, visualize it
conf_df <- as.data.frame(conf_mat)
ggplot(conf_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "steelblue", high = "darkred") +
  theme_minimal() +
  ggtitle("Confusion Matrix Heatmap")

# ROC Curve
roc_obj <- roc(y_test, y_prob)
plot(roc_obj, col = "blue", lwd = 2, main = "ROC Curve")
auc_val <- auc(roc_obj)
cat("\nROC AUC:", auc_val, "\n")

#  Feature Importance
# Get importance matrix
importance_matrix <- xgb.importance(model = xgb_model)

# View top features
head(importance_matrix, 10)

# Plot importance
xgb.plot.importance(importance_matrix[1:20,], main = "Top 20 Feature Importance")

# -------------------------------
# Optional: Save model
# -------------------------------
# xgb.save(xgb_model, "models/xgb_model.model")
