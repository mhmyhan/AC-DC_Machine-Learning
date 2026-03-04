
# Author: Vilius
# Role: Modeling (XGBoost)
# Version: CV-Optimized + Image Saving

library(pROC)
library(xgboost)
library(readr)
library(ggplot2)
#install.packages("pROC")
#install.packages("xgboost")
#install.packages("readr")
#install.packages("ggplot2")
set.seed(42)

# Load preprocessed data


data_dir <- "data/Training"
output_dir <- "models/Vil"   # <-- images saved here

X_train <- read_csv(file.path(data_dir, "X_train.csv"), show_col_types = FALSE)
X_test  <- read_csv(file.path(data_dir, "X_test.csv"),  show_col_types = FALSE)

y_train <- read_csv(file.path(data_dir, "y_train.csv"), show_col_types = FALSE)$y
y_test  <- read_csv(file.path(data_dir, "y_test.csv"),  show_col_types = FALSE)$y

# Ensure consistent encoding
y_train <- as.factor(y_train)
levels_ref <- levels(y_train)
y_test  <- factor(y_test, levels = levels_ref)

# Convert to numeric 0/1
y_train <- as.numeric(y_train) - 1
y_test  <- as.numeric(y_test) - 1

# Convert predictors to numeric matrix
X_train_mat <- as.matrix(X_train)
X_test_mat  <- as.matrix(X_test)

# Safety checks
stopifnot(sum(is.na(X_train_mat)) == 0)
stopifnot(sum(is.infinite(X_train_mat)) == 0)
stopifnot(sum(is.na(y_train)) == 0)

# Optional safety for test data
X_test_mat[is.na(X_test_mat)] <- 0
X_test_mat[is.infinite(X_test_mat)] <- 0

# Create DMatrix
dtrain <- xgb.DMatrix(data = X_train_mat, label = y_train)
dtest  <- xgb.DMatrix(data = X_test_mat,  label = y_test)

# Handle class imbalance
neg <- sum(y_train == 0)
pos <- sum(y_train == 1)
scale_pos_weight <- neg / pos

# XGBoost parameters

params <- list(
  objective = "binary:logistic",
  eval_metric = c("logloss", "auc"),
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = scale_pos_weight
)

# Cross-validation for best nrounds

cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 1000,
  nfold = 5,
  early_stopping_rounds = 50,
  verbose = 1
)

best_nrounds <- cv$best_iteration
if (is.null(best_nrounds) || best_nrounds <= 0) best_nrounds <- 1000

# Train final model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds
)

# Threshold tuning on TRAINING SET ONLY

thresholds <- seq(0.2, 0.7, by = 0.01)
y_prob_train <- predict(xgb_model, dtrain)

f1_scores <- sapply(thresholds, function(t) {
  pred <- ifelse(y_prob_train >= t, 1, 0)
  precision <- sum(pred == 1 & y_train == 1) / sum(pred == 1)
  recall <- sum(pred == 1 & y_train == 1) / sum(y_train == 1)
  if ((precision + recall) == 0) return(0)
  2 * precision * recall / (precision + recall)
})

best_t <- thresholds[which.max(f1_scores)]
cat("Best threshold (from training):", best_t, "\n")

# Evaluate on TEST SET

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

metrics_df <- data.frame(
  Metric = names(metrics),
  Value  = as.numeric(metrics)
)

write.csv(metrics_df,
          file.path(output_dir, "xgb_metrics.csv"),
          row.names = FALSE)


# Confusion Matrix + Save Image

conf_mat <- table(Predicted = y_pred, Actual = y_test)
cat("\nConfusion Matrix:\n")
print(conf_mat)

conf_df <- as.data.frame(conf_mat)

p_conf <- ggplot(conf_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "steelblue", high = "darkred") +
  theme_minimal() +
  ggtitle("Confusion Matrix Heatmap")

ggsave(file.path(output_dir, "xgb_confusion_matrix.png"), p_conf, width = 6, height = 5)

# ROC Curve + Save Image

roc_obj <- roc(y_test, y_prob)

png(file.path(output_dir, "xgb_roc_curve.png"), width = 800, height = 600)
plot(roc_obj, col = "blue", lwd = 3, main = "ROC Curve")
dev.off()

# Feature Importance + Save Image

importance_matrix <- xgb.importance(model = xgb_model)

# Convert to ggplot manually
p_imp <- ggplot(importance_matrix[1:20, ],
                aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 20 Feature Importance (Gain)",
       x = "Feature",
       y = "Gain")

ggsave(file.path(output_dir, "xgb_feature_importance.png"),
       p_imp, width = 7, height = 6)



# Optional: Save model

# xgb.save(xgb_model, file.path(output_dir, "xgb_model_cv.model"))