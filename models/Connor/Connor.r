# Author: Connor 
# Role: Modeling (Random Forest)
# Version: Image Saving + Metrics + Plots

library(randomForest)
library(pROC)
library(readr)
library(ggplot2)
library(caret)

set.seed(42)

data_dir <- "Data/Training"
output_dir <- "models/Connor"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Load data

X_train <- read_csv(file.path(data_dir, "X_train.csv"), show_col_types = FALSE)
X_test  <- read_csv(file.path(data_dir, "X_test.csv"),  show_col_types = FALSE)

y_train <- read_csv(file.path(data_dir, "y_train.csv"), show_col_types = FALSE)$y
y_test  <- read_csv(file.path(data_dir, "y_test.csv"),  show_col_types = FALSE)$y

# Clean column names
colnames(X_train) <- make.names(colnames(X_train))
colnames(X_test)  <- make.names(colnames(X_test))

# Factor encoding
y_train <- as.factor(y_train)
y_test  <- factor(y_test, levels = levels(y_train))

# Combine
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

y_pred <- predict(rf_model, newdata = test_data)

y_prob <- predict(rf_model, newdata = test_data, type = "prob")[,2]

# Metrics

accuracy  <- mean(y_pred == y_test)

precision <- sum(y_pred == 1 & y_test == 1) / sum(y_pred == 1)

recall    <- sum(y_pred == 1 & y_test == 1) / sum(y_test == 1)

f1_score  <- 2 * precision * recall / (precision + recall)

roc_auc   <- as.numeric(roc(y_test, y_prob)$auc)

metrics <- c(
  Accuracy  = accuracy,
  Precision = precision,
  Recall    = recall,
  F1_score  = f1_score,
  ROC_AUC   = roc_auc
)

cat("\nRandom Forest Performance:\n")
print(metrics)

metrics_df <- data.frame(
  Metric = names(metrics),
  Value  = as.numeric(metrics)
)

write.csv(
  metrics_df,
  file.path(output_dir, "rf_metrics.csv"),
  row.names = FALSE
)

# Confusion Matrix

conf_mat <- table(Predicted = y_pred, Actual = y_test)

print(conf_mat)

conf_df <- as.data.frame(conf_mat)

p_conf <- ggplot(conf_df,
                 aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "steelblue", high = "darkred") +
  theme_minimal() +
  ggtitle("Random Forest Confusion Matrix")

ggsave(
  file.path(output_dir, "rf_confusion_matrix.png"),
  p_conf,
  width = 6,
  height = 5
)

# ROC Curve

roc_obj <- roc(y_test, y_prob)

png(file.path(output_dir, "rf_roc_curve.png"),
    width = 800,
    height = 600)

plot(roc_obj,
     col = "blue",
     lwd = 3,
     main = "Random Forest ROC Curve")

dev.off()

# Feature Importance

importance_df <- as.data.frame(importance(rf_model))
importance_df$Feature <- rownames(importance_df)

importance_df <- importance_df[
  order(importance_df$MeanDecreaseGini,
        decreasing = TRUE),
]

top_imp <- importance_df[1:20, ]

p_imp <- ggplot(
  top_imp,
  aes(
    x = reorder(Feature, MeanDecreaseGini),
    y = MeanDecreaseGini
  )
) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top 20 Feature Importance (Random Forest)",
    x = "Feature",
    y = "MeanDecreaseGini"
  )

ggsave(
  file.path(output_dir, "rf_feature_importance.png"),
  p_imp,
  width = 7,
  height = 6
)