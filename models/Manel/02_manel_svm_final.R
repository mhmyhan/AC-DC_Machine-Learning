# =========================================================
# Author: Manel
# Role: Modeling (SVM)
# Script: 03_manel_svm_final.R
# Version: v1.0
# DateTime (UTC): 2026-04-04T00:00:00Z
# Educational-only; results are not validated for clinical use.
# =========================================================

library(tidyverse)
library(tidymodels)
library(tune)
library(kernlab)
library(themis)
library(vip)

# =========================================================
# 1) Reproducibility + folders
# =========================================================
SEED <- 20260404
set.seed(SEED)

EVID_DIR <- file.path("models", "Manel", "evidence")
dir.create(file.path(EVID_DIR, "logs"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(EVID_DIR, "tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(EVID_DIR, "figs"), recursive = TRUE, showWarnings = FALSE)

SESSIONINFO_FILE <- file.path(EVID_DIR, "logs", "sessionInfo.txt")
writeLines(capture.output(sessionInfo()), SESSIONINFO_FILE)

RUN_SUMMARY <- file.path(EVID_DIR, "logs", "run_summary.csv")

if (!file.exists(RUN_SUMMARY)) {
  tibble(
    run_id = character(),
    datetime_utc = character(),
    author = character(),
    script = character(),
    target = character(),
    model = character(),
    seed = double(),
    primary_metric = character(),
    metric_value = double(),
    notes = character(),
    artefacts = character()
  ) %>% write_csv(RUN_SUMMARY)
}

append_run <- function(run_id, target, model, seed, primary_metric, metric_value, notes, artefacts) {
  existing <- read_csv(RUN_SUMMARY, show_col_types = FALSE)
  
  new_row <- tibble(
    run_id = as.character(run_id),
    datetime_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    author = "Manel",
    script = "R/03_manel_svm_final.R",
    target = as.character(target),
    model = as.character(model),
    seed = as.numeric(seed),
    primary_metric = as.character(primary_metric),
    metric_value = as.numeric(metric_value),
    notes = as.character(notes),
    artefacts = as.character(artefacts)
  )
  
  bind_rows(existing, new_row) %>% write_csv(RUN_SUMMARY)
}

# =========================================================
# 2) Load shared team split
# =========================================================
TARGET_NAME <- "Relapse Free Status"

X_train <- read_csv("data/Training/X_train.csv", show_col_types = FALSE)
X_test  <- read_csv("data/Training/X_test.csv", show_col_types = FALSE)
y_train <- read_csv("data/Training/y_train.csv", show_col_types = FALSE) %>% pull(1)
y_test  <- read_csv("data/Training/y_test.csv", show_col_types = FALSE) %>% pull(1)

stopifnot(nrow(X_train) == length(y_train))
stopifnot(nrow(X_test)  == length(y_test))

# =========================================================
# 3) Outcome handling
# =========================================================
# Adjust labels if your shared preprocessing exported spaces instead of underscores
y_train <- as.character(y_train)
y_test  <- as.character(y_test)

y_train <- str_replace_all(y_train, " ", "_")
y_test  <- str_replace_all(y_test, " ", "_")

y_train <- factor(y_train, levels = c("Not_Recurred", "Recurred"))
y_test  <- factor(y_test, levels = c("Not_Recurred", "Recurred"))

cat("Outcome levels:\n")
print(levels(y_train))

cat("\nClass balance (train):\n")
print(table(y_train))

# =========================================================
# 4) Build train/test frames
# =========================================================
X_train <- X_train %>% mutate(across(everything(), as.numeric))
X_test  <- X_test  %>% mutate(across(everything(), as.numeric))

train_df <- bind_cols(tibble(y = y_train), X_train)
test_df  <- bind_cols(tibble(y = y_test),  X_test)

# =========================================================
# 5) Recipe
# =========================================================
rec <- recipe(y ~ ., data = train_df) %>%
  step_zv(all_predictors()) %>%  #to remove useless columns with the same info repeating
  step_impute_median(all_numeric_predictors()) %>% # median where there is na
  step_normalize(all_numeric_predictors()) %>% #svm likes normalised data esp with the big data i have in mine
  step_upsample(y) # svm would be confused by the big difference between relapse and non relapse as there is a big diff in number, so it duplicates relapse

# =========================================================
# 6) Tunable RBF SVM (I let the model find the best params  using cross-validation)
# =========================================================
svm_spec <- svm_rbf(
  cost = tune(),#Cost controls how much the model punishes errors of hyperplane and margins..
  rbf_sigma = tune() 
) %>%
  set_engine("kernlab", prob.model = TRUE) %>%
  set_mode("classification") #obv class problm

wf <- workflow() %>% 
  add_recipe(rec) %>%
  add_model(svm_spec)

# =========================================================
# 7) Cross-validation + tuning
# =========================================================
folds <- vfold_cv(train_df, v = 5, strata = y) #4 folds of training

grid <- grid_regular(
  cost(range = c(-3, 3)), 
  rbf_sigma(range = c(-4, 0)),
  levels = 4 #16 combinations so 80 model fits
)

tuned <- tune_grid(
  wf,
  resamples = folds,
  grid = grid,
  metrics = metric_set(roc_auc, pr_auc, accuracy, bal_accuracy),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

tune_file <- file.path(EVID_DIR, "tables", "svm_rbf_tune_results.csv")
collect_metrics(tuned) %>% write_csv(tune_file)

# Primary selection = PR-AUC, fallback = ROC-AUC
best <- select_best(tuned, metric = "pr_auc")
if (nrow(best) == 0) {
  best <- select_best(tuned, metric = "roc_auc")
}

best_params_file <- file.path(EVID_DIR, "tables", "svm_rbf_best_params.csv")
write_csv(best, best_params_file)

# =========================================================
# 8) Final fit on train
# =========================================================
final_wf  <- finalize_workflow(wf, best)
final_fit <- fit(final_wf, data = train_df)

# =========================================================
# 8b) Manual permutation feature importance for SVM
# =========================================================
set.seed(SEED)

# Baseline accuracy on untouched test set
baseline_pred <- predict(final_fit, test_df, type = "class")
baseline_acc <- mean(baseline_pred$.pred_class == test_df$y)

feature_names <- setdiff(names(test_df), "y")

perm_importance <- map_dfr(feature_names, function(feat) {
  temp_df <- test_df
  
  # Shuffle one feature only
  temp_df[[feat]] <- sample(temp_df[[feat]])
  
  # Predict again
  perm_pred <- predict(final_fit, temp_df, type = "class")
  perm_acc <- mean(perm_pred$.pred_class == temp_df$y)
  
  tibble(
    Feature = feat,
    Baseline_Accuracy = baseline_acc,
    Permuted_Accuracy = perm_acc,
    Importance = baseline_acc - perm_acc
  )
})

# Keep top 20 most important features
perm_top <- perm_importance %>%
  arrange(desc(Importance)) %>%
  slice_head(n = 20)

perm_file <- file.path(EVID_DIR, "tables", "svm_rbf_permutation_importance.csv")
write_csv(perm_top, perm_file)

perm_plot <- ggplot(perm_top, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "SVM RBF — Permutation Feature Importance",
    x = "Feature",
    y = "Drop in Accuracy After Permutation"
  ) +
  theme_minimal(base_size = 14)

perm_plot_file <- file.path(EVID_DIR, "figs", "svm_rbf_feature_importance.png")
ggsave(perm_plot_file, perm_plot, width = 8, height = 6, dpi = 200)

# =========================================================
# 9) Predict on test
# =========================================================
test_pred <- bind_cols(
  test_df %>% select(y),
  predict(final_fit, test_df, type = "class"),
  predict(final_fit, test_df, type = "prob") # gen predictions on test inc pred class and prob
)

cat("\nPrediction columns:\n")
print(names(test_pred))

# =========================================================
# 10) Metrics on test
# =========================================================
metrics_test <- bind_rows(
  roc_auc(test_pred, truth = y, .pred_Recurred, event_level = "second"),
  pr_auc(test_pred, truth = y, .pred_Recurred, event_level = "second"),
  accuracy(test_pred, truth = y, estimate = .pred_class),
  bal_accuracy(test_pred, truth = y, estimate = .pred_class),
  precision(test_pred, truth = y, estimate = .pred_class, event_level = "second"),
  sens(test_pred, truth = y, estimate = .pred_class, event_level = "second"),
  spec(test_pred, truth = y, estimate = .pred_class, event_level = "second"),
  f_meas(test_pred, truth = y, estimate = .pred_class, event_level = "second")
)

metrics_file <- file.path(EVID_DIR, "tables", "svm_rbf_metrics_test.csv")
write_csv(metrics_test, metrics_file)

cat("\nTest metrics:\n")
print(metrics_test)

# =========================================================
# 11) Confusion matrix
# =========================================================
cm <- conf_mat(test_pred, truth = y, estimate = .pred_class)
cm_df <- as.data.frame(cm$table)

if (!("n" %in% names(cm_df))) {
  count_col <- setdiff(names(cm_df), c("Truth", "Prediction"))[1]
  cm_df <- cm_df %>% rename(n = all_of(count_col))
}

cm_plot <- ggplot(cm_df, aes(x = Prediction, y = Truth, fill = n)) +
  geom_tile() +
  geom_text(aes(label = n), size = 6) +
  labs(
    title = "SVM RBF — Confusion Matrix (Test)",
    x = "Prediction",
    y = "Truth"
  ) +
  theme_minimal(base_size = 14)

cm_file <- file.path(EVID_DIR, "figs", "svm_rbf_confusion_matrix_test.png")
ggsave(cm_file, cm_plot, width = 6, height = 4, dpi = 200)

# =========================================================
# 12) ROC + PR curves
# =========================================================
roc_df <- roc_curve(test_pred, truth = y, .pred_Recurred, event_level = "second")
roc_plot <- autoplot(roc_df) +
  ggtitle("SVM RBF — ROC Curve (Test)") +
  theme_minimal(base_size = 14)

roc_file <- file.path(EVID_DIR, "figs", "svm_rbf_roc_curve_test.png")
ggsave(roc_file, roc_plot, width = 6, height = 4, dpi = 200)

pr_df <- pr_curve(test_pred, truth = y, .pred_Recurred, event_level = "second")
pr_plot <- autoplot(pr_df) +
  ggtitle("SVM RBF — PR Curve (Test)") +
  theme_minimal(base_size = 14)

pr_file <- file.path(EVID_DIR, "figs", "svm_rbf_pr_curve_test.png")
ggsave(pr_file, pr_plot, width = 6, height = 4, dpi = 200)

# =========================================================
# 13) Simple summary table
# =========================================================
na_train_total <- sum(is.na(X_train))
na_test_total  <- sum(is.na(X_test))
na_status <- if ((na_train_total + na_test_total) == 0) "Clean" else "Present"

acc_val <- accuracy(test_pred, truth = y, estimate = .pred_class)$.estimate
prec_val <- precision(test_pred, truth = y, estimate = .pred_class, event_level = "second")$.estimate
recall_val <- sens(test_pred, truth = y, estimate = .pred_class, event_level = "second")$.estimate
f1_val <- f_meas(test_pred, truth = y, estimate = .pred_class, event_level = "second")$.estimate
roc_val <- roc_auc(test_pred, truth = y, .pred_Recurred, event_level = "second")$.estimate
pr_val  <- pr_auc(test_pred, truth = y, .pred_Recurred, event_level = "second")$.estimate

model_summary <- tibble(
  Metric = c("N/A values", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"),
  Value = c(
    na_status,
    sprintf("%.4f", acc_val),
    sprintf("%.4f", prec_val),
    sprintf("%.4f", recall_val),
    sprintf("%.4f", f1_val),
    sprintf("%.4f", roc_val),
    sprintf("%.4f", pr_val)
  )
)

summary_file <- file.path(EVID_DIR, "tables", "svm_rbf_model_summary.csv")
write_csv(model_summary, summary_file)

print(model_summary)

# =========================================================
# 14) Append run log
# =========================================================
primary <- metrics_test %>%
  filter(.metric == "pr_auc") %>%
  slice(1)

if (nrow(primary) == 0) {
  primary <- metrics_test %>%
    filter(.metric == "roc_auc") %>%
    slice(1)
}

append_run(
  run_id = "RUN-ML06-SVM-FINAL-001",
  target = "Relapse Free Status (Recurred vs Not_Recurred)",
  model = "SVM_RBF (kernlab) — shared team split",
  seed = SEED,
  primary_metric = primary$.metric[1],
  metric_value = as.numeric(primary$.estimate[1]),
  notes = "5-fold CV on training only; PR-AUC primary; upsampling in recipe; positive class fixed as second level; ROC/PR/confusion matrix exported.",
  artefacts = paste(
    c(
      SESSIONINFO_FILE,
      tune_file,
      best_params_file,
      metrics_file,
      cm_file,
      roc_file,
      pr_file,
      summary_file
    ),
    collapse = ";"
  )
)

cat("\nDone: final SVM trained and evaluated.\n")
cat("Evidence folder: ", EVID_DIR, "\n")