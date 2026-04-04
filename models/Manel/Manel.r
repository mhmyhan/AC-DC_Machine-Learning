# =========================================================
# Author: Manel
# Role: Modeling (SVM)
# Version: v2 
# =========================================================

library(tidyverse)
library(tidymodels)
library(tune)

if (!requireNamespace("kernlab", quietly = TRUE)) install.packages("kernlab")
library(kernlab)

if (!requireNamespace("themis", quietly = TRUE)) install.packages("themis")
library(themis)

# ---- Evidence (Manel/evidence) ----
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
  existing <- read_csv(RUN_SUMMARY, show_col_types = FALSE) %>%
    mutate(
      seed = suppressWarnings(as.numeric(seed)),
      metric_value = suppressWarnings(as.numeric(metric_value))
    )
  
  new_row <- tibble(
    run_id = as.character(run_id),
    datetime_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    author = "Manel",
    script = "models/Manel/Manel.r",
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

# ---- Config ----
SEED <- 20260302
set.seed(SEED)

TARGET_NAME <- "Relapse Free Status (Recurred vs Not_Recurred)"

# ---- Load group split ----
X_train <- read.csv("Data/Training/X_train.csv")
X_test  <- read.csv("Data/Training/X_test.csv")
y_train <- read.csv("Data/Training/y_train.csv")[[1]]
y_test  <- read.csv("Data/Training/y_test.csv")[[1]]

# ---- Sanity checks ----
stopifnot(nrow(X_train) == length(y_train))
stopifnot(nrow(X_test)  == length(y_test))

# ---- Outcome as factor ----
y_train <- as.factor(y_train)
y_test  <- as.factor(y_test)

cat("Class balance (train):\n")
print(table(y_train))

# ---- Ensure numeric predictors ----
X_train <- X_train %>% mutate(across(everything(), ~ as.numeric(.x)))
X_test  <- X_test  %>% mutate(across(everything(), ~ as.numeric(.x)))

# ---- Build data frames ----
train_df <- bind_cols(tibble(y = y_train), as_tibble(X_train))
test_df  <- bind_cols(tibble(y = y_test),  as_tibble(X_test))

# ---- Set class order explicitly ----
# First level = Not_Recurred
# Second level = Recurred
# We will use event_level = "second" everywhere relevant
train_df <- train_df %>% mutate(y = fct_relevel(y, "Not_Recurred", "Recurred"))
test_df  <- test_df  %>% mutate(y = fct_relevel(y, "Not_Recurred", "Recurred"))

cat("\nOutcome levels:\n")
print(levels(train_df$y))

# ---- Recipe: train-only imputation + scaling ----
rec <- recipe(y ~ ., data = train_df) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ---- Recipe ----
rec <- recipe(y ~ ., data = train_df) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_upsample(y)

# ---- Model ----
svm_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab", prob.model = TRUE) %>%
  set_mode("classification")

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(svm_spec)

# ---- CV ----
folds <- vfold_cv(train_df, v = 5, strata = y)

grid <- grid_regular(
  cost(range = c(-3, 6)),
  rbf_sigma(range = c(-6, 0)),
  levels = 4
)

tuned <- tune_grid(
  wf,
  resamples = folds,
  grid = grid,
  metrics = metric_set(roc_auc, pr_auc, accuracy, bal_accuracy),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

collect_metrics(tuned) %>%
  write_csv(file.path(EVID_DIR, "tables", "svm_rbf_tune_results_improved.csv"))

best <- select_best(tuned, metric = "pr_auc")
if (nrow(best) == 0) best <- select_best(tuned, metric = "roc_auc")

final_wf  <- finalize_workflow(wf, best)
final_fit <- fit(final_wf, data = train_df)

test_pred <- bind_cols(
  test_df %>% select(y),
  predict(final_fit, test_df, type = "class"),
  predict(final_fit, test_df, type = "prob")
)

metrics_test <- bind_rows(
  roc_auc(test_pred, truth = y, .pred_Recurred, event_level = "second"),
  pr_auc(test_pred, truth = y, .pred_Recurred, event_level = "second"),
  accuracy(test_pred, truth = y, estimate = .pred_class),
  bal_accuracy(test_pred, truth = y, estimate = .pred_class),
  precision(test_pred, truth = y, estimate = .pred_class, event_level = "second"),
  sens(test_pred, truth = y, estimate = .pred_class, event_level = "second"),
  f_meas(test_pred, truth = y, estimate = .pred_class, event_level = "second")
)

print(metrics_test)
write_csv(metrics_test, file.path(EVID_DIR, "tables", "svm_rbf_metrics_test_improved.csv"))

# ---- Pick best params ----
best <- select_best(tuned, metric = "roc_auc")
if (nrow(best) == 0) best <- select_best(tuned, metric = "pr_auc")

final_wf  <- finalize_workflow(wf, best)
final_fit <- fit(final_wf, data = train_df)

# ---- Evaluate on TEST ----
test_pred <- tryCatch({
  p_class <- predict(final_fit, test_df, type = "class")
  p_prob  <- predict(final_fit, test_df, type = "prob")
  bind_cols(test_df %>% select(y), p_class, p_prob)
}, error = function(e) {
  p_class <- predict(final_fit, test_df, type = "class")
  bind_cols(test_df %>% select(y), p_class)
})

cat("\nPrediction columns:\n")
print(names(test_pred))

# ---- Metrics on TEST ----
if (".pred_Recurred" %in% names(test_pred)) {
  
  roc_auc_tbl <- roc_auc(
    test_pred,
    truth = y,
    .pred_Recurred,
    event_level = "second"
  )
  
  pr_auc_tbl <- pr_auc(
    test_pred,
    truth = y,
    .pred_Recurred,
    event_level = "second"
  )
  
  acc_tbl <- accuracy(
    test_pred,
    truth = y,
    estimate = .pred_class
  )
  
  bal_acc_tbl <- bal_accuracy(
    test_pred,
    truth = y,
    estimate = .pred_class
  )
  
  f1_tbl <- f_meas(
    test_pred,
    truth = y,
    estimate = .pred_class,
    event_level = "second"
  )
  
  sens_tbl <- sens(
    test_pred,
    truth = y,
    estimate = .pred_class,
    event_level = "second"
  )
  
  spec_tbl <- spec(
    test_pred,
    truth = y,
    estimate = .pred_class,
    event_level = "second"
  )
  
  metrics_test <- bind_rows(
    roc_auc_tbl,
    pr_auc_tbl,
    acc_tbl,
    bal_acc_tbl,
    f1_tbl,
    sens_tbl,
    spec_tbl
  )
  
} else {
  acc_tbl <- accuracy(
    test_pred,
    truth = y,
    estimate = .pred_class
  )
  
  bal_acc_tbl <- bal_accuracy(
    test_pred,
    truth = y,
    estimate = .pred_class
  )
  
  f1_tbl <- f_meas(
    test_pred,
    truth = y,
    estimate = .pred_class,
    event_level = "second"
  )
  
  sens_tbl <- sens(
    test_pred,
    truth = y,
    estimate = .pred_class,
    event_level = "second"
  )
  
  spec_tbl <- spec(
    test_pred,
    truth = y,
    estimate = .pred_class,
    event_level = "second"
  )
  
  metrics_test <- bind_rows(
    acc_tbl,
    bal_acc_tbl,
    f1_tbl,
    sens_tbl,
    spec_tbl
  )
}

metrics_file <- file.path(EVID_DIR, "tables", "svm_rbf_metrics_test.csv")
metrics_test %>% write_csv(metrics_file)

cat("\nTest metrics:\n")
print(metrics_test)

# ---- Confusion matrix plot ----
cm <- conf_mat(test_pred, truth = y, estimate = .pred_class)
cm_df <- as.data.frame(cm$table)

if (!("n" %in% names(cm_df))) {
  count_col <- setdiff(names(cm_df), c("Truth", "Prediction"))[1]
  cm_df <- cm_df %>% rename(n = all_of(count_col))
}

cm_plot <- ggplot(cm_df, aes(x = Prediction, y = Truth, fill = n)) +
  geom_tile() +
  geom_text(aes(label = n), size = 7) +
  labs(
    title = "SVM RBF — Confusion Matrix (Test)",
    x = "Prediction",
    y = "Truth"
  ) +
  theme_minimal(base_size = 16)

cm_file <- file.path(EVID_DIR, "figs", "svm_rbf_confusion_matrix_test.png")
ggsave(cm_file, cm_plot, width = 6, height = 4, dpi = 200)

# ---- ROC curve if probabilities exist ----
roc_file <- NA_character_

if (".pred_Recurred" %in% names(test_pred)) {
  roc_df <- roc_curve(
    test_pred,
    truth = y,
    .pred_Recurred,
    event_level = "second"
  )
  
  roc_plot <- autoplot(roc_df) +
    ggtitle("SVM RBF — ROC Curve (Test)") +
    theme_minimal(base_size = 16)
  
  roc_file <- file.path(EVID_DIR, "figs", "svm_rbf_roc_curve_test.png")
  ggsave(roc_file, roc_plot, width = 6, height = 4, dpi = 200)
}

# ---- Optional PR curve if probabilities exist ----
pr_file <- NA_character_

if (".pred_Recurred" %in% names(test_pred)) {
  pr_df <- pr_curve(
    test_pred,
    truth = y,
    .pred_Recurred,
    event_level = "second"
  )
  
  pr_plot <- autoplot(pr_df) +
    ggtitle("SVM RBF — PR Curve (Test)") +
    theme_minimal(base_size = 16)
  
  pr_file <- file.path(EVID_DIR, "figs", "svm_rbf_pr_curve_test.png")
  ggsave(pr_file, pr_plot, width = 6, height = 4, dpi = 200)
}

# ---- Audit log row ----
primary <- metrics_test %>%
  filter(.metric %in% c("roc_auc", "pr_auc")) %>%
  slice(1)

if (nrow(primary) == 0) {
  primary <- metrics_test %>%
    filter(.metric == "bal_accuracy") %>%
    slice(1)
}

append_run(
  run_id = "RUN-ML06-SVM-002",
  target = TARGET_NAME,
  model = "SVM_RBF (kernlab) — group split",
  seed = SEED,
  primary_metric = primary$.metric[1],
  metric_value = as.numeric(primary$.estimate[1]),
  notes = ifelse(
    !is.na(roc_file),
    "5-fold CV tuning on train only; test evaluated once; median impute + normalise; event_level fixed to second; ROC and PR exported.",
    "5-fold CV tuning on train only; test evaluated once; median impute + normalise; event_level fixed to second; probabilities unavailable."
  ),
  artefacts = paste(
    na.omit(c(
      SESSIONINFO_FILE,
      tune_file,
      metrics_file,
      cm_file,
      roc_file,
      pr_file
    )),
    collapse = ";"
  )
)

cat("\nDone: SVM trained + evaluated.\n")
cat("Evidence folder: ", EVID_DIR, "\n")
cat("Tune: ", tune_file, "\n")
cat("Metrics: ", metrics_file, "\n")
cat("CM: ", cm_file, "\n")
if (!is.na(roc_file)) cat("ROC: ", roc_file, "\n")
if (!is.na(pr_file)) cat("PR: ", pr_file, "\n")
# ---- Simple summary block for report/demo ----

# Check raw NA values before recipe preprocessing
na_train_total <- sum(is.na(X_train))
na_test_total  <- sum(is.na(X_test))

na_status <- if ((na_train_total + na_test_total) == 0) "Clean" else "Present"

# Compute metrics
acc_val <- accuracy(test_pred, truth = y, estimate = .pred_class)$.estimate

prec_val <- precision(
  test_pred,
  truth = y,
  estimate = .pred_class,
  event_level = "second"
)$.estimate

recall_val <- sens(
  test_pred,
  truth = y,
  estimate = .pred_class,
  event_level = "second"
)$.estimate

f1_val <- f_meas(
  test_pred,
  truth = y,
  estimate = .pred_class,
  event_level = "second"
)$.estimate

roc_val <- if (".pred_Recurred" %in% names(test_pred)) {
  roc_auc(
    test_pred,
    truth = y,
    .pred_Recurred,
    event_level = "second"
  )$.estimate
} else {
  NA_real_
}

# Build neat summary table
model_summary <- tibble(
  Metric = c("N/A values", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"),
  Value = c(
    na_status,
    sprintf("%.4f", acc_val),
    sprintf("%.4f", prec_val),
    sprintf("%.4f", recall_val),
    sprintf("%.4f", f1_val),
    ifelse(is.na(roc_val), "Unavailable", sprintf("%.4f", roc_val))
  )
)

print(model_summary)

# Save it
summary_file <- file.path(EVID_DIR, "tables", "svm_rbf_model_summary.csv")
write_csv(model_summary, summary_file)

# ---- Simple summary block for report/demo ----

# Check raw NA values before recipe preprocessing
na_train_total <- sum(is.na(X_train))
na_test_total  <- sum(is.na(X_test))

na_status <- if ((na_train_total + na_test_total) == 0) "Clean" else "Present"

# Compute metrics
acc_val <- accuracy(test_pred, truth = y, estimate = .pred_class)$.estimate

prec_val <- precision(
  test_pred,
  truth = y,
  estimate = .pred_class,
  event_level = "second"
)$.estimate

recall_val <- sens(
  test_pred,
  truth = y,
  estimate = .pred_class,
  event_level = "second"
)$.estimate

f1_val <- f_meas(
  test_pred,
  truth = y,
  estimate = .pred_class,
  event_level = "second"
)$.estimate

roc_val <- if (".pred_Recurred" %in% names(test_pred)) {
  roc_auc(
    test_pred,
    truth = y,
    .pred_Recurred,
    event_level = "second"
  )$.estimate
} else {
  NA_real_
}

# Build neat summary table
model_summary <- tibble(
  Metric = c("N/A values", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"),
  Value = c(
    na_status,
    sprintf("%.4f", acc_val),
    sprintf("%.4f", prec_val),
    sprintf("%.4f", recall_val),
    sprintf("%.4f", f1_val),
    ifelse(is.na(roc_val), "Unavailable", sprintf("%.4f", roc_val))
  )
)

print(model_summary)

# Save it
summary_file <- file.path(EVID_DIR, "tables", "svm_rbf_model_summary.csv")
write_csv(model_summary, summary_file)



library(tidyverse)
library(readr)

df <- read_csv("data/Processed/metabric_clean.csv")

# Flag MTS rows
df_check <- df %>%
  mutate(has_MTS = grepl("MTS", `Patient ID`))

# 1) How many are there?
df_check %>%
  count(has_MTS)

# 2) Relapse distribution in MTS vs non-MTS
df_check %>%
  count(has_MTS, `Relapse Free Status`) %>%
  group_by(has_MTS) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

# 3) Overall missingness per row
df_check <- df_check %>%
  mutate(
    row_missing_n = rowSums(across(everything(), ~ is.na(.) | . == "")),
    row_missing_pct = row_missing_n / ncol(df_check)
  )

df_check %>%
  group_by(has_MTS) %>%
  summarise(
    n = n(),
    avg_missing_n = mean(row_missing_n),
    median_missing_n = median(row_missing_n),
    avg_missing_pct = mean(row_missing_pct)
  )

# 4) Missingness in key columns
df_check %>%
  group_by(has_MTS) %>%
  summarise(
    missing_target = mean(is.na(`Relapse Free Status`) | `Relapse Free Status` == ""),
    missing_tumor_stage = mean(is.na(`Tumor Stage`) | `Tumor Stage` == ""),
    missing_cellularity = mean(is.na(Cellularity) | Cellularity == ""),
    missing_chemo = mean(is.na(Chemotherapy) | Chemotherapy == ""),
    missing_pr = mean(is.na(`PR Status`) | `PR Status` == "")
  )

# 5) Show the MTS rows
df_check %>%
  filter(has_MTS) %>%
  select(`Patient ID`, `Relapse Free Status`, `Tumor Stage`, Cellularity, Chemotherapy, `PR Status`) %>%
  print(n = 50)