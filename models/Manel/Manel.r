# =========================================================
# Author: Manel
# Role: Modeling (SVM)
# Version: v1 (i wanna be sure)
# =========================================================
# ---- Packages ----
library(tidyverse)
library(tidymodels)
library(tune)

# Ensure SVM engine exists (run once if needed)
if (!requireNamespace("kernlab", quietly = TRUE)) install.packages("kernlab")
library(kernlab)

# ---- Evidence location (Manel folder) ----
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
    seed = double(),              # <- keep numeric
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

# ---- Profiling NA (evidence only; does NOT change CSVs) ----
na_report <- tibble(
  feature  = names(X_train),
  na_train = colSums(is.na(X_train)),
  na_test  = colSums(is.na(X_test))
) %>%
  filter(na_train > 0 | na_test > 0) %>%
  arrange(desc(na_test), desc(na_train))

NA_FILE <- file.path(EVID_DIR, "tables", "na_profile_train_test.csv")
write.csv(na_report, NA_FILE, row.names = FALSE)

# ---- Sanity checks ----
stopifnot(nrow(X_train) == length(y_train))
stopifnot(nrow(X_test)  == length(y_test))

y_train <- as.factor(y_train)
y_test  <- as.factor(y_test)

cat("Class balance (train):\n")
print(table(y_train))

# Ensure numeric matrix (safe cast)
X_train <- X_train %>% mutate(across(everything(), ~ as.numeric(.x)))
X_test  <- X_test  %>% mutate(across(everything(), ~ as.numeric(.x)))

# ---- INF profiling (evidence only) ----
inf_report <- tibble(
  feature  = names(X_train),
  inf_train = colSums(is.infinite(as.matrix(X_train))),
  inf_test  = colSums(is.infinite(as.matrix(X_test)))
) %>%
  filter(inf_train > 0 | inf_test > 0) %>%
  arrange(desc(inf_test), desc(inf_train))

INF_FILE <- file.path(EVID_DIR, "tables", "inf_profile_train_test.csv")
write.csv(inf_report, INF_FILE, row.names = FALSE)

# ---- Convert Inf/-Inf to NA (in-memory only) ----
X_train[is.infinite(as.matrix(X_train))] <- NA
X_test[is.infinite(as.matrix(X_test))] <- NA

# ---- Build data frames ----
train_df <- bind_cols(tibble(y = y_train), as_tibble(X_train))
test_df  <- bind_cols(tibble(y = y_test),  as_tibble(X_test))

# Ensure positive class = Recurred
if ("Recurred" %in% levels(train_df$y)) {
  train_df <- train_df %>% mutate(y = fct_relevel(y, "Not_Recurred", "Recurred"))
  test_df  <- test_df  %>% mutate(y = fct_relevel(y, "Not_Recurred", "Recurred"))
}

# ---- Recipe: train-only imputation + scaling ----
rec <- recipe(y ~ ., data = train_df) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ---- Model: SVM RBF ----
svm_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab", prob.model = TRUE) %>%
  set_mode("classification")

wf <- workflow() %>% add_recipe(rec) %>% add_model(svm_spec)

# ---- 5-fold CV on TRAIN only ----
folds <- vfold_cv(train_df, v = 5, strata = y)

mset <- metric_set(roc_auc, pr_auc, accuracy, bal_accuracy, f_meas, sens, spec)

grid <- grid_regular(
  cost(range = c(-2, 4)),
  rbf_sigma(range = c(-5, -1)),
  levels = 5
)

tuned <- tune_grid(
  wf,
  resamples = folds,
  grid = grid,
  metrics = mset,
  control = control_grid(save_pred = TRUE)
)

tune_file <- file.path(EVID_DIR, "tables", "svm_rbf_tune_results.csv")
collect_metrics(tuned) %>% write_csv(tune_file)

best <- select_best(tuned, metric = "roc_auc")
if (nrow(best) == 0) best <- select_best(tuned, metric = "pr_auc")

final_wf <- finalize_workflow(wf, best)
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

metrics_test <- if (".pred_Recurred" %in% names(test_pred)) {
  mset(test_pred, truth = y, estimate = .pred_class, .pred_Recurred)
} else if (".pred_1" %in% names(test_pred)) {
  mset(test_pred, truth = y, estimate = .pred_class, .pred_1)
} else {
  mset(test_pred, truth = y, estimate = .pred_class)
}

metrics_file <- file.path(EVID_DIR, "tables", "svm_rbf_metrics_test.csv")
metrics_test %>% write_csv(metrics_file)

# ---- Confusion matrix plot ----
cm <- conf_mat(test_pred, truth = y, estimate = .pred_class)
cm_df <- as.data.frame(cm$table)

if (!("n" %in% names(cm_df))) {
  count_col <- setdiff(names(cm_df), c("Truth", "Prediction"))[1]
  cm_df <- cm_df %>% rename(n = all_of(count_col))
}

cm_plot <- ggplot(cm_df, aes(x = Prediction, y = Truth, fill = n)) +
  geom_tile() +
  geom_text(aes(label = n)) +
  labs(title = "SVM RBF — Confusion Matrix (Test)")

cm_file <- file.path(EVID_DIR, "figs", "svm_rbf_confusion_matrix_test.png")
ggsave(cm_file, cm_plot, width = 6, height = 4, dpi = 200)

# ---- ROC curve if probs exist ----
roc_file <- NA_character_
if (".pred_Recurred" %in% names(test_pred)) {
  roc_df <- roc_curve(test_pred, truth = y, .pred_Recurred)
  roc_plot <- autoplot(roc_df) + ggtitle("SVM RBF — ROC Curve (Test)")
  roc_file <- file.path(EVID_DIR, "figs", "svm_rbf_roc_curve_test.png")
  ggsave(roc_file, roc_plot, width = 6, height = 4, dpi = 200)
}

# ---- Audit log row (NOW CALLED) ----
primary <- metrics_test %>% filter(.metric %in% c("roc_auc", "pr_auc")) %>% slice(1)
if (nrow(primary) == 0) primary <- metrics_test %>% filter(.metric == "bal_accuracy") %>% slice(1)

append_run(
  run_id = "RUN-ML06-SVM-001",
  target = TARGET_NAME,
  model = "SVM_RBF (kernlab) — group split",
  seed = SEED,
  primary_metric = primary$.metric[1],
  metric_value = as.numeric(primary$.estimate[1]),
  notes = ifelse(!is.na(roc_file),
                 "5-fold CV tuning on train only; test evaluated once; median impute + normalise; ROC exported.",
                 "5-fold CV tuning on train only; test evaluated once; median impute + normalise; probabilities unavailable."),
  artefacts = paste(na.omit(c(SESSIONINFO_FILE, NA_FILE, INF_FILE, tune_file, metrics_file, cm_file, roc_file)), collapse = ";")
)

cat("\n one: SVM trained + evaluated.\n")
cat("Evidence folder: ", EVID_DIR, "\n")
cat("Tune: ", tune_file, "\n")
cat("Metrics: ", metrics_file, "\n")
cat("CM: ", cm_file, "\n")
if (!is.na(roc_file)) cat("ROC: ", roc_file, "\n")