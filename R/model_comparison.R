# =========================================================
# Script: model_comparison.R
# Purpose: Compare team model outputs
# =========================================================

library(tidyverse)

# Helper: read a simple Metric/Value CSV
read_metric_summary <- function(path, model_name) {
  if (!file.exists(path)) {
    return(tibble(
      model = model_name,
      Accuracy = NA_real_,
      Precision = NA_real_,
      Recall = NA_real_,
      F1 = NA_real_,
      `ROC-AUC` = NA_real_,
      source_status = "missing_file"
    ))
  }
  
  df <- read_csv(path, show_col_types = FALSE)
  
  if (!all(c("Metric", "Value") %in% names(df))) {
    return(tibble(
      model = model_name,
      Accuracy = NA_real_,
      Precision = NA_real_,
      Recall = NA_real_,
      F1 = NA_real_,
      `ROC-AUC` = NA_real_,
      source_status = "wrong_format"
    ))
  }
  
  df <- df %>%
    mutate(
      Metric = case_when(
        Metric == "F1_score" ~ "F1",
        Metric == "ROC_AUC" ~ "ROC-AUC",
        TRUE ~ Metric
      ),
      Value = suppressWarnings(as.numeric(Value))
    )
  
  out <- df %>%
    select(Metric, Value) %>%
    pivot_wider(names_from = Metric, values_from = Value)
  
  for (nm in c("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")) {
    if (!nm %in% names(out)) out[[nm]] <- NA_real_
  }
  
  out %>%
    mutate(model = model_name, source_status = "loaded") %>%
    select(model, Accuracy, Precision, Recall, F1, `ROC-AUC`, source_status)
}

# ---------- Manel SVM ----------
manel <- read_metric_summary(
  "models/Manel/evidence/tables/svm_rbf_model_summary.csv",
  "SVM RBF (Manel)"
)

# ---------- Vilius XGBoost ----------
vilius <- read_metric_summary(
  "models/Vil/xgb_metrics.csv",
  "XGBoost (Vilius)"
)

# ---------- Max LightGBM ----------
# No metrics CSV yet, so created a placeholder
max_model <- tibble(
  model = "Max model",
  Accuracy = NA_real_,
  Precision = NA_real_,
  Recall = NA_real_,
  F1 = NA_real_,
  `ROC-AUC` = NA_real_,
  source_status = "images_only_no_csv"
)

# ---------- Connor ----------
#same for connor
connor <- tibble(
  model = "Connor model",
  Accuracy = NA_real_,
  Precision = NA_real_,
  Recall = NA_real_,
  F1 = NA_real_,
  `ROC-AUC` = NA_real_,
  source_status = "not_submitted_yet"
)

# ---------- Combine ----------
comparison_table <- bind_rows(
  manel,
  vilius,
  max_model,
  connor
)

print(comparison_table)

dir.create("evidence/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("evidence/figs", recursive = TRUE, showWarnings = FALSE)

write_csv(
  comparison_table,
  "evidence/tables/final_model_comparison.csv"
)

# Plot only models with numeric metrics 
plot_df <- comparison_table %>%
  filter(!if_all(c(Accuracy, Precision, Recall, F1), is.na)) %>%
  select(model, Accuracy, Precision, Recall, F1) %>%
  pivot_longer(-model, names_to = "metric", values_to = "value")

if (nrow(plot_df) > 0) {
  p <- ggplot(plot_df, aes(x = model, y = value, fill = metric)) +
    geom_col(position = "dodge") +
    theme_minimal(base_size = 14) +
    labs(
      title = "Team Model Comparison",
      x = "Model",
      y = "Score"
    )
  
  ggsave(
    "evidence/figs/final_model_comparison.png",
    p,
    width = 8,
    height = 5,
    dpi = 200
  )
}