library(tidyverse)
library(recipes)
library(rsample)
library(janitor)


TARGET <- "Relapse Free Status" # the binary one 

# Confirm working directory
getwd()

# Load data set
file_path <- "data/Processed/metabric_clean.csv"
df <- read_csv(file_path)

#audit MTS rows before exclusion

mts_audit <- df %>%
  mutate(has_MTS = grepl("MTS", `Patient ID`)) %>%
  mutate(
    row_missing_n = rowSums(across(everything(), ~ is.na(.) | . == "")),
    row_missing_pct = row_missing_n / ncol(.)
  )

mts_count <- mts_audit %>%
  count(has_MTS)

mts_relapse <- mts_audit %>%
  count(has_MTS, .data[[TARGET]]) %>%
  group_by(has_MTS) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

mts_missing_summary <- mts_audit %>%
  group_by(has_MTS) %>%
  summarise(
    n = n(),
    avg_missing_n = mean(row_missing_n),
    median_missing_n = median(row_missing_n),
    avg_missing_pct = mean(row_missing_pct),
    missing_target = mean(is.na(.data[[TARGET]]) | .data[[TARGET]] == ""),
    missing_tumor_stage = mean(is.na(`Tumor Stage`) | `Tumor Stage` == ""),
    missing_cellularity = mean(is.na(Cellularity) | Cellularity == ""),
    missing_chemo = mean(is.na(Chemotherapy) | Chemotherapy == ""),
    missing_pr = mean(is.na(`PR Status`) | `PR Status` == "")
  )

write_csv(mts_count, "evidence/tables/mts_count.csv")
write_csv(mts_relapse, "evidence/tables/mts_relapse_distribution.csv")
write_csv(mts_missing_summary, "evidence/tables/mts_missingness_summary.csv")

# Keep rows where "MTS" is NOT found in the ID column
df <- df[!grepl("MTS", df$`Patient ID`), ]

# Dropping columns

cols_to_drop <- c(
  "Patient ID", # Irrelevant to data
  "Cohort", # Exists for clinical bias purposes (Irrelevant to data processing)
  "Sex", # all female drop for sure
  "Overall Survival (Months)", # Irrelevant to calculating relapse
  "Relapse Free Status (Months)", # Irrelevant to calculating relapse
  "Overall Survival Status", # Temporal
  "Patient's Vital Status"   # Temporal
  )

  df <- df %>% select(-any_of(cols_to_drop))
  
  # Fix data types
  
  df <- df %>% 
    mutate(
      `Tumor Stage` = as.factor(`Tumor Stage`),
      `Neoplasm Histologic Grade` =as.factor(`Neoplasm Histologic Grade`),
      !!TARGET := as.factor(.data[[TARGET]])
    )
  
  # Cleaning
  df <- df %>%
    mutate(`Integrative Cluster` = recode(
      `Integrative Cluster`,
      "4ER+" = "X4ERplus",
      "4ER-" = "X4ERminus"
    ))
  
  # Generic factor cleaning (no make.unique)
  df <- df %>% 
    mutate(across(where(is.factor), ~ {
      x <- as.character(.)
      x <- gsub("\\+", "plus", x)
      x <- gsub("-", "minus", x)
      x <- gsub("/", "_", x)
      x <- gsub("\\s+", "_", x)
      x <- gsub("[^A-Za-z0-9_]", "", x)
      x <- ifelse(grepl("^[0-9]", x), paste0("X", x), x)
      factor(x)
    }))
  
  df <- df %>%
    filter(!is.na(.data[[TARGET]]))
  
  
  # Train Split
  
  set.seed(42)
  
  split <- initial_split(df, prop = 0.8, strata = TARGET)
  
  train_raw <- training(split)
  test_raw <- testing(split)
  
  # Check Class Balance

  
  # Shared recipe
  
  rec <- recipe(train_raw) %>%
    update_role(all_of(TARGET), new_role = "outcome") %>%
    update_role(-all_of(TARGET), new_role = "predictor") %>%
    
    step_zv(all_predictors()) %>%
    #Missing Values
    step_impute_median(all_numeric_predictors()) %>%
    step_impute_mode(all_nominal_predictors()) %>%

    # One-hot encode categorical variables 
    step_dummy(all_nominal_predictors(), one_hot = TRUE)
    
  # Prepare recipe
  prep_rec <- prep(rec, training = train_raw)

  train_processed <- bake(prep_rec, new_data = train_raw)
  test_processed  <- bake(prep_rec, new_data = test_raw)
  
  
  colnames(train_processed) <- as.character(colnames(train_processed))
  colnames(test_processed)  <- as.character(colnames(test_processed))
  

  #train_processed <- janitor::clean_names(train_processed, unique = TRUE)
  #test_processed  <- janitor::clean_names(test_processed, unique = TRUE)
  
  # separate x and y 
  
  y_train <- train_processed[[TARGET]]
  y_test  <- test_processed[[TARGET]]
  
  X_train <- train_processed %>% select(-all_of(TARGET))
  X_test  <- test_processed %>% select(-all_of(TARGET))
  
   # save
  
  write_csv(X_train, "data/Training/X_train.csv")
  write_csv(X_test,  "data/Training/X_test.csv")
  write_csv(tibble(y = y_train), "data/Training/y_train.csv")
  write_csv(tibble(y = y_test),  "data/Training/y_test.csv")
  
  cat("Shared preprocessing complete. Binary classification target:", TARGET, "\n")
  
