library(tidyverse)
library(recipes)
library(rsample)

# Confirm working directory
getwd()

# Load data set
file_path <- "data/Processed/metabric_clean.csv"
df <- read_csv(file_path)

# Dropping columns

cols_to_drop <- c( 
  "Patient ID",
  "Cancer Type",
  "Cancer Type Detailed",
  "Cohort", "Integrative Cluster",
  "Oncotree Code",
  "3-Gene cl",
  "Sex",# all female
  "Type of Breast Surgery",
  "Radio Therapy",
  "Chemotherapy",
  "Overall Survival Status"# drop if predicting survival time 
  )

  df <- df %>% select(-any_of(cols_to_drop))
  
  # Define numeric + categorical columns
  
  numeric_cols <- c(
    "Age at Diagnosis",
    "Neoplasm Histologic Grade",
    "Lymph nodes examined positive",
    "Mutation Count",
    "Nottingham prognostic index",
    "Overall Survival (Months)",
    "Relapse Free Status (Months)",
    "Tumor Size",
    "Tumor Stage"
  )
  
  categorical_cols <- c(
    "Cellularity",
    "Pam50 + Claudin-low subtype",
    "ER status measured by IHC",
    "ER Status",
    "HER2 status measured by SNP6",
    "HER2 Status",
    "Tumor Other Histologic Subtype",
    "Hormone Therapy",
    "Inferred Menopausal State",
    "Primary Tumor Laterality",
    "PR Status",
    "Relapse Free Status"
  )
  
  # Building pre processing 
  
  rec <- recipe(~ ., data = df) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
    step_center(all_numeric_predictors()) %>%
    step_scale(all_numeric_predictors())
  
  prep_rec <- prep(rec)
  df_processed <- bake(prep_rec, new_data = df)
  
  # Train Split
  
  set.seed(42)
  split <- initial_split(df_processed, prop = 0.8, strata = `Tumor Stage`)
  train <- training(split)
  test <- testing(split)
  
  # Save outputs
  
  write_csv(train, "data/Processed/X_train.csv")
  write_csv(test, "data/Processed/X_test.csv")
  