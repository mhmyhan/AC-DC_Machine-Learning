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


# Dropping columns

cols_to_drop <- c( 
  "Patient ID", # for sure
  "Nottingham prognostic index", # for not sure
  "Cohort", # for sure 
  "Sex",# all female drop for sure
  "Overall Survival (Months)", # drop if doing relapse
  "Relapse Free Status (Months)", # drop if doing relapse
  "Overall Survival Status"# drop if predicting survival time 
  )

  df <- df %>% select(-any_of(cols_to_drop))
  
  #Fix data types
  
  df <- df %>% 
    mutate(
      `Tumor Stage` = as.factor(`Tumor Stage`),
      `Neoplasm Histologic Grade` =as.factor(`Neoplasm Histologic Grade`),
      !!TARGET := as.factor(.data[[TARGET]])
    )
  
  #Cleaning
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
  
  #Train Split
  
  set.seed(42)
  
  split <- initial_split(df, prop = 0.8)
  
  train_raw <- training(split)
  test_raw <- testing(split)
  
  # Shared recipe
  
  rec <- recipe(train_raw) %>%
    update_role(all_of(TARGET), new_role = "outcome") %>%
    update_role(-all_of(TARGET), new_role = "predictor") %>%
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
  
