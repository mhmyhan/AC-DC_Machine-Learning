library(tidyverse)
library(readr)
library(ggplot2)
library(dplyr)
library(skimr)
library(janitor)

# Confirm working directory
getwd()

# Load data set
file_path <- "data/raw/BreastCancerMETABRIC.csv"
df <- read_csv(file_path)


# Preview
head(df)
glimpse(df)

# Summary statistics
summary(df)
skim(df)

#Missing Value check
colSums(is.na(df))


#Numeric and Categorical Column identification
numeric_cols <- df %>% select(where(is.numeric)) %>% names()
categorical_cols <- df %>% select(where(is.character)) %>% names()

numeric_cols
categorical_cols


df <- df %>% mutate(across(all_of(categorical_cols), as.factor))

#Visualization
df %>%
  select(all_of(numeric_cols)) %>%
  pivot_longer(everything()) %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 20, fill = "steelblue", color = "black") +
  facet_wrap(~ name, scales = "free") +
  theme_minimal()

library(corrplot)

corr_matrix <- df %>%
  select(all_of(numeric_cols)) %>%
  cor(use = "complete.obs")

corrplot(corr_matrix, method = "color", addCoef.col = "black")


#Saving 

write_csv(df, "data/processed/metabric_clean.csv")
