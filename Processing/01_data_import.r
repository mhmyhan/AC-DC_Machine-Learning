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

