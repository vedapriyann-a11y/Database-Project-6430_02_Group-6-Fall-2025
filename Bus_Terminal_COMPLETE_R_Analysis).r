
################################################################################
# BUS TERMINAL PASSENGER FORECASTING - COMPLETE R ANALYSIS
# Port Authority Temporary Staging Facilities Planning

# Technique: R + Regression Analysis
################################################################################

# INSTALL REQUIRED PACKAGES (Run once)
################################################################################
# Uncomment and run these lines if packages are not installed:
# install.packages(c("tidyverse", "lubridate", "zoo", "caret", "car", 
#                    "glmnet", "randomForest", "ggplot2", "gridExtra"))

# LOAD LIBRARIES
################################################################################
library(tidyverse)
library(lubridate)
library(zoo)
library(caret)
library(car)
library(glmnet)
library(randomForest)
library(ggplot2)
library(gridExtra)

cat("\n")
cat("================================================================================\n")
cat("BUS TERMINAL PASSENGER FORECASTING - R ANALYSIS\n")
cat("================================================================================\n\n")

# SET WORKING DIRECTORY
################################################################################
# Change this to your project folder location
# setwd("~/bus_terminal_project")

# STEP 1: LOAD DATA
################################################################################
cat("STEP 1: Loading data...\n")

# Load bus passenger data
bus_data <- read_csv("bus_passenger_data.csv", show_col_types = FALSE)
weekly_data <- read_csv("weekly_totals.csv", show_col_types = FALSE)
external_data <- read_csv("external_factors_weekly.csv", show_col_types = FALSE)

cat("  ✓ Bus data loaded:", nrow(bus_data), "rows\n")
cat("  ✓ Weekly data loaded:", nrow(weekly_data), "rows\n")
cat("  ✓ External factors loaded:", nrow(external_data), "rows\n")

# STEP 2: DATA PREPARATION & FEATURE ENGINEERING
################################################################################
cat("\nSTEP 2: Feature engineering...\n")

# Merge weekly data with external factors
weekly_data <- weekly_data %>%
  mutate(week_start = floor_date(date_parsed, "week"))

merged_data <- weekly_data %>%
  left_join(external_data, by = c("week_start" = "week_start_date"), suffix = c("", "_ext"))

# Feature engineering
merged_data <- merged_data %>%
  mutate(
    # Temporal features
    time_index = row_number(),
    sin_week = sin(2 * pi * week_of_year / 52),
    cos_week = cos(2 * pi * week_of_year / 52),
    sin_month = sin(2 * pi * month / 12),
    cos_month = cos(2 * pi * month / 12),
    
    # Lag features
    passengers_lag1 = lag(passengers, 1),
    passengers_lag4 = lag(passengers, 4),
    passengers_lag52 = lag(passengers, 52),
    
    # Rolling statistics
    passengers_roll_mean_4 = rollmean(passengers, k = 4, fill = NA, align = "right"),
    passengers_roll_mean_12 = rollmean(passengers, k = 12, fill = NA, align = "right"),
    passengers_roll_std_4 = rollapply(passengers, width = 4, FUN = sd, fill = NA, align = "right"),
    
    # Growth rate
    growth_rate = (passengers - lag(passengers)) / lag(passengers),
    
    # Weather interactions
    temp_squared = avg_weekly_temp_f^2,
    is_bad_weather = as.integer(severe_weather_days > 0 | total_snow_inches > 3),
    
    # Holiday indicators
    major_holiday = as.integer(is_thanksgiving_week == 1 | is_christmas_week == 1)
  )

# Fill missing values
merged_data <- merged_data %>%
  fill(everything(), .direction = "downup") %>%
  replace_na(list(
    passengers_lag1 = 0, passengers_lag4 = 0, passengers_lag52 = 0,
    passengers_roll_mean_4 = 0, passengers_roll_mean_12 = 0,
    passengers_roll_std_4 = 0, growth_rate = 0
  ))

cat("  ✓ Features created:", ncol(merged_data), "columns\n")

# Split into training and test sets
train_data <- merged_data %>% filter(year >= 2021 & year <= 2024)
test_data <- merged_data %>% filter(year == 2025)

cat("  ✓ Training set:", nrow(train_data), "observations (2021-2024)\n")
cat("  ✓ Test set:", nrow(test_data), "observations (2025)\n")

# STEP 3: DEFINE VARIABLES
################################################################################
cat("\nSTEP 3: Defining variables...\n")

# Independent variables (29 total)
independent_vars <- c(
  # Temporal (9)
  "time_index", "year", "month", "quarter", "week_of_year",
  "sin_week", "cos_week", "sin_month", "cos_month",
  # Lagged (3)
  "passengers_lag1", "passengers_lag4", "passengers_lag52",
  # Rolling statistics (2)
  "passengers_roll_mean_4", "passengers_roll_mean_12",
  # Capacity (1)
  "buses",
  # Weather (6)
  "avg_weekly_temp_f", "temp_squared", "total_precipitation_inches",
  "total_snow_inches", "is_bad_weather", "extreme_cold_days",
  # Holiday (5)
  "holidays_in_week", "is_thanksgiving_week", "is_christmas_week",
  "major_holiday", "is_summer_holiday",
  # Other (3)
  "is_school_break", "commuter_days_count", "covid_recovery"
)

# Dependent variable
dependent_var <- "passengers"

cat("  ✓ Independent variables:", length(independent_vars), "\n")
cat("  ✓ Dependent variable:", dependent_var, "\n")

# STEP 4: MODEL 1 - OLS REGRESSION
################################################################################
cat("\n")
cat("================================================================================\n")
cat("MODEL 1: OLS REGRESSION\n")
cat("================================================================================\n")

# Prepare data
X_train <- train_data[, independent_vars]
y_train <- train_data[[dependent_var]]
X_test <- test_data[, independent_vars]
y_test <- test_data[[dependent_var]]

# Standardize features
preProc <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preProc, X_train)
X_test_scaled <- predict(preProc, X_test)

# Combine for modeling
train_df <- cbind(passengers = y_train, X_train_scaled)
test_df <- cbind(passengers = y_test, X_test_scaled)

# Build OLS model
ols_model <- lm(passengers ~ ., data = train_df)

# Model summary
cat("\nOLS Model Summary:\n")
print(summary(ols_model))

# Predictions
train_pred_ols <- predict(ols_model, train_df)
test_pred_ols <- predict(ols_model, test_df)

# Performance metrics
train_r2_ols <- summary(ols_model)$r.squared
train_adj_r2_ols <- summary(ols_model)$adj.r.squared
test_r2_ols <- 1 - sum((y_test - test_pred_ols)^2) / sum((y_test - mean(y_test))^2)
test_mae_ols <- mean(abs(y_test - test_pred_ols))
test_rmse_ols <- sqrt(mean((y_test - test_pred_ols)^2))

cat("\nOLS Performance Metrics:\n")
cat("  Training R²:", round(train_r2_ols, 4), "\n")
cat("  Adjusted R²:", round(train_adj_r2_ols, 4), "\n")
cat("  Test R²:", round(test_r2_ols, 4), "\n")
cat("  Test MAE:", format(round(test_mae_ols, 0), big.mark=","), "passengers\n")
cat("  Test RMSE:", format(round(test_rmse_ols, 0), big.mark=","), "passengers\n")

# Variable importance
ols_coef <- data.frame(
  variable = names(coef(ols_model))[-1],
  coefficient = coef(ols_model)[-1],
  abs_coefficient = abs(coef(ols_model)[-1])
) %>%
  arrange(desc(abs_coefficient))

cat("\nTop 10 Most Important Variables (OLS):\n")
print(head(ols_coef, 10), row.names = FALSE)

# Save OLS results
write_csv(ols_coef, "R_OLS_Variable_Importance.csv")

# STEP 5: MODEL 2 - RIDGE REGRESSION
################################################################################
cat("\n")
cat("================================================================================\n")
cat("MODEL 2: RIDGE REGRESSION\n")
cat("================================================================================\n")

# Prepare matrices for glmnet
X_train_matrix <- as.matrix(X_train_scaled)
X_test_matrix <- as.matrix(X_test_scaled)

# Ridge regression (alpha = 0 for L2)
ridge_model <- glmnet(
  x = X_train_matrix,
  y = y_train,
  alpha = 0,  # Ridge (L2 regularization)
  lambda = 1.0
)

# Predictions
train_pred_ridge <- predict(ridge_model, newx = X_train_matrix, s = 1.0)
test_pred_ridge <- predict(ridge_model, newx = X_test_matrix, s = 1.0)

# Performance metrics
train_r2_ridge <- 1 - sum((y_train - train_pred_ridge)^2) / sum((y_train - mean(y_train))^2)
test_r2_ridge <- 1 - sum((y_test - test_pred_ridge)^2) / sum((y_test - mean(y_test))^2)
test_mae_ridge <- mean(abs(y_test - test_pred_ridge))
test_rmse_ridge <- sqrt(mean((y_test - test_pred_ridge)^2))

cat("\nRidge Performance Metrics:\n")
cat("  Training R²:", round(train_r2_ridge, 4), "\n")
cat("  Test R²:", round(test_r2_ridge, 4), "\n")
cat("  Test MAE:", format(round(test_mae_ridge, 0), big.mark=","), "passengers\n")
cat("  Test RMSE:", format(round(test_rmse_ridge, 0), big.mark=","), "passengers\n")

# STEP 6: MODEL 3 - RANDOM FOREST
################################################################################
cat("\n")
cat("================================================================================\n")
cat("MODEL 3: RANDOM FOREST REGRESSION\n")
cat("================================================================================\n")

# Prepare data
train_rf <- train_data[, c(dependent_var, independent_vars)]
test_rf <- test_data[, c(dependent_var, independent_vars)]

# Build Random Forest
set.seed(42)
rf_model <- randomForest(
  passengers ~ .,
  data = train_rf,
  ntree = 200,
  maxnodes = 15,
  mtry = 9,  # sqrt(29) for regression
  importance = TRUE,
  na.action = na.omit
)

cat("\nRandom Forest Model:\n")
print(rf_model)

# Predictions
train_pred_rf <- predict(rf_model, train_rf)
test_pred_rf <- predict(rf_model, test_rf)

# Performance metrics
train_r2_rf <- 1 - sum((train_rf$passengers - train_pred_rf)^2) / 
                   sum((train_rf$passengers - mean(train_rf$passengers))^2)
test_r2_rf <- 1 - sum((test_rf$passengers - test_pred_rf)^2) / 
                  sum((test_rf$passengers - mean(test_rf$passengers))^2)
test_mae_rf <- mean(abs(test_rf$passengers - test_pred_rf))
test_rmse_rf <- sqrt(mean((test_rf$passengers - test_pred_rf)^2))

cat("\nRandom Forest Performance Metrics:\n")
cat("  Training R²:", round(train_r2_rf, 4), "\n")
cat("  Test R²:", round(test_r2_rf, 4), "\n")
cat("  Test MAE:", format(round(test_mae_rf, 0), big.mark=","), "passengers\n")
cat("  Test RMSE:", format(round(test_rmse_rf, 0), big.mark=","), "passengers\n")

# Variable importance
rf_importance <- data.frame(
  variable = rownames(importance(rf_model)),
  importance = importance(rf_model)[, "%IncMSE"]
) %>%
  arrange(desc(importance))

cat("\nTop 10 Most Important Variables (Random Forest):\n")
print(head(rf_importance, 10), row.names = FALSE)

# Save RF results
write_csv(rf_importance, "R_RandomForest_Variable_Importance.csv")

# STEP 7: MODEL COMPARISON
################################################################################
cat("\n")
cat("================================================================================\n")
cat("MODEL COMPARISON\n")
cat("================================================================================\n")

model_comparison <- data.frame(
  Model = c("OLS Regression", "Ridge Regression", "Random Forest"),
  Train_R2 = c(train_r2_ols, train_r2_ridge, train_r2_rf),
  Test_R2 = c(test_r2_ols, test_r2_ridge, test_r2_rf),
  Test_MAE = c(test_mae_ols, test_mae_ridge, test_mae_rf),
  Test_RMSE = c(test_rmse_ols, test_rmse_ridge, test_rmse_rf)
)

cat("\n")
print(model_comparison, row.names = FALSE)

# Best model
best_model <- model_comparison %>%
  filter(Test_MAE == min(Test_MAE)) %>%
  pull(Model)

cat("\n✓ BEST R MODEL:", best_model, "\n")
cat("✓ Test MAE:", format(round(min(model_comparison$Test_MAE), 0), big.mark=","), "passengers\n")

# Save comparison
write_csv(model_comparison, "R_Model_Comparison.csv")

# STEP 8: VISUALIZATIONS
################################################################################
cat("\n")
cat("================================================================================\n")
cat("CREATING VISUALIZATIONS\n")
cat("================================================================================\n")

# Plot 1: Variable Importance (OLS)
p1 <- ggplot(head(ols_coef, 15), aes(x = reorder(variable, abs_coefficient), y = abs_coefficient)) +
  geom_bar(stat = "identity", fill = "#1f77b4") +
  coord_flip() +
  labs(title = "OLS Regression: Top 15 Variables by Importance",
       x = "Variable", y = "Absolute Coefficient") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Plot 2: Model Performance Comparison
p2 <- ggplot(model_comparison, aes(x = Model, y = Test_MAE, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = format(round(Test_MAE, 0), big.mark=",")), vjust = -0.5) +
  labs(title = "R Models: Performance Comparison (Test MAE)",
       x = "Model", y = "Mean Absolute Error (Passengers)") +
  scale_fill_manual(values = c("#1f77b4", "#ff7f0e", "#2ca02c")) +
  theme_minimal() +
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5, face = "bold"))

# Plot 3: Random Forest Variable Importance
p3 <- ggplot(head(rf_importance, 15), aes(x = reorder(variable, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "#2ca02c") +
  coord_flip() +
  labs(title = "Random Forest: Top 15 Variables by Importance",
       x = "Variable", y = "Importance Score") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Plot 4: Actual vs Predicted (OLS)
predictions_df <- data.frame(
  actual = y_test,
  predicted = test_pred_ols
)

p4 <- ggplot(predictions_df, aes(x = actual, y = predicted)) +
  geom_point(color = "#1f77b4", size = 3, alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) +
  labs(title = "OLS Regression: Actual vs Predicted (2025 Test)",
       x = "Actual Passengers", y = "Predicted Passengers") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Save plots
png("R_Analysis_Summary.png", width = 1600, height = 1200, res = 150)
grid.arrange(p1, p2, p3, p4, ncol = 2)
dev.off()

cat("  ✓ Visualization saved: R_Analysis_Summary.png\n")

# STEP 9: SAVE ALL RESULTS
################################################################################
cat("\n")
cat("================================================================================\n")
cat("SAVING RESULTS\n")
cat("================================================================================\n")

# Save predictions
predictions_all <- data.frame(
  date = test_data$date_parsed,
  year = test_data$year,
  week = test_data$week_of_year,
  actual = y_test,
  ols_predicted = test_pred_ols,
  ridge_predicted = test_pred_ridge,
  rf_predicted = test_pred_rf,
  ols_error = y_test - test_pred_ols,
  ridge_error = y_test - test_pred_ridge,
  rf_error = y_test - test_pred_rf
)

write_csv(predictions_all, "R_All_Predictions.csv")
cat("  ✓ R_All_Predictions.csv\n")

# Save model comparison
write_csv(model_comparison, "R_Model_Comparison.csv")
cat("  ✓ R_Model_Comparison.csv\n")

# Save variable importance
write_csv(ols_coef, "R_OLS_Variable_Importance.csv")
cat("  ✓ R_OLS_Variable_Importance.csv\n")

write_csv(rf_importance, "R_RandomForest_Variable_Importance.csv")
cat("  ✓ R_RandomForest_Variable_Importance.csv\n")

cat("\n")
cat("================================================================================\n")
cat("R ANALYSIS COMPLETE!\n")
cat("================================================================================\n")
cat("\nBest Model:", best_model, "\n")
cat("Test MAE:", format(round(min(model_comparison$Test_MAE), 0), big.mark=","), "passengers\n")
cat("\nAll results saved to CSV files.\n")
cat("Visualization saved as R_Analysis_Summary.png\n")
cat("\n✓ PROJECT COMPLETE - Ready for submission!\n")
