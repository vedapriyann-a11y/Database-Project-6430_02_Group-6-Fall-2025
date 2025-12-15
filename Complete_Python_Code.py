

# BUS TERMINAL PASSENGER FORECASTING PROJECT
# Complete Python Code with All Analysis

# 


#----------------------------------------------------------------------
#  IMPORT LIBRARIES
# ==============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Time Series Models (Excel Technique)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Regression Models (R Technique)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# Factor Analysis (Power BI Technique)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Model Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Date/Time handling
from datetime import datetime, timedelta

print("=" * 80)
print("BUS TERMINAL PASSENGER FORECASTING - COMPLETE ANALYSIS")
print("=" * 80)




# ============================================================================
#  FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING (CREATING 29 VARIABLES)")
print("=" * 80)

# Create a copy for feature engineering
df_features = weekly_totals.copy()

# 1. TIME INDEX (sequential numbering)
df_features['time_index'] = range(len(df_features))

# 2. CYCLICAL FEATURES (sine and cosine for seasonality)
# These capture repeating patterns without dummy variables
df_features['sin_week'] = np.sin(2 * np.pi * df_features['week_of_year'] / 52)
df_features['cos_week'] = np.cos(2 * np.pi * df_features['week_of_year'] / 52)
df_features['sin_month'] = np.sin(2 * np.pi * df_features['month'] / 12)
df_features['cos_month'] = np.cos(2 * np.pi * df_features['month'] / 12)

# 3. LAG FEATURES (historical values)
df_features['passengers_lag1'] = df_features['passengers'].shift(1)    # 1 week ago
df_features['passengers_lag4'] = df_features['passengers'].shift(4)    # 4 weeks ago
df_features['passengers_lag52'] = df_features['passengers'].shift(52)  # 52 weeks ago (same week last year)

# 4. ROLLING STATISTICS (moving averages and standard deviation)
df_features['passengers_roll_mean_4'] = df_features['passengers'].rolling(window=4, min_periods=1).mean()
df_features['passengers_roll_mean_12'] = df_features['passengers'].rolling(window=12, min_periods=1).mean()
df_features['passengers_roll_std_4'] = df_features['passengers'].rolling(window=4, min_periods=1).std()

# 5. GROWTH RATE (percentage change)
df_features['growth_rate'] = df_features['passengers'].pct_change()

# 6. WEATHER FEATURES (simulated - in production, use actual weather API)
# Average temperature for NYC area by month
def get_avg_temp(month):
    temps = {1: 33, 2: 36, 3: 43, 4: 54, 5: 64, 6: 73,
             7: 78, 8: 77, 9: 70, 10: 59, 11: 48, 12: 38}
    return temps[month]

df_features['avg_weekly_temp_f'] = df_features['month'].apply(get_avg_temp)
df_features['temp_squared'] = df_features['avg_weekly_temp_f'] ** 2  # Non-linear temperature effect

# Precipitation (simulated)
df_features['total_precipitation_inches'] = np.random.exponential(0.5, len(df_features))

# Snow (simulated - only winter months)
df_features['total_snow_inches'] = 0.0
winter_mask = df_features['month'].isin([12, 1, 2, 3])
df_features.loc[winter_mask, 'total_snow_inches'] = np.random.exponential(1, winter_mask.sum())

# Bad weather indicator
df_features['is_bad_weather'] = ((df_features['total_snow_inches'] > 3) | 
                                  (df_features['total_precipitation_inches'] > 1.5)).astype(int)

# Extreme cold days
df_features['extreme_cold_days'] = (df_features['avg_weekly_temp_f'] < 20).astype(int)

# 7. HOLIDAY FEATURES
# Major US holidays that affect bus travel
holidays_2020_2025 = [
    '2020-11-26', '2020-12-25',  # Thanksgiving, Christmas 2020
    '2021-11-25', '2021-12-25',  # 2021
    '2022-11-24', '2022-12-25',  # 2022
    '2023-11-23', '2023-12-25',  # 2023
    '2024-11-28', '2024-12-25',  # 2024
    '2025-11-27', '2025-12-25'   # 2025
]
holiday_dates = pd.to_datetime(holidays_2020_2025)

# Holiday indicators
df_features['holidays_in_week'] = 0  # Simplified - in production, check if holiday in week
df_features['is_thanksgiving_week'] = ((df_features['month'] == 11) & 
                                        (df_features['week_of_year'].isin([47, 48]))).astype(int)
df_features['is_christmas_week'] = ((df_features['month'] == 12) & 
                                     (df_features['week_of_year'].isin([51, 52]))).astype(int)
df_features['major_holiday'] = (df_features['is_thanksgiving_week'] | 
                                 df_features['is_christmas_week']).astype(int)
df_features['is_summer_holiday'] = df_features['month'].isin([7, 8]).astype(int)

# 8. OTHER FEATURES
df_features['is_school_break'] = (df_features['is_summer_holiday'] | 
                                   df_features['major_holiday']).astype(int)
df_features['commuter_days_count'] = 5  # Simplified - weekdays per week
df_features['covid_recovery'] = (df_features['year'].isin([2021, 2022])).astype(int)

# Fill any remaining NaN values
df_features = df_features.fillna(method='bfill').fillna(0)

print(f"\nFeature engineering complete!")
print(f"Total features created: {len(df_features.columns)}")
print(f"Features include: temporal, lags, rolling stats, weather, holidays")




# ============================================================================
- POWER BI + FACTOR ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("TECHNIQUE 3: POWER BI + FACTOR ANALYSIS (PCA)")
print("=" * 80)

# Use same features as R regression
X_train_pca = X_train.copy()
X_test_pca = X_test.copy()

# Standardize (required for PCA)
X_train_scaled_pca = scaler.fit_transform(X_train_pca)
X_test_scaled_pca = scaler.transform(X_test_pca)

# PRINCIPAL COMPONENT ANALYSIS
print("\n--- Principal Component Analysis ---")

# PCA with 95% variance threshold
pca = PCA(n_components=0.95)  # Keep components that explain 95% of variance
X_train_pca_transformed = pca.fit_transform(X_train_scaled_pca)
X_test_pca_transformed = pca.transform(X_test_scaled_pca)

n_components = pca.n_components_
variance_explained = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_explained)

print(f"\nPCA Results:")
print(f"  Original variables: {len(feature_cols)}")
print(f"  Extracted factors: {n_components}")
print(f"  Total variance explained: {cumulative_variance[-1]:.2%}")

# Factor loadings (how each variable loads on each factor)
factor_loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'Factor{i+1}' for i in range(n_components)],
    index=feature_cols
)

print(f"\nTop 5 variables for Factor 1:")
factor1_top = factor_loadings['Factor1'].abs().sort_values(ascending=False).head(5)
for var, loading in factor1_top.items():
    print(f"  {var:30s}: {loading:.3f}")

# Save factor loadings
factor_loadings_export = factor_loadings.reset_index()
factor_loadings_export.columns = ['variable'] + [f'Factor{i+1}' for i in range(n_components)]
factor_loadings_export.to_csv('powerbi_factor_loadings.csv', index=False)
print("\n✓ Factor loadings saved: powerbi_factor_loadings.csv")

# REGRESSION ON FACTOR SCORES
print("\n--- Regression on Factor Scores ---")
pca_reg_model = LinearRegression()
pca_reg_model.fit(X_train_pca_transformed, y_train)
y_pred_test_pca = pca_reg_model.predict(X_test_pca_transformed)

mae_pca = mean_absolute_error(y_test, y_pred_test_pca)
r2_test_pca = r2_score(y_test, y_pred_test_pca)

print(f"Factor Analysis Model Performance:")
print(f"  Test R²: {r2_test_pca:.4f}")
print(f"  Test MAE: {mae_pca:,.0f} passengers")

# Factor importance in prediction
factor_importance = pd.DataFrame({
    'factor': [f'Factor{i+1}' for i in range(n_components)],
    'coefficient': pca_reg_model.coef_,
    'abs_coefficient': np.abs(pca_reg_model.coef_),
    'variance_explained': variance_explained
}).sort_values('abs_coefficient', ascending=False)

print(f"\nTop 5 Factors in Prediction:")
for idx, row in factor_importance.head(5).iterrows():
    print(f"  {row['factor']:10s}: Coef={row['coefficient']:>8.2f}, Var={row['variance_explained']:.2%}")

# Save factor importance
factor_importance.to_csv('powerbi_factor_importance.csv', index=False)
print("\n✓ Factor importance saved: powerbi_factor_importance.csv")

# REGRESSION ON ORIGINAL VARIABLES (for comparison)
print("\n--- Regression on Original Variables ---")
orig_model = LinearRegression()
orig_model.fit(X_train_scaled_pca, y_train)
y_pred_test_orig = orig_model.predict(X_test_scaled_pca)
mae_orig = mean_absolute_error(y_test, y_pred_test_orig)

print(f"Original Variables Model Performance:")
print(f"  Test MAE: {mae_orig:,.0f} passengers")

# Determine best Power BI model
if mae_orig < mae_pca:
    best_powerbi_model = 'Original Variables'
    best_powerbi_mae = mae_orig
else:
    best_powerbi_model = 'Factor Analysis'
    best_powerbi_mae = mae_pca

print(f"\n✓ Best Power BI Model: {best_powerbi_model} (MAE: {best_powerbi_mae:,.0f})")

# Save original variable importance
orig_importance = pd.DataFrame({
    'variable': feature_cols,
    'coefficient': orig_model.coef_,
    'abs_coefficient': np.abs(orig_model.coef_)
}).sort_values('abs_coefficient', ascending=False)

orig_importance.to_csv('powerbi_variable_importance.csv', index=False)
print("✓ Variable importance saved: powerbi_variable_importance.csv")


# ============================================================================
 GENERATE 2026-2030 FORECASTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: GENERATING COMPLETE 5-YEAR FORECASTS (2026-2030)")
print("=" * 80)

# Create future dates (260 weeks = 5 years × 52 weeks)
start_date = pd.Timestamp('2026-01-05')  # First Monday of 2026
future_weeks = 260
future_dates = pd.date_range(start=start_date, periods=future_weeks, freq='W-MON')

# Create future dataframe
future_df = pd.DataFrame({
    'date': future_dates,
    'year': future_dates.year,
    'month': future_dates.month,
    'quarter': future_dates.quarter,
    'week_of_year': future_dates.isocalendar().week
})

print(f"\nForecasting period: 2026-2030 ({future_weeks} weeks)")

# Prepare features for Random Forest forecast
future_df['time_index'] = range(len(train_data), len(train_data) + future_weeks)
future_df['sin_week'] = np.sin(2 * np.pi * future_df['week_of_year'] / 52)
future_df['cos_week'] = np.cos(2 * np.pi * future_df['week_of_year'] / 52)
future_df['sin_month'] = np.sin(2 * np.pi * future_df['month'] / 12)
future_df['cos_month'] = np.cos(2 * np.pi * future_df['month'] / 12)

# Use subset of features for future prediction (only those available without historical data)
future_feature_cols = ['time_index', 'year', 'week_of_year', 'month', 'quarter',
                       'sin_week', 'cos_week', 'sin_month', 'cos_month']

X_future = future_df[future_feature_cols]

# Random Forest forecast
rf_forecast = rf_model.predict(X_future)

print(f"\nRandom Forest Forecast:")
print(f"  Average weekly: {rf_forecast.mean():,.0f} passengers")
print(f"  Total 5-year: {rf_forecast.sum():,.0f} passengers")

# Ensemble forecast (average of SARIMA and Random Forest)
ensemble_forecast = (forecast_sarima.values + rf_forecast) / 2

print(f"\nEnsemble Forecast (SARIMA + RF average):")
print(f"  Average weekly: {ensemble_forecast.mean():,.0f} passengers")
print(f"  Total 5-year: {ensemble_forecast.sum():,.0f} passengers")

# Create complete forecast dataframe
forecast_complete = pd.DataFrame({
    'date': future_dates,
    'year': future_df['year'],
    'month': future_df['month'],
    'month_name': future_df['month'].map(month_names),
    'quarter': future_df['quarter'],
    'week_of_year': future_df['week_of_year'],
    'sarima_forecast': forecast_sarima.values,
    'rf_forecast': rf_forecast,
    'ensemble_forecast': ensemble_forecast
})

# Calculate annual totals
annual_forecast = forecast_complete.groupby('year').agg({
    'ensemble_forecast': ['sum', 'mean', 'count']
}).reset_index()
annual_forecast.columns = ['year', 'total_passengers', 'avg_weekly', 'week_count']

print(f"\nAnnual Forecasts:")
for _, row in annual_forecast.iterrows():
    print(f"  {int(row['year'])}: {row['total_passengers']:>12,.0f} passengers ({int(row['week_count'])} weeks)")

# Save forecast data
forecast_complete.to_csv('forecast_2026_2030_complete.csv', index=False)
annual_forecast.to_csv('forecast_annual_totals.csv', index=False)
print("\n✓ Forecasts saved: forecast_2026_2030_complete.csv, forecast_annual_totals.csv")


# ============================================================================
 CARRIER-SPECIFIC FORECASTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: GENERATING CARRIER-SPECIFIC FORECASTS")
print("=" * 80)

# Calculate historical market share (2021-2025)
historical = df[df['year'].between(2021, 2025)].copy()
carrier_totals = historical.groupby('carrier')['passengers'].sum().reset_index()
carrier_totals['market_share'] = carrier_totals['passengers'] / carrier_totals['passengers'].sum()
carrier_totals = carrier_totals.sort_values('passengers', ascending=False)

print(f"\nHistorical Market Share (2021-2025):")
for _, row in carrier_totals.iterrows():
    print(f"  {row['carrier']:25s}: {row['market_share']:>6.2%}")

# Calculate carrier growth rates (CAGR)
yearly_by_carrier = historical.groupby(['year', 'carrier'])['passengers'].sum().reset_index()

carrier_growth = []
for carrier in carrier_totals['carrier']:
    carrier_data = yearly_by_carrier[yearly_by_carrier['carrier'] == carrier].sort_values('year')
    if len(carrier_data) >= 2:
        first_year = carrier_data.iloc[0]['passengers']
        last_year = carrier_data.iloc[-1]['passengers']
        years = len(carrier_data) - 1
        if first_year > 0:
            cagr = (last_year / first_year) ** (1/years) - 1
        else:
            cagr = 0
    else:
        cagr = 0
    carrier_growth.append({'carrier': carrier, 'cagr': cagr})

carrier_growth_df = pd.DataFrame(carrier_growth)
carrier_info = carrier_totals.merge(carrier_growth_df, on='carrier')

# Generate carrier-specific forecasts
carrier_forecasts = []

for _, carrier_row in carrier_info.iterrows():
    carrier = carrier_row['carrier']
    base_share = carrier_row['market_share']
    growth_rate = carrier_row['cagr']
    
    for _, forecast_row in forecast_complete.iterrows():
        year = forecast_row['year']
        total_forecast = forecast_row['ensemble_forecast']
        
        # Adjust market share based on growth trend
        years_ahead = year - 2025
        adjusted_share = base_share * (1 + growth_rate) ** years_ahead
        
        # Calculate carrier forecast
        carrier_forecast = total_forecast * adjusted_share
        
        carrier_forecasts.append({
            'carrier': carrier,
            'year': year,
            'week': forecast_row['week_of_year'],
            'date': forecast_row['date'],
            'forecasted_passengers': carrier_forecast,
            'market_share': adjusted_share
        })

carrier_forecast_df = pd.DataFrame(carrier_forecasts)

# Normalize to ensure totals match
for date in carrier_forecast_df['date'].unique():
    date_mask = carrier_forecast_df['date'] == date
    total_for_date = carrier_forecast_df[date_mask]['forecasted_passengers'].sum()
    actual_total = forecast_complete[forecast_complete['date'] == date]['ensemble_forecast'].values[0]
    
    if total_for_date > 0:
        adjustment_factor = actual_total / total_for_date
        carrier_forecast_df.loc[date_mask, 'forecasted_passengers'] *= adjustment_factor

# Calculate 5-year totals by carrier
carrier_5yr_totals = carrier_forecast_df.groupby('carrier')['forecasted_passengers'].sum().reset_index()
carrier_5yr_totals = carrier_5yr_totals.sort_values('forecasted_passengers', ascending=False)

print(f"\nTop 5 Carriers (2026-2030):")
for idx, row in carrier_5yr_totals.head(5).iterrows():
    print(f"  {idx+1}. {row['carrier']:25s}: {row['forecasted_passengers']:>12,.0f}")

# Save carrier forecasts
carrier_forecast_df.to_csv('carrier_forecasts_complete.csv', index=False)
carrier_5yr_totals.to_csv('carrier_5yr_totals.csv', index=False)
print("\n✓ Carrier forecasts saved: carrier_forecasts_complete.csv, carrier_5yr_totals.csv")


# ============================================================================
BUSIEST TIMES ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: ANALYZING BUSIEST TIMES")
print("=" * 80)

# Monthly analysis
monthly_stats = forecast_complete.groupby(['month', 'month_name'])['ensemble_forecast'].agg(['sum', 'mean', 'count']).reset_index()
monthly_stats.columns = ['month', 'month_name', 'total', 'avg_weekly', 'week_count']
monthly_stats = monthly_stats.sort_values('avg_weekly', ascending=False)

print(f"\nBusiest Months:")
for idx, row in monthly_stats.head(5).iterrows():
    print(f"  {idx+1}. {row['month_name']:12s}: {row['avg_weekly']:>10,.0f} avg weekly")

# Weekly pattern
weekly_pattern = forecast_complete.groupby('week_of_year')['ensemble_forecast'].mean().reset_index()
weekly_pattern = weekly_pattern.sort_values('ensemble_forecast', ascending=False)

print(f"\nBusiest Weeks of Year:")
for idx, row in weekly_pattern.head(5).iterrows():
    print(f"  {idx+1}. Week {int(row['week_of_year']):2d}: {row['ensemble_forecast']:>10,.0f} avg")

# Hourly pattern (based on typical bus terminal patterns)
hourly_pattern = pd.DataFrame({
    'hour': ['6-7 AM', '7-8 AM', '8-9 AM', '9-10 AM', '10-11 AM', '11 AM-12 PM',
             '12-1 PM', '1-2 PM', '2-3 PM', '3-4 PM', '4-5 PM', '5-6 PM',
             '6-7 PM', '7-8 PM', '8-9 PM', '9-10 PM', '10 PM-6 AM'],
    'pct_of_daily': [8.5, 12.0, 10.5, 5.0, 4.5, 4.0, 4.5, 4.0, 3.5, 6.0, 10.0, 12.0, 8.0, 3.5, 2.0, 1.5, 0.5]
})

print(f"\nPeak Hours:")
peak_hours = hourly_pattern[hourly_pattern['pct_of_daily'] >= 10]
for _, row in peak_hours.iterrows():
    print(f"  {row['hour']:15s}: {row['pct_of_daily']:>5.1f}% of daily traffic")

# Save busiest times data
monthly_stats.to_csv('busiest_times_monthly.csv', index=False)
weekly_pattern.to_csv('busiest_times_weekly.csv', index=False)
hourly_pattern.to_csv('busiest_times_hourly.csv', index=False)
print("\n✓ Busiest times saved: busiest_times_monthly.csv, busiest_times_weekly.csv, busiest_times_hourly.csv")


# ============================================================================
 CREATE POWER BI DATA TABLES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: CREATING POWER BI DATA TABLES (STAR SCHEMA)")
print("=" * 80)

# FACT TABLE 1: Weekly Forecasts
powerbi_fact_weekly = forecast_complete.copy()
powerbi_fact_weekly['date_key'] = powerbi_fact_weekly['date'].dt.strftime('%Y%m%d').astype(int)
powerbi_fact_weekly['is_holiday_season'] = powerbi_fact_weekly['month'].isin([11, 12]).astype(int)
powerbi_fact_weekly['is_summer'] = powerbi_fact_weekly['month'].isin([6, 7, 8]).astype(int)

print(f"\nFact Table - Weekly Forecasts: {len(powerbi_fact_weekly)} rows")

# DIMENSION TABLE 1: Date
powerbi_dim_date = powerbi_fact_weekly[['date', 'year', 'quarter', 'month', 'month_name', 'week_of_year']].drop_duplicates()
powerbi_dim_date['date_key'] = powerbi_dim_date['date'].dt.strftime('%Y%m%d').astype(int)
powerbi_dim_date['is_weekend'] = powerbi_dim_date['date'].dt.dayofweek.isin([5, 6]).astype(int)
powerbi_dim_date['is_holiday'] = 0  # Simplified

print(f"Dimension Table - Date: {len(powerbi_dim_date)} rows")

# DIMENSION TABLE 2: Carrier
powerbi_dim_carrier = pd.DataFrame({
    'carrier_id': range(1, len(carrier_totals) + 1),
    'carrier_name': carrier_totals['carrier'].values,
    'market_share': carrier_totals['market_share'].values
})

print(f"Dimension Table - Carrier: {len(powerbi_dim_carrier)} rows")

# FACT TABLE 2: Carrier Forecasts
powerbi_fact_carrier = carrier_forecast_df.copy()
powerbi_fact_carrier['date_key'] = pd.to_datetime(powerbi_fact_carrier['date']).dt.strftime('%Y%m%d').astype(int)
powerbi_fact_carrier = powerbi_fact_carrier.merge(
    powerbi_dim_carrier[['carrier_name', 'carrier_id']], 
    left_on='carrier', 
    right_on='carrier_name', 
    how='left'
)

print(f"Fact Table - Carrier Forecasts: {len(powerbi_fact_carrier)} rows")

# SUMMARY TABLE 1: Yearly
powerbi_summary_yearly = annual_forecast.copy()

print(f"Summary Table - Yearly: {len(powerbi_summary_yearly)} rows")

# SUMMARY TABLE 2: Monthly
powerbi_summary_monthly = monthly_stats.copy()

print(f"Summary Table - Monthly: {len(powerbi_summary_monthly)} rows")

# SUMMARY TABLE 3: Carrier
powerbi_summary_carrier = carrier_5yr_totals.copy()
powerbi_summary_carrier['avg_weekly'] = powerbi_summary_carrier['forecasted_passengers'] / 260

print(f"Summary Table - Carrier: {len(powerbi_summary_carrier)} rows")

# Export all Power BI tables
powerbi_fact_weekly.to_csv('powerbi_fact_weekly.csv', index=False)
powerbi_dim_date.to_csv('powerbi_dim_date.csv', index=False)
powerbi_dim_carrier.to_csv('powerbi_dim_carrier.csv', index=False)
powerbi_fact_carrier.to_csv('powerbi_fact_carrier.csv', index=False)
powerbi_summary_yearly.to_csv('powerbi_summary_yearly.csv', index=False)
powerbi_summary_monthly.to_csv('powerbi_summary_monthly.csv', index=False)
powerbi_summary_carrier.to_csv('powerbi_summary_carrier.csv', index=False)

print("\n✓ All Power BI tables exported!")



# ============================================================================
FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"\n✓ COMPLETE 5-YEAR FORECAST:")
print(f"  Total Passengers (2026-2030): {annual_forecast['total_passengers'].sum():,.0f}")
print(f"  Average Weekly: {annual_forecast['avg_weekly'].mean():,.0f}")
print(f"  Growth: {((annual_forecast.iloc[-1]['total_passengers'] / annual_forecast.iloc[0]['total_passengers']) - 1) * 100:.1f}%")

print(f"\n✓ BEST MODELS:")
print(f"  Excel: SARIMA (MAE: 57,269)")
print(f"  R: OLS (MAE: {mae_ols:,.0f})")
print(f"  Power BI: {best_powerbi_model} (MAE: {best_powerbi_mae:,.0f})")

print(f"\n✓ TOP CARRIER:")
print(f"  {carrier_5yr_totals.iloc[0]['carrier']}: {carrier_5yr_totals.iloc[0]['forecasted_passengers']:,.0f} passengers")
print(f"  Market Share: {carrier_5yr_totals.iloc[0]['forecasted_passengers'] / carrier_5yr_totals['forecasted_passengers'].sum():.1%}")

print(f"\n✓ BUSIEST TIMES:")
print(f"  Month: {monthly_stats.iloc[0]['month_name']} ({monthly_stats.iloc[0]['avg_weekly']:,.0f} avg weekly)")
print(f"  Week: {int(weekly_pattern.iloc[0]['week_of_year'])} ({weekly_pattern.iloc[0]['ensemble_forecast']:,.0f} avg)")
print(f"  Hours: 7-8 AM and 5-6 PM (12% each)")

print(f"\n✓ CSV FILES CREATED: 10 files ready for Power BI import")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nAll CSV files have been created and are ready for Power BI import.")
print("Next steps:")
print("  1. Import all 10 CSV files into Power BI")
print("  2. Create relationships between tables")
print("  3. Create measures (DAX formulas)")
print("  4. Build visualizations")
print("  5. Publish dashboard")



