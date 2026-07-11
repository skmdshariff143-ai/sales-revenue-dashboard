"""
Time-Series Forecasting & Revenue Prediction Pipeline
=====================================================
This script implements a machine learning forecasting pipeline. It generates
historical monthly revenue data, constructs features (lag values, seasonal indexes,
rolling averages, marketing spend), trains and evaluates Linear Regression vs
Random Forest Regressor models, forecasts future revenue for the next 6 months,
saves CSV data, and produces evaluation charts.

Author: Shaik Mahammad Shariff
Tech Stack: Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
RANDOM_SEED = 42
HISTORICAL_MONTHS = 24
FORECAST_MONTHS = 6
TEST_SIZE_MONTHS = 6  # Time-based holdout validation size

# Styling
DARK_BG = '#0a0e17'
CARD_BG = '#111827'
TEXT_COLOR = '#f1f5f9'
TEXT_SECONDARY = '#94a3b8'
GRID_COLOR = '#1e293b'
MODEL_COLORS = ['#6366f1', '#10b981']  # Indigo for Linear Regression, Emerald for Random Forest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(SCRIPT_DIR, 'charts')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')


def generate_time_series(months_count=HISTORICAL_MONTHS, seed=RANDOM_SEED):
    """
    Generate synthetic historical monthly data with trend, seasonality,
    marketing spend, and random noise.
    """
    np.random.seed(seed)
    date_range = pd.date_range(end='2026-06-30', periods=months_count, freq='M')
    
    # 1. Base trend (linear growth)
    time_index = np.arange(months_count)
    base_revenue = 150000 + 3500 * time_index
    
    # 2. Seasonality (peaks in Nov/Dec, valleys in Jan/Feb)
    seasonal_factors = np.sin(2 * np.pi * date_range.month / 12) * 25000
    # Add year-end shopping peak
    seasonal_factors += np.where(date_range.month.isin([11, 12]), 35000, 0)
    # Valleys in Jan/Feb
    seasonal_factors -= np.where(date_range.month.isin([1, 2]), 20000, 0)

    # 3. Marketing Spend (feature influencing revenue)
    marketing_spend = 12000 + 400 * time_index + np.random.normal(0, 1500, months_count)
    marketing_spend = np.clip(marketing_spend, 5000, None)
    marketing_influence = marketing_spend * 4.5
    
    # 4. Noise
    noise = np.random.normal(0, 8000, months_count)
    
    # Combined Revenue
    revenue = base_revenue + seasonal_factors + marketing_influence + noise
    
    df = pd.DataFrame({
        'Date': date_range,
        'Revenue': np.round(revenue, 2),
        'MarketingSpend': np.round(marketing_spend, 2),
        'Month': date_range.month,
        'Year': date_range.year
    })
    
    print(f"[INFO] Generated {len(df)} months of historical revenue data.")
    return df


def engineer_features(df):
    """
    Create feature engineering lag variables, rolling averages, and seasonal indices.
    """
    df = df.copy()
    
    # Lags
    df['Lag_Revenue_1'] = df['Revenue'].shift(1)
    df['Lag_Revenue_2'] = df['Revenue'].shift(2)
    
    # Rolling averages
    df['Rolling_3_Mean'] = df['Revenue'].shift(1).rolling(window=3).mean()
    
    # Seasonal indices (one-hot encode month)
    for m in range(1, 13):
        df[f'Month_{m}'] = (df['Month'] == m).astype(int)
        
    # Drop rows with NaN due to lags/rolling averages
    df_clean = df.dropna().reset_index(drop=True)
    print(f"[INFO] Engineered features. Dataset size after lag drops: {len(df_clean)} months.")
    return df_clean


def train_and_evaluate(df):
    """
    Train Linear Regression & Random Forest Regressor models using a time-based split.
    """
    features = [
        'MarketingSpend', 'Lag_Revenue_1', 'Lag_Revenue_2', 'Rolling_3_Mean'
    ] + [f'Month_{m}' for m in range(1, 13)]
    
    X = df[features]
    y = df['Revenue']
    
    # Time-based split: Use last N months as test/holdout set
    split_idx = len(df) - TEST_SIZE_MONTHS
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"[INFO] Training models: {len(X_train)} months train, {len(X_test)} months test...")
    
    # Model 1: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    # Model 2: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    # Metrics calculation helper
    def get_metrics(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        print(f"  {name:25s} | R²: {r2:.3f} | MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | MAPE: {mape:.2f}%")
        return {'Model': name, 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    metrics_lr = get_metrics(y_test, y_pred_lr, 'Linear Regression')
    metrics_rf = get_metrics(y_test, y_pred_rf, 'Random Forest Regressor')
    
    eval_df = pd.DataFrame([metrics_lr, metrics_rf])
    
    # Fit final models on full historical data for future forecasting
    lr_final = LinearRegression().fit(X, y)
    rf_final = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED).fit(X, y)
    
    return {
        'lr_model': lr_final,
        'rf_model': rf_final,
        'metrics': eval_df,
        'features': features,
        'y_test': y_test,
        'y_pred_rf': y_pred_rf
    }


def make_forecast(df, model, features, months=FORECAST_MONTHS):
    """
    Generate recursive monthly forecast for the next 6 months.
    """
    print(f"[INFO] Generating forecast for the next {months} months...")
    
    # Get last known state
    current_data = df.copy()
    forecasts = []
    
    last_date = current_data['Date'].iloc[-1]
    
    for i in range(1, months + 1):
        next_date = last_date + pd.DateOffset(months=i)
        next_month = next_date.month
        next_year = next_date.year
        
        # Estimate marketing spend (growing linearly)
        last_idx = len(current_data) - 1
        est_marketing = current_data['MarketingSpend'].iloc[-1] + 400
        
        # Lag features from current state
        lag_1 = current_data['Revenue'].iloc[-1]
        lag_2 = current_data['Revenue'].iloc[-2]
        roll_3 = current_data['Revenue'].iloc[-3:].mean()
        
        # Build feature row
        feat_dict = {
            'MarketingSpend': est_marketing,
            'Lag_Revenue_1': lag_1,
            'Lag_Revenue_2': lag_2,
            'Rolling_3_Mean': roll_3
        }
        for m in range(1, 13):
            feat_dict[f'Month_{m}'] = 1 if next_month == m else 0
            
        feat_df = pd.DataFrame([feat_dict])
        
        # Predict using final model
        predicted_revenue = model.predict(feat_df[features])[0]
        
        # Create row for main data frame to allow recursion
        new_row = {
            'Date': next_date,
            'Revenue': round(predicted_revenue, 2),
            'MarketingSpend': round(est_marketing, 2),
            'Month': next_month,
            'Year': next_year
        }
        
        # Add lags to display in final output
        new_row.update(feat_dict)
        forecasts.append(new_row)
        
        # Append to df for next iteration lags
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        
    return pd.DataFrame(forecasts)


def plot_and_save_charts(df_clean, forecast_df, metrics, y_test, y_pred_rf, model_rf, features):
    """Generate and save PNG plots for Task 3."""
    print("[INFO] Generating performance and evaluation charts...")
    
    # Chart 1: Forecast Overview
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    
    # Combined series for plotting
    hist_dates = df_clean['Date']
    hist_rev = df_clean['Revenue']
    
    fc_dates = forecast_df['Date']
    fc_rev = forecast_df['Revenue']
    
    ax.plot(hist_dates, hist_rev, 'o-', color='#6366f1', label='Historical Revenue', linewidth=2)
    ax.plot(fc_dates, fc_rev, 'o--', color='#10b981', label='Forecasted Revenue', linewidth=2)
    
    # Confidence Intervals (Visual Simulation)
    lower_bound = fc_rev * 0.93
    upper_bound = fc_rev * 1.07
    ax.fill_between(fc_dates, lower_bound, upper_bound, color='#10b981', alpha=0.15, label='95% Confidence Interval')
    
    ax.set_title('Monthly Revenue Forecast (Next 6 Months)', color=TEXT_COLOR, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Timeline', color=TEXT_SECONDARY)
    ax.set_ylabel('Revenue ($)', color=TEXT_SECONDARY)
    ax.grid(True, color=GRID_COLOR, alpha=0.5)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'forecast_overview.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    # Chart 2: Residuals Analysis (Random Forest Model residuals)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={'width_ratios': [2, 1]})
    fig.patch.set_facecolor(DARK_BG)
    ax1.set_facecolor(DARK_BG)
    ax2.set_facecolor(DARK_BG)
    
    residuals = y_test - y_pred_rf
    
    # Scatter plot
    ax1.scatter(y_pred_rf, residuals, color='#10b981', alpha=0.7, s=50, edgecolors='white', linewidths=0.3)
    ax1.axhline(0, color='#ef4444', linestyle='--', linewidth=1.5)
    ax1.set_title('Predicted vs. Residuals', color=TEXT_COLOR, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Revenue ($)', color=TEXT_SECONDARY)
    ax1.set_ylabel('Residuals ($)', color=TEXT_SECONDARY)
    ax1.tick_params(colors=TEXT_SECONDARY)
    ax1.grid(True, color=GRID_COLOR, alpha=0.5)
    for spine in ax1.spines.values():
        spine.set_color(GRID_COLOR)
        
    # Histogram
    sns.histplot(residuals, kde=True, color='#6366f1', ax=ax2, alpha=0.4)
    ax2.set_title('Residuals Distribution', color=TEXT_COLOR, fontsize=12, fontweight='bold')
    ax2.set_xlabel('Residual ($)', color=TEXT_SECONDARY)
    ax2.set_ylabel('Count', color=TEXT_SECONDARY)
    ax2.tick_params(colors=TEXT_SECONDARY)
    ax2.grid(True, color=GRID_COLOR, alpha=0.3)
    for spine in ax2.spines.values():
        spine.set_color(GRID_COLOR)

    plt.suptitle('Residual Analysis - Random Forest Regressor', color=TEXT_COLOR, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(CHARTS_DIR, 'residuals.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    # Chart 3: Model Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax1.set_facecolor(DARK_BG)
    ax2.set_facecolor(DARK_BG)
    
    # Plot R2
    ax1.bar(metrics['Model'], metrics['R2'], color=MODEL_COLORS, width=0.5, edgecolor=GRID_COLOR)
    ax1.set_title('R-Squared (Higher is better)', color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax1.set_ylabel('R² Score', color=TEXT_SECONDARY)
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(colors=TEXT_SECONDARY)
    ax1.grid(axis='y', color=GRID_COLOR, alpha=0.5)
    for spine in ax1.spines.values():
        spine.set_color(GRID_COLOR)
        
    # Plot MAPE
    ax2.bar(metrics['Model'], metrics['MAPE'], color=MODEL_COLORS, width=0.5, edgecolor=GRID_COLOR)
    ax2.set_title('MAPE % (Lower is better)', color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax2.set_ylabel('MAPE (%)', color=TEXT_SECONDARY)
    ax2.tick_params(colors=TEXT_SECONDARY)
    ax2.grid(axis='y', color=GRID_COLOR, alpha=0.5)
    for spine in ax2.spines.values():
        spine.set_color(GRID_COLOR)
        
    plt.suptitle('Model Performance Evaluation (Holdout Test Set)', color=TEXT_COLOR, fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(CHARTS_DIR, 'model_comparison.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    # Chart 4: Feature Importance
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    
    importances = model_rf.feature_importances_
    # Group other variables to keep it simple and clean
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(5)
    
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis', ax=ax, edgecolor=GRID_COLOR)
    ax.set_title('Feature Importance (Random Forest Regressor)', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Relative Importance', color=TEXT_SECONDARY)
    ax.set_ylabel('Feature Name', color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.grid(axis='x', color=GRID_COLOR, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'feature_importance.png'), dpi=150, facecolor=DARK_BG)
    plt.close()


def save_results(df_clean, forecast_df, metrics):
    """Save cleaned datasets, evaluation data, and predictions to CSV."""
    print("[INFO] Saving output data CSVs...")
    
    historical_path = os.path.join(DATA_DIR, 'historical_cleaned.csv')
    df_clean.to_csv(historical_path, index=False)
    print(f"  Saved: {historical_path} ({len(df_clean)} rows)")
    
    forecast_path = os.path.join(DATA_DIR, 'forecast_next_6_months.csv')
    forecast_df.to_csv(forecast_path, index=False)
    print(f"  Saved: {forecast_path} ({len(forecast_df)} rows)")
    
    evaluation_path = os.path.join(DATA_DIR, 'model_evaluation.csv')
    metrics.to_csv(evaluation_path, index=False)
    print(f"  Saved: {evaluation_path} ({len(metrics)} rows)")


def main():
    print("=" * 60)
    print("  Task 3: Predictive Analytics Pipeline")
    print("=" * 60)
    
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Step 1: Generate time-series
    df = generate_time_series()
    
    # Step 2: Feature engineering
    df_clean = engineer_features(df)
    
    # Step 3: Train and evaluate
    results = train_and_evaluate(df_clean)
    
    # Step 4: Run recursive forecast on final model
    forecast_df = make_forecast(df_clean, results['rf_model'], results['features'])
    
    # Step 5: Save results
    save_results(df_clean, forecast_df, results['metrics'])
    
    # Step 6: Plot charts
    plot_and_save_charts(
        df_clean, forecast_df, results['metrics'],
        results['y_test'], results['y_pred_rf'],
        results['rf_model'], results['features']
    )
    
    print("\n" + "=" * 60)
    print("  Predictive Analytics Run Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
