# ============================================================
#   SALES & DEMAND FORECASTING - Complete ML Project
#   For: Future Interns Machine Learning Task 1 (2026)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta


# ============================================================
# STEP 1: GENERATE REALISTIC SAMPLE SALES DATA
# (Simulates 2 years of daily sales for a retail store)
# ============================================================

def generate_sales_data():
    """Generate realistic retail sales data with trends and seasonality."""
    print("=" * 60)
    print("  SALES & DEMAND FORECASTING SYSTEM")
    print("=" * 60)
    print("\n[1/6] Generating sales dataset...\n")

    np.random.seed(42)

    # Date range: Jan 2023 to Dec 2024 (2 full years)
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    n = len(dates)

    # --- Base trend (slowly growing business) ---
    trend = np.linspace(200, 400, n)

    # --- Weekly seasonality (weekends sell more) ---
    weekly = np.array([1.0, 1.05, 1.1, 1.1, 1.2, 1.5, 1.4])  # Mon-Sun
    weekly_pattern = np.array([weekly[d.weekday()] for d in dates])

    # --- Monthly seasonality (holiday peaks) ---
    monthly = {
        1: 0.85, 2: 0.80, 3: 0.90, 4: 0.95, 5: 1.00,
        6: 1.05, 7: 1.10, 8: 1.05, 9: 0.95,
        10: 1.00, 11: 1.30, 12: 1.50  # Nov/Dec = holiday season
    }
    monthly_pattern = np.array([monthly[d.month] for d in dates])

    # --- Random noise ---
    noise = np.random.normal(0, 20, n)

    # --- Final sales ---
    sales = (trend * weekly_pattern * monthly_pattern + noise).clip(min=0)
    sales = sales.round(2)

    df = pd.DataFrame({
        "date": dates,
        "sales": sales,
        "category": np.random.choice(
            ["Electronics", "Clothing", "Grocery", "Furniture"],
            size=n,
            p=[0.25, 0.30, 0.35, 0.10]
        )
    })

    print(f"  ✅ Dataset created: {len(df)} rows | {df['date'].min().date()} to {df['date'].max().date()}")
    return df


# ============================================================
# STEP 2: DATA CLEANING & EXPLORATION
# ============================================================

def clean_and_explore(df):
    print("\n[2/6] Cleaning and exploring data...\n")

    # --- Check for missing values ---
    missing = df.isnull().sum()
    print(f"  Missing values:\n{missing}\n")

    # --- Fill missing if any ---
    df["sales"] = df["sales"].fillna(df["sales"].median())

    # --- Basic stats ---
    print("  📊 Sales Summary:")
    print(f"     Total Sales (2 years): ₹{df['sales'].sum():,.2f}")
    print(f"     Average Daily Sales:   ₹{df['sales'].mean():,.2f}")
    print(f"     Max Sales in a Day:    ₹{df['sales'].max():,.2f}")
    print(f"     Min Sales in a Day:    ₹{df['sales'].min():,.2f}")

    # --- Monthly totals ---
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["sales"].sum().reset_index()
    monthly["month"] = monthly["month"].astype(str)

    print(f"\n  Top 3 months by sales:")
    top3 = monthly.nlargest(3, "sales")
    for _, row in top3.iterrows():
        print(f"     {row['month']}: ₹{row['sales']:,.2f}")

    return df


# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================

def create_features(df):
    print("\n[3/6] Creating time-based features...\n")

    df = df.copy()

    # --- Date features ---
    df["year"]        = df["date"].dt.year
    df["month_num"]   = df["date"].dt.month
    df["day"]         = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek        # 0=Mon, 6=Sun
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"]= df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]     = df["date"].dt.quarter
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # --- Is holiday month (Nov, Dec) ---
    df["is_holiday_month"] = df["month_num"].isin([11, 12]).astype(int)

    # --- Lag features (past sales) ---
    df = df.sort_values("date").reset_index(drop=True)
    df["lag_1"]  = df["sales"].shift(1)    # yesterday's sales
    df["lag_7"]  = df["sales"].shift(7)    # same day last week
    df["lag_30"] = df["sales"].shift(30)   # same day last month

    # --- Rolling averages ---
    df["rolling_7d"]  = df["sales"].shift(1).rolling(7).mean()   # 7-day avg
    df["rolling_30d"] = df["sales"].shift(1).rolling(30).mean()  # 30-day avg

    # --- Encode category ---
    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category"])

    # --- Drop rows with NaN (from lag features) ---
    df = df.dropna().reset_index(drop=True)

    print(f"  ✅ Features created. Dataset shape: {df.shape}")
    print(f"  Features: year, month, day, day_of_week, quarter, is_weekend,")
    print(f"            is_holiday_month, lag_1, lag_7, lag_30,")
    print(f"            rolling_7d, rolling_30d, category_encoded")

    return df, le


# ============================================================
# STEP 4: TRAIN ML MODELS & EVALUATE
# ============================================================

def train_and_evaluate(df):
    print("\n[4/6] Training ML models...\n")

    feature_cols = [
        "year", "month_num", "day", "day_of_week", "day_of_year",
        "week_of_year", "quarter", "is_weekend", "is_holiday_month",
        "lag_1", "lag_7", "lag_30", "rolling_7d", "rolling_30d",
        "category_encoded"
    ]

    X = df[feature_cols]
    y = df["sales"]

    # --- Train/Test split (last 60 days = test) ---
    split_idx = len(df) - 60
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"  Training set: {len(X_train)} days")
    print(f"  Test set:     {len(X_test)} days (last 60 days)\n")

    # --- Models to compare ---
    models = {
        "Linear Regression":   LinearRegression(),
        "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    best_model = None
    best_mae = float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)

        results[name] = {
            "model": model,
            "predictions": preds,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4)
        }

        print(f"  📌 {name}")
        print(f"     MAE  = ₹{mae:.2f}  (avg error per day)")
        print(f"     RMSE = ₹{rmse:.2f}")
        print(f"     R²   = {r2:.4f} ({r2*100:.1f}% variance explained)\n")

        if mae < best_mae:
            best_mae = mae
            best_model = name

    print(f"  🏆 Best Model: {best_model} (lowest MAE = ₹{best_mae:.2f})")

    return results, best_model, X_test, y_test, X_train, y_train, df, feature_cols


# ============================================================
# STEP 5: FORECAST NEXT 30 DAYS
# ============================================================

def forecast_future(df, results, best_model, feature_cols):
    print("\n[5/6] Forecasting next 30 days...\n")

    model = results[best_model]["model"]
    last_date = df["date"].max()
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=30, freq="D"
    )

    forecasts = []
    temp_df = df.copy()

    for fdate in forecast_dates:
        last_row = temp_df.tail(1).iloc[0]

        # Build a feature row for the forecast date
        row = {
            "date":             fdate,
            "year":             fdate.year,
            "month_num":        fdate.month,
            "day":              fdate.day,
            "day_of_week":      fdate.dayofweek,
            "day_of_year":      fdate.dayofyear,
            "week_of_year":     fdate.isocalendar()[1],
            "quarter":          (fdate.month - 1) // 3 + 1,
            "is_weekend":       int(fdate.weekday() >= 5),
            "is_holiday_month": int(fdate.month in [11, 12]),
            "lag_1":            last_row["sales"],
            "lag_7":            temp_df["sales"].iloc[-7] if len(temp_df) >= 7 else last_row["sales"],
            "lag_30":           temp_df["sales"].iloc[-30] if len(temp_df) >= 30 else last_row["sales"],
            "rolling_7d":       temp_df["sales"].iloc[-7:].mean(),
            "rolling_30d":      temp_df["sales"].iloc[-30:].mean(),
            "category_encoded": 2,  # default: Grocery
            "sales":            0,
            "category":         "Grocery"
        }

        X_pred = pd.DataFrame([row])[feature_cols]
        predicted_sales = model.predict(X_pred)[0]
        row["sales"] = predicted_sales

        forecasts.append({"date": fdate, "forecasted_sales": round(predicted_sales, 2)})

        new_row = pd.DataFrame([row])
        temp_df = pd.concat([temp_df, new_row], ignore_index=True)

    forecast_df = pd.DataFrame(forecasts)

    print(f"  📅 Forecast Period: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
    print(f"  💰 Estimated Total Sales (next 30 days): ₹{forecast_df['forecasted_sales'].sum():,.2f}")
    print(f"  📈 Avg Daily Forecast: ₹{forecast_df['forecasted_sales'].mean():,.2f}")
    print(f"\n  Daily Forecast Preview:")
    print(f"  {'Date':<15} {'Day':<12} {'Forecast (₹)'}")
    print(f"  {'-'*40}")
    for _, row in forecast_df.iterrows():
        day_name = row["date"].strftime("%A")
        print(f"  {str(row['date'].date()):<15} {day_name:<12} ₹{row['forecasted_sales']:>10,.2f}")

    return forecast_df


# ============================================================
# STEP 6: VISUALIZATIONS (Business-Friendly Charts)
# ============================================================

def create_visualizations(df, results, best_model, X_test, y_test, forecast_df):
    print("\n[6/6] Creating business-friendly visualizations...\n")

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "actual":    "#2196F3",
        "forecast":  "#FF5722",
        "predicted": "#4CAF50",
        "bar":       "#9C27B0"
    }

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("Sales & Demand Forecasting Dashboard", fontsize=22, fontweight="bold", y=0.98)

    # -----------------------------------------------------------
    # CHART 1: Historical Sales Overview (Monthly)
    # -----------------------------------------------------------
    ax1 = fig.add_subplot(4, 2, (1, 2))
    monthly_sales = df.groupby(df["date"].dt.to_period("M"))["sales"].sum()
    monthly_sales.index = monthly_sales.index.to_timestamp()

    ax1.bar(monthly_sales.index, monthly_sales.values,
            color=colors["bar"], alpha=0.7, width=20, label="Monthly Sales")
    ax1.plot(monthly_sales.index, monthly_sales.values,
             color=colors["actual"], linewidth=2, marker="o", markersize=5, label="Trend")
    ax1.set_title("📊 Monthly Sales Overview (Historical)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Total Sales (₹)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

    # -----------------------------------------------------------
    # CHART 2: Actual vs Predicted (Best Model - last 60 days)
    # -----------------------------------------------------------
    ax2 = fig.add_subplot(4, 2, (3, 4))
    test_dates = df["date"].iloc[-len(y_test):]
    best_preds = results[best_model]["predictions"]

    ax2.plot(test_dates.values, y_test.values,
             color=colors["actual"], linewidth=2, label="Actual Sales", alpha=0.9)
    ax2.plot(test_dates.values, best_preds,
             color=colors["predicted"], linewidth=2, linestyle="--",
             label=f"Predicted ({best_model})", alpha=0.9)
    ax2.fill_between(test_dates.values, y_test.values, best_preds,
                     alpha=0.15, color="gray", label="Error Gap")
    ax2.set_title(f"✅ Actual vs Predicted Sales — {best_model} (Last 60 Days)",
                  fontsize=14, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Daily Sales (₹)")
    ax2.legend()
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

    # -----------------------------------------------------------
    # CHART 3: 30-Day Future Forecast
    # -----------------------------------------------------------
    ax3 = fig.add_subplot(4, 2, (5, 6))
    recent_sales = df.tail(60)

    ax3.plot(recent_sales["date"].values, recent_sales["sales"].values,
             color=colors["actual"], linewidth=2, label="Historical (last 60 days)")
    ax3.plot(forecast_df["date"].values, forecast_df["forecasted_sales"].values,
             color=colors["forecast"], linewidth=2.5, linestyle="--",
             marker="o", markersize=4, label="Forecast (next 30 days)")
    ax3.axvline(x=df["date"].max(), color="gray", linestyle=":", linewidth=1.5,
                label="Today")
    ax3.fill_between(forecast_df["date"].values,
                     forecast_df["forecasted_sales"].values * 0.90,
                     forecast_df["forecasted_sales"].values * 1.10,
                     alpha=0.2, color=colors["forecast"], label="±10% Confidence")
    ax3.set_title("🔮 30-Day Sales Forecast", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Forecasted Sales (₹)")
    ax3.legend()
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

    # -----------------------------------------------------------
    # CHART 4: Model Comparison (MAE)
    # -----------------------------------------------------------
    ax4 = fig.add_subplot(4, 2, 7)
    model_names = list(results.keys())
    maes  = [results[m]["MAE"]  for m in model_names]
    r2s   = [results[m]["R2"]   for m in model_names]
    bar_colors = ["#FF5722" if m == best_model else "#90A4AE" for m in model_names]

    bars = ax4.bar(model_names, maes, color=bar_colors, edgecolor="white", linewidth=1.5)
    for bar, mae in zip(bars, maes):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"₹{mae}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax4.set_title("📉 Model Comparison (MAE — Lower is Better)",
                  fontsize=13, fontweight="bold")
    ax4.set_ylabel("Mean Absolute Error (₹)")
    ax4.set_xticklabels(model_names, rotation=10)

    # -----------------------------------------------------------
    # CHART 5: Sales by Day of Week
    # -----------------------------------------------------------
    ax5 = fig.add_subplot(4, 2, 8)
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_sales = df.groupby("day_of_week")["sales"].mean().reindex(range(7))
    bar_colors_dow = ["#FF5722" if i >= 5 else "#2196F3" for i in range(7)]

    bars2 = ax5.bar(day_names, dow_sales.values, color=bar_colors_dow, edgecolor="white")
    for bar, val in zip(bars2, dow_sales.values):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"₹{val:,.0f}", ha="center", va="bottom", fontsize=9)
    ax5.set_title("📅 Avg Sales by Day of Week", fontsize=13, fontweight="bold")
    ax5.set_ylabel("Avg Daily Sales (₹)")
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("sales_forecast_dashboard.png", dpi=150, bbox_inches="tight")
    print("  ✅ Dashboard saved: sales_forecast_dashboard.png")
    plt.show()


# ============================================================
# STEP 7: BUSINESS SUMMARY REPORT
# ============================================================

def print_business_report(df, forecast_df, results, best_model):
    total_hist   = df["sales"].sum()
    avg_daily    = df["sales"].mean()
    best_month   = df.groupby("month_num")["sales"].mean().idxmax()
    month_names  = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    next30_total = forecast_df["forecasted_sales"].sum()
    next30_avg   = forecast_df["forecasted_sales"].mean()
    mae          = results[best_model]["MAE"]
    r2           = results[best_model]["R2"]

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║          📋 BUSINESS SALES FORECAST REPORT                  ║
╠══════════════════════════════════════════════════════════════╣
║  HISTORICAL PERFORMANCE (2 Years)                            ║
║  ─────────────────────────────────────────────────────────  ║
║  Total Revenue:        ₹{total_hist:>15,.2f}                 ║
║  Average Daily Sales:  ₹{avg_daily:>15,.2f}                  ║
║  Best Sales Month:     {month_names[best_month]:<15}                        ║
╠══════════════════════════════════════════════════════════════╣
║  ML MODEL PERFORMANCE                                        ║
║  ─────────────────────────────────────────────────────────  ║
║  Best Model:  {best_model:<20}                        ║
║  Accuracy:    R² = {r2:.4f} ({r2*100:.1f}% of sales explained)      ║
║  Avg Error:   ₹{mae:.2f} per day                             ║
╠══════════════════════════════════════════════════════════════╣
║  NEXT 30-DAY FORECAST                                        ║
║  ─────────────────────────────────────────────────────────  ║
║  Expected Total Sales: ₹{next30_total:>15,.2f}               ║
║  Expected Daily Avg:   ₹{next30_avg:>15,.2f}                 ║
╠══════════════════════════════════════════════════════════════╣
║  💡 BUSINESS RECOMMENDATIONS                                 ║
║  ─────────────────────────────────────────────────────────  ║
║  1. Stock up inventory before Nov-Dec (holiday spike)        ║
║  2. Plan extra staff for weekends (35% higher sales)         ║
║  3. Use forecast to set weekly purchase budgets              ║
║  4. Target promotions in low months (Feb, Mar)               ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(report)


# ============================================================
# MAIN — Run Everything
# ============================================================

if __name__ == "__main__":

    # 1. Generate data
    df = generate_sales_data()

    # 2. Clean & explore
    df = clean_and_explore(df)

    # 3. Feature engineering
    df, label_encoder = create_features(df)

    # 4. Train models & evaluate
    results, best_model, X_test, y_test, X_train, y_train, df, feature_cols = train_and_evaluate(df)

    # 5. Forecast next 30 days
    forecast_df = forecast_future(df, results, best_model, feature_cols)

    # 6. Visualizations
    create_visualizations(df, results, best_model, X_test, y_test, forecast_df)

    # 7. Business report
    print_business_report(df, forecast_df, results, best_model)

    print("\n  🎉 Project Complete! Files saved:")
    print("     📊 sales_forecast_dashboard.png")
    print("     📄 sales_forecast.py\n")