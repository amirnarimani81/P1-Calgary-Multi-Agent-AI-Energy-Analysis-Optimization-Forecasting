


=========================================================
# Multi-Agent AI Energy Consumpotion & Forecasting Platform
# =========================================================
import warnings
warnings.filterwarnings('ignore')

# Core imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta
# Machine Learning
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency

from scipy.stats import linregress
from sklearn.metrics import r2_score
import statsmodels.api as sm
# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# LangChain
from langchain_community.llms import Ollama
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import ollama




# =========================================================
# LLM CALL FUNCTION

# LLM
def call_llm(prompt):
    try:
        response = ollama.chat(model="llama3.2:latest",
                               messages=[{"role":"user","content":prompt}],
            options={"temperature":0.2})

        return response["message"]["content"]
    except Exception as e:
        return str(e)


def energy_analysis_agent(module, results):
    prompt=f"""
You are a Senior Energy Market Data Scientist.
Analyze the following energy market analysis.

Module:
{module}

Results:
{json.dumps(results, indent=2, default=str)}

Provide:

1. Executive Summary
2. Key Findings
3. Market Risks
4. Statistical Interpretation
5. Forecasting Readiness
6. Business Impact
7. Feature Engineering Ideas
8. Business Recommendations
9. Data Quality Score (0-100)
10. Forecast Readiness Score (0-100)

Write professionally for executives.

"""
    return call_llm(prompt)
# =========================================================
# LOAD DATA
# =========================================================
def load_data(file):
    df = pd.read_csv(file)
    # Handle datetime column (assuming 'Date' column exists)
    df['datetime'] = pd.to_datetime(df['Date'])
    df = df.sort_values('datetime').set_index('datetime')
    return df




# =========================
#  DATA CHECK FUNCTION
# =========================
def check_df(df, name="Dataset", head=5):

    report = {}

    report["shape"] = {
        "rows": df.shape[0],
        "columns": df.shape[1]}

    report["dtypes"] = df.dtypes.astype(str).to_dict()
    report["dtype_summary"] = df.dtypes.value_counts().astype(str).to_dict()
    report["missing_values"] = df.isnull().sum().to_dict()
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["head"] = df.head(head).to_dict()
    report["tail"] = df.tail(head).to_dict()
    report["describe"] = df.describe().to_dict()

    return report


eda_report = check_df(df)

compact_report = {
    "shape": eda_report["shape"],
    "dtypes": eda_report["dtypes"],
    "missing_values": eda_report["missing_values"],
    "duplicate_rows": eda_report["duplicate_rows"],
    "describe": eda_report["describe"]}



# =========================================================
# DATA CLEANING & OUTLIER DETECTION
# =========================================================
def clean_data(df, target_col='Electricity_Price_CAD_MWh', iqr_multiplier=1.5, ai_consultant=None):
    
    # Get full date range information
    start_date = df.index.min()
    end_date = df.index.max()
    total_hours = len(df)
    total_days = total_hours / 24

    print("=" * 50)
    print("DATE RANGE ANALYSIS")
    print("=" * 50)
    print(f"Start date:     {start_date}")
    print(f"End date:       {end_date}")
    print(f"Total hours:    {total_hours:,}")
    print(f"Total days:     {total_days:.1f}")
    print(f"Total months:   {total_days / 30.4:.1f}")
    print(f"Total years:    {total_days / 365.25:.2f}")
    print("=" * 50)

    # Check for missing dates
    expected_hours = pd.date_range(start=start_date, end=end_date, freq='h')
    missing_hours = expected_hours.difference(df.index)
    print(f"\nMissing hours:  {len(missing_hours):,} ({len(missing_hours)/len(expected_hours)*100:.1f}%)")
    
    df = df.copy()
    initial_rows = len(df)
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    
    # Convert numeric columns
    numeric_cols = ['Electricity_Price_CAD_MWh', 'AIL_Total_Demand_MW', 'Temperature_C', 'Humidity_%']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Label encode categorical columns
    categorical_cols = ["Season", "City", "Peak_Hour", "Holiday", "Grid_Stress_Level"]
    encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    # Track missing values before
    missing_before = df[['AIL_Total_Demand_MW', 'Electricity_Price_CAD_MWh']].isnull().sum()
    
    # Fill missing values
    df['AIL_Total_Demand_MW'] = df['AIL_Total_Demand_MW'].interpolate(method='linear')
    df['Electricity_Price_CAD_MWh'] = df['Electricity_Price_CAD_MWh'].interpolate(method='linear')
    df['AIL_Total_Demand_MW'] = df['AIL_Total_Demand_MW'].ffill().bfill()
    df['Electricity_Price_CAD_MWh'] = df['Electricity_Price_CAD_MWh'].ffill().bfill()
    
    # Interpolate missing values for target column
    if target_col in df.columns:
        df[target_col] = df[target_col].interpolate(method='linear')
    
    # Outlier detection with IQR (capping, not removal)
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    outlier_count = ((df[target_col] < lower_bound) | (df[target_col] > upper_bound)).sum()
    
    df.loc[df[target_col] < lower_bound, target_col] = lower_bound
    df.loc[df[target_col] > upper_bound, target_col] = upper_bound
    
    # Track missing values after
    missing_after = df[['AIL_Total_Demand_MW', 'Electricity_Price_CAD_MWh']].isnull().sum()
    has_nan = df[['AIL_Total_Demand_MW', 'Electricity_Price_CAD_MWh']].isnull().any().any()
    
    # Create cleaning report
    report = {
        "status": "success" if not has_nan else "partial",
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "rows_removed": initial_rows - len(df),
        "duplicates_removed": int(duplicates),
        "outliers_capped": int(outlier_count),
        "missing_before_demand": int(missing_before.get('AIL_Total_Demand_MW', 0)),
        "missing_before_price": int(missing_before.get('Electricity_Price_CAD_MWh', 0)),
        "missing_after_demand": int(missing_after.get('AIL_Total_Demand_MW', 0)),
        "missing_after_price": int(missing_after.get('Electricity_Price_CAD_MWh', 0)),
        "has_nan_remaining": has_nan,
        "cleaning_percentage": round((1 - len(df)/initial_rows) * 100, 2),
        "date_range": {
            "start": str(start_date),
            "end": str(end_date),
            "total_hours": total_hours,
            "missing_hours": len(missing_hours),
            "missing_percentage": round(len(missing_hours)/len(expected_hours)*100, 2)}}
    
    # AI analysis
    if ai_consultant:
        ai_insight = ai_consultant._call_llm(f"Data quality analysis: {report}")
    else:
        ai_insight = None
    
    return df, report, ai_insight




def plot_iqr(df, target_col='Electricity_Price_CAD_MWh', iqr_multiplier=1.5):
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - iqr_multiplier * IQR
    upper = Q3 + iqr_multiplier * IQR
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(df[target_col], vert=False)
    ax.axvline(x=lower, color='red', linestyle='--', label=f'Lower Bound: {lower:.2f}')
    ax.axvline(x=upper, color='red', linestyle='--', label=f'Upper Bound: {upper:.2f}')
    ax.axvline(x=Q1, color='blue', linestyle=':', label=f'Q1: {Q1:.2f}')
    ax.axvline(x=Q3, color='blue', linestyle=':', label=f'Q3: {Q3:.2f}')
    ax.set_xlabel('Price ($/MWh)')
    ax.set_title(f'IQR Analysis - {target_col}')
    ax.legend()
    return fig


# ============================================================================
# sql_query(df, target_col)
# ============================================================================

def run_sql_queries(df, ai_consultant=None):
    # Prepare data for SQL
    sql_df = df.reset_index().copy()
    sql_df['hour'] = sql_df['datetime'].dt.hour
    sql_df['month'] = sql_df['datetime'].dt.month
    sql_df['weekday'] = sql_df['datetime'].dt.weekday
    sql_df['is_weekend'] = (sql_df['weekday'] >= 5).astype(int)
    
    conn = sqlite3.connect(":memory:")
    sql_df.to_sql("energy", conn, index=False, if_exists="replace")
    
    results = {}
    ai_insights = {}
    
    # Helper function
    def run_query(key, query, ai_prompt=None, post_process=None):
        result = pd.read_sql_query(query, conn)
        results[key] = result
        
        if ai_consultant and ai_prompt:
            try:
                context = result.to_dict()
                ai_insights[key] = ai_consultant._call_llm(
                    f"{ai_prompt}\nData: {context}",
                    "Energy market analysis"
                )
            except Exception as e:
                ai_insights[key] = f"AI error: {str(e)}"
        
        if post_process:
            post_process(result)
    
    # Q1 Peak Hours
    run_query(
        "peak_hours",
        """SELECT hour, ROUND(AVG(Electricity_Price_CAD_MWh), 2) as avg_price
           FROM energy GROUP BY hour ORDER BY avg_price DESC LIMIT 5""",
        "Analyze expensive electricity hours and explain demand behavior" )
    
    # Q2 Cheap Hours
    run_query(
        "cheap_hours",
        """SELECT hour, ROUND(AVG(Electricity_Price_CAD_MWh), 2) as avg_price
           FROM energy GROUP BY hour ORDER BY avg_price ASC LIMIT 5""",
        "Analyze cheapest electricity hours for cost optimization")
    
    # Q3 Weekend Effect
    def weekend_ai(df_res):
        try:
            wk = df_res[df_res["day_type"] == "Weekday"]["avg_price"].values[0]
            we = df_res[df_res["day_type"] == "Weekend"]["avg_price"].values[0]
            discount = ((wk - we) / wk) * 100
            
            ai_insights["weekend_discount"] = ai_consultant._call_llm(
                f"Weekend discount: {discount:.1f}%, Weekday: {wk}, Weekend: {we}",
                "Weekend pricing pattern analysis")
        except:
            ai_insights["weekend_discount"] = "Insufficient data"
    
    run_query(
        "weekend_discount",
        """SELECT 
            CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END as day_type,
            ROUND(AVG(Electricity_Price_CAD_MWh), 2) as avg_price
           FROM energy GROUP BY is_weekend""",
        post_process=weekend_ai)
    
    # Q4 Seasonal Pattern
    run_query(
        "seasonal",
        """SELECT month, ROUND(AVG(Electricity_Price_CAD_MWh), 2) as avg_price
           FROM energy GROUP BY month ORDER BY month""",
        "Analyze seasonal electricity price patterns in Calgary")
    
    # Q5 Temperature Impact
    run_query(
        "temp_impact",
        """SELECT 
            CASE 
                WHEN Temperature_C < -10 THEN 'Extreme Cold'
                WHEN Temperature_C < 0 THEN 'Cold'
                WHEN Temperature_C < 10 THEN 'Mild'
                WHEN Temperature_C < 20 THEN 'Warm'
                ELSE 'Hot'
            END as temp_range,
            ROUND(AVG(Electricity_Price_CAD_MWh), 2) as avg_price
           FROM energy GROUP BY temp_range
           ORDER BY avg_price DESC""",
        "Explain how temperature affects electricity pricing")
    
    # Q6 Anomalies
    run_query(
        "anomalies",
        """SELECT datetime, hour, Electricity_Price_CAD_MWh as price
           FROM energy
           WHERE Electricity_Price_CAD_MWh > (SELECT AVG(Electricity_Price_CAD_MWh) + 2.5 * 
                 (SELECT AVG(Electricity_Price_CAD_MWh) FROM energy))
           LIMIT 10""",
        "Analyze abnormal price spikes and possible causes")
    
    # Q7 Peak Index
    run_query(
        "peak_index",
        """WITH peak AS (
            SELECT AVG(Electricity_Price_CAD_MWh) as p FROM energy WHERE hour IN (7,8,9,17,18,19)),
        offpeak AS (
            SELECT AVG(Electricity_Price_CAD_MWh) as op FROM energy WHERE hour BETWEEN 2 AND 5)
        SELECT p as peak_price, op as offpeak_price,
               ROUND(p / op, 2) as peak_index
        FROM peak, offpeak""")
    
    conn.close()
    return results, ai_insights



# =========================================================
# FEATURES
# =========================================================



def feature_engineering(df):
    df = df.copy()

    # Time-based features
    df['Hour'] = df.index.hour
    df['Day'] = df.index.day
    df['Weekday'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Weekend'] = df['Weekday'].isin([5,6]).astype(int)

    # Lag features
    df['AIL_lag_1'] = df['AIL_Total_Demand_MW'].shift(1)
    df['AIL_lag_24'] = df['AIL_Total_Demand_MW'].shift(24)

    # Rolling statistics
    df['Rolling_Load_24'] = df['AIL_Total_Demand_MW'].rolling(24).mean()
    df['Rolling_Price_24'] = df['Electricity_Price_CAD_MWh'].rolling(24).mean()
    df['Price_Volatility'] = df['Electricity_Price_CAD_MWh'].rolling(24).std()

    # Interaction features
    df['Total_Cost'] = df['AIL_Total_Demand_MW'] * df['Electricity_Price_CAD_MWh']
    df['Renewable_Ratio_%'] = (
        (df['Solar_Generation_kWh'] + df['Wind_Generation_kWh'])
        / (df['Electricity_kWh'] + 1)) * 100

    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Rows after feature engineering: {len(df)}")
    print(f"Rows after dropping NaN: {len(df.dropna())}")

    return df.dropna()

# Apply feature engineering
df_featured = feature_engineering(df_hourly)

# ============================================
# PLOTS FOR FEATURE VISUALIZATION
# ============================================

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Energy Forecasting - Engineered Features Analysis', fontsize=16, fontweight='bold')

# Plot 1: Hourly Pattern
hourly_pattern = df_featured.groupby('Hour')['AIL_Total_Demand_MW'].mean()
axes[0, 0].plot(hourly_pattern.index, hourly_pattern.values, marker='o', color='blue', linewidth=2)
axes[0, 0].set_title('1. Hour of Day Pattern', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Hour (0-23)')
axes[0, 0].set_ylabel('Avg Demand (MW)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=9, color='red', linestyle='--', alpha=0.5, label='Morning Peak')
axes[0, 0].axvline(x=18, color='orange', linestyle='--', alpha=0.5, label='Evening Peak')
axes[0, 0].legend()

# Plot 2: Weekly Pattern
weekly_pattern = df_featured.groupby('Weekday')['AIL_Total_Demand_MW'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[0, 1].bar(days, weekly_pattern.values, color=['steelblue']*5 + ['lightcoral']*2)
axes[0, 1].set_title('2. Day of Week Pattern', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Weekday')
axes[0, 1].set_ylabel('Avg Demand (MW)')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Monthly Pattern
monthly_pattern = df_featured.groupby('Month')['AIL_Total_Demand_MW'].mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
axes[0, 2].plot(range(1,13), monthly_pattern.values, marker='s', color='green', linewidth=2)
axes[0, 2].set_title('3. Monthly Pattern', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Month')
axes[0, 2].set_ylabel('Avg Demand (MW)')
axes[0, 2].set_xticks(range(1,13))
axes[0, 2].set_xticklabels(months, rotation=45)
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Lag 1 vs Current (autocorrelation)
sample_data = df_featured.sample(min(1000, len(df_featured)))
axes[1, 0].scatter(sample_data['AIL_lag_1'], sample_data['AIL_Total_Demand_MW'], 
                   alpha=0.5, s=10, color='purple')
axes[1, 0].plot([df_featured['AIL_Total_Demand_MW'].min(), df_featured['AIL_Total_Demand_MW'].max()],
                [df_featured['AIL_Total_Demand_MW'].min(), df_featured['AIL_Total_Demand_MW'].max()],
                'k--', alpha=0.5, label='Perfect correlation')
axes[1, 0].set_title('4. Lag 1 (Previous Hour) Correlation', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Demand 1 hour ago (MW)')
axes[1, 0].set_ylabel('Current Demand (MW)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Lag 24 vs Current
axes[1, 1].scatter(sample_data['AIL_lag_24'], sample_data['AIL_Total_Demand_MW'], 
                   alpha=0.5, s=10, color='orange')
axes[1, 1].plot([df_featured['AIL_Total_Demand_MW'].min(), df_featured['AIL_Total_Demand_MW'].max()],
                [df_featured['AIL_Total_Demand_MW'].min(), df_featured['AIL_Total_Demand_MW'].max()],
                'k--', alpha=0.5)
axes[1, 1].set_title('5. Lag 24 (Same Hour Yesterday) Correlation', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Demand 24 hours ago (MW)')
axes[1, 1].set_ylabel('Current Demand (MW)')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Rolling Load (Trend)
sample_week = df_featured.tail(168)  # Last 7 days
axes[1, 2].plot(sample_week.index, sample_week['AIL_Total_Demand_MW'], 
                label='Actual', alpha=0.5, linewidth=1)
axes[1, 2].plot(sample_week.index, sample_week['Rolling_Load_24'], 
                label='24h Rolling Avg', color='red', linewidth=2)
axes[1, 2].set_title('6. Rolling Load (24h Moving Average)', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Date')
axes[1, 2].set_ylabel('Demand (MW)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

# Plot 7: Price Volatility
axes[2, 0].plot(sample_week.index, sample_week['Price_Volatility'], 
                color='red', linewidth=1.5)
axes[2, 0].fill_between(sample_week.index, 0, sample_week['Price_Volatility'], 
                        alpha=0.3, color='red')
axes[2, 0].set_title('7. Price Volatility (24h Std Dev)', fontsize=12, fontweight='bold')
axes[2, 0].set_xlabel('Date')
axes[2, 0].set_ylabel('Volatility ($/MWh)')
axes[2, 0].grid(True, alpha=0.3)

# Plot 8: Total Cost over time
axes[2, 1].plot(sample_week.index, sample_week['Total_Cost'] / 1e6, 
                color='green', linewidth=1.5)
axes[2, 1].set_title('8. Total Cost (Demand × Price)', fontsize=12, fontweight='bold')
axes[2, 1].set_xlabel('Date')
axes[2, 1].set_ylabel('Total Cost (Millions $/h)')
axes[2, 1].grid(True, alpha=0.3)

# Plot 9: Renewable Ratio Distribution
axes[2, 2].hist(df_featured['Renewable_Ratio_%'].dropna(), bins=50, 
                color='green', alpha=0.7, edgecolor='black')
axes[2, 2].axvline(df_featured['Renewable_Ratio_%'].median(), 
                   color='red', linestyle='--', label=f"Median: {df_featured['Renewable_Ratio_%'].median():.1f}%")
axes[2, 2].set_title('9. Renewable Energy Ratio Distribution', fontsize=12, fontweight='bold')
axes[2, 2].set_xlabel('Renewable Ratio (%)')
axes[2, 2].set_ylabel('Frequency')
axes[2, 2].legend()
axes[2, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================
# CORRELATION HEATMAP
# ============================================
plt.figure(figsize=(12, 8))
feature_cols = ['Hour', 'Weekday', 'Month', 'Weekend', 
                'AIL_lag_1', 'AIL_lag_24', 
                'Rolling_Load_24', 'Rolling_Price_24', 
                'Price_Volatility', 'Total_Cost', 'Renewable_Ratio_%',
                'AIL_Total_Demand_MW', 'Electricity_Price_CAD_MWh']

correlation_matrix = df_featured[feature_cols].corr()

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================
# OLLAMA AI EXPLANATION
# ============================================
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama")

prompt = """
You are an expert energy data scientist.

Explain ONLY the following engineered features in a simple and clear way.
Do NOT analyze data. Do NOT use dataset values.

FEATURES:

1. Hour = df.index.hour
2. Day = df.index.day
3. Weekday = df.index.dayofweek
4. Month = df.index.month
5. Year = df.index.year
6. Weekend = df['Weekday'].isin([5,6])

7. AIL_lag_1 = previous hour demand
8. AIL_lag_24 = demand 24 hours ago

9. Rolling_Load_24 = 24-hour moving average of demand
10. Rolling_Price_24 = 24-hour moving average of price

11. Price_Volatility = 24-hour rolling standard deviation of price

12. Total_Cost = demand × price

13. Renewable_Ratio_% = (solar + wind) / total electricity

Explain:
- meaning of each feature
- why it is used in energy forecasting
- how it helps prediction models
"""

response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": prompt}])

print("\n" + "="*80)
print("AI EXPLANATION OF ENGINEERED FEATURES")
print("="*80)
print(response.choices[0].message.content)
print("="*80)

# =========================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
def exploratory_data_analysis(df):

    df = df.copy()

    # =========================
    # Consumption distribution
    # =========================
    plt.figure(figsize=(10, 5))
    sns.histplot(df["AIL_Total_Demand_MW"], bins=50, kde=True)
    plt.title("Energy Load Distribution")
    plt.show()

    # =========================
    # Electricity price distribution
    # =========================
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Electricity_Price_CAD_MWh"], bins=50, kde=True)
    plt.title("Electricity Price Distribution")
    plt.show()

    # =========================
    # Gas price distribution
    # =========================
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Gas_Price_CAD_GJ"], bins=50, kde=True)
    plt.title("Gas Price Distribution")
    plt.show()

    # =========================
    # Correlation matrix
    # =========================
   
    correlation_matrix = df.corr(numeric_only=True)
    correlation_price = correlation_matrix['Electricity_Price_CAD_MWh'].sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 5))
    sns.heatmap(correlation_price.to_frame(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation with Electricity_Price_CAD_MWh')
    plt.show()

    # =========================
    # Price vs Demand
    # =========================
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=df["AIL_Total_Demand_MW"],
        y=df["Electricity_Price_CAD_MWh"]
    )
    plt.title("Demand vs Price")
    plt.show()

   
    return df

# =========================================================
# TIME-SERIES ANALYSIS
# =========================================================
def time_series_analysis(df, use_llm=False, client=None):

    df = df.copy()

    # =========================
    # CLEANING
    # =========================
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.index = pd.to_datetime(df.index)

    # =========================
    # HOURLY PRICE PATTERN
    # =========================
    hourly_avg = df.groupby(df.index.hour)['Electricity_Price_CAD_MWh'].mean()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, marker='o')
    plt.title("Average Hourly Price Pattern")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Price ($/MWh)")
    plt.xticks(range(0, 24))
    plt.grid(alpha=0.3)
    plt.show()

    # =========================
    # WEEKLY PATTERN
    # =========================
    weekday_avg = df.groupby(df.index.dayofweek)['Electricity_Price_CAD_MWh'].mean()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=weekday_avg.index, y=weekday_avg.values, marker='o')
    plt.title("Average Weekly Price Pattern")
    plt.xlabel("Day of Week (0=Mon, 6=Sun)")
    plt.ylabel("Average Price ($/MWh)")
    plt.xticks(range(0, 7))
    plt.grid(alpha=0.3)
    plt.show()

    # =========================
    # HEATMAP (Weekend vs Weekday)
    # =========================
    df["Hour"] = df.index.hour

    df["Day_Type"] = df["Weekend"].map({
        1: "Weekend",
        0: "Weekday"})

    heatmap_data = (df.groupby(["Hour", "Day_Type"])["AIL_Total_Demand_MW"].mean().unstack())

    plt.figure(figsize=(12, 6))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="viridis")

    plt.title("Energy Demand: Weekday vs Weekend")
    plt.xlabel("Day Type")
    plt.ylabel("Hour of Day")

    plt.show()

    # =========================
    # YEARLY TREND
    # =========================
    yearly = df.groupby(df.index.year)["AIL_Total_Demand_MW"].sum()

    yearly.plot(kind="line", marker="o")
    plt.title("Yearly Energy Consumption")
    plt.xlabel("Year")
    plt.ylabel("Total Consumption (MWh)")
    plt.grid(alpha=0.3)
    plt.show()

    # =========================
    # PRICE vs DEMAND
    # =========================
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        x=df["AIL_Total_Demand_MW"],
        y=df["Electricity_Price_CAD_MWh"])

    plt.title("Price vs Demand")
    plt.xlabel("Hourly Demand")
    plt.ylabel("Hourly Price ($/MWh)")
    plt.grid(alpha=0.3)
    plt.show()

    # =========================
    # MONTHLY TREND
    # =========================
    monthly = df.groupby(df.index.month)["AIL_Total_Demand_MW"].mean()

    plt.figure(figsize=(10, 6))

    monthly.plot(kind="line", marker="o")

    plt.title("Monthly Consumption Pattern")
    plt.xlabel("Month")
    plt.ylabel("Average Consumption (MW)")
    plt.xticks(range(1, 13))
    plt.grid(alpha=0.3)

    plt.show()

    # =========================
    # LLM INPUT
    # =========================
    llm_output = None

    if use_llm and client is not None:

        summary = {
            "hourly_price_pattern": hourly_avg.to_dict(),
            "weekly_price_pattern": weekday_avg.to_dict(),
            "yearly_demand_trend": yearly.to_dict(),
            "monthly_demand_pattern": monthly.to_dict(),

            "heatmap_weekday_weekend": heatmap_data.round(2).fillna(0).to_dict(),

            "correlation_price_demand": float(
                df["AIL_Total_Demand_MW"].corr(
                    df["Electricity_Price_CAD_MWh"])),

            "price_stats": {
                "mean": float(df["Electricity_Price_CAD_MWh"].mean()),
                "std": float(df["Electricity_Price_CAD_MWh"].std()),
                "min": float(df["Electricity_Price_CAD_MWh"].min()),
                "max": float(df["Electricity_Price_CAD_MWh"].max())}}

        prompt = f"""
You are a senior energy market strategist.

Analyze this time-series dataset:

{summary}

Focus on:
1. Hourly behavior
2. Weekly pattern
3. Weekend vs weekday load differences
4. Seasonal demand trends
5. Price-demand relationship
6. Operational insights for grid optimization
"""

        response = client.chat.completions.create(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )

        llm_output = response.choices[0].message.content

    return df, llm_output

# =========================================================
# seasonal_market_analysis
# =========================================================

def seasonal_market_analysis(df, ai_consultant=None):


    df = df.copy()

    # =========================
    # Monthly Price Pattern
    # =========================
    monthly_avg = df.groupby(df.index.month)['Electricity_Price_CAD_MWh'].mean()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=monthly_avg.index, y=monthly_avg.values)
    plt.title("Average Monthly Price Pattern")
    plt.xlabel("Month")
    plt.ylabel("Average Price ($/MWh)")
    plt.xticks(range(1, 13))
    plt.show()

    # =========================
    # Temperature Impact
    # =========================
    temp_avg = None

    if "Temperature_C" in df.columns:
        temp_bins = pd.cut(df["Temperature_C"], bins=10)
        temp_avg = df.groupby(temp_bins)['Electricity_Price_CAD_MWh'].mean()

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(temp_avg)), temp_avg.values, marker='o')
        plt.title("Price vs Temperature")
        plt.xlabel("Temperature Range")
        plt.ylabel("Average Price ($/MWh)")
        plt.xticks(
            range(len(temp_avg)),
            [str(b.left) for b in temp_avg.index],
            rotation=45)
        plt.show()

    # =========================
    # Seasonal Analysis
    # =========================
    if 'Season' in df.columns:

        if "Total_Cost" not in df.columns:
            df["Total_Cost"] = (
                df["AIL_Total_Demand_MW"] * df["Electricity_Price_CAD_MWh"])

        season_mean = df.groupby("Season")["AIL_Total_Demand_MW"].mean()
        season_max = df.groupby("Season")["AIL_Total_Demand_MW"].max()
        season_cost_mean = df.groupby("Season")["Total_Cost"].mean()
        season_cost_sum = df.groupby("Season")["Total_Cost"].sum()

        season_mean.plot(kind="bar", figsize=(8,5), title="Seasonal Energy Consumption")
        plt.show()

        season_max.plot(kind="bar", figsize=(8,5), title="Seasonal Peak Load")
        plt.show()

        season_cost_mean.plot(kind="bar", figsize=(8,5), title="Seasonal Cost (Mean)")
        plt.show()

        season_cost_sum.plot(kind="bar", figsize=(8,5), title="Seasonal Total Energy Cost")
        plt.show()

    # =====================================
        # Pie Charts
    # =====================================
    energy_counts = df['Season'].value_counts()
    funding_counts = df['Peak_Hour'].value_counts() 

    fig, ax = plt.subplots(1, 2, figsize=(14, 7)) 

    ax[0].pie(energy_counts, labels=energy_counts.index, autopct='%1.1f%%', startangle=140)
    ax[0].set_title('Seasonal') 
    ax[1].pie(funding_counts, labels=funding_counts.index, autopct='%1.1f%%', startangle=140)
    ax[1].set_title('Peak Hour') 
    plt.tight_layout()
    plt.show()


    # =========================
    # LLM Insight Layer
    # =========================
    ai_insight = None

    if ai_consultant is not None:

        payload = {
            "monthly_price_pattern": monthly_avg.to_dict(),
            "temperature_price_relation": temp_avg.to_dict() if temp_avg is not None else None}

        if 'Season' in df.columns:
            payload.update({
                "seasonal_consumption_mean": season_mean.to_dict(),
                "seasonal_peak_load": season_max.to_dict()})

        prompt = f"""
You are a senior energy market analyst (Calgary electricity system).

Analyze the following seasonal patterns:

DATA:
{payload}

TASK:
1. Seasonal price behavior
2. Root causes (weather, demand, load)
3. Most expensive season
4. Cheapest season
5. Cost optimization strategy
6. Operational recommendations

Keep it concise and executive-level.
"""

        ai_insight = ai_consultant._call_llm(prompt)

    return ai_insight


# =========================================================
# gas_behavior
# =========================================================

def gas_behavior_plot(df):

   
    df = df.copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=["Gas_Price_CAD_GJ"])
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.day_name()
    df["month_name"] = df.index.month_name()

    # ====================================================
    #  Hourly Gas Price by Day of Week
    # ====================================================
    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=df,
        x="Hour",
        y="Gas_Price_CAD_GJ",
        hue="DayOfWeek",
        palette="viridis",
        errorbar=None)

    plt.title("Gas Price by Hour and Day of Week")
    plt.xlabel("Hour of Day")
    plt.ylabel("Gas Price (CAD/GJ)")
    plt.legend(title="Day of Week", bbox_to_anchor=(1.05, 1))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ====================================================
    #  Monthly Trend by Year
    # ====================================================
    monthly_sales = (df.groupby(['Year', 'Month'])['Gas_Price_CAD_GJ'].mean().reset_index())

    plt.figure(figsize=(12, 6))
    years = monthly_sales['Year'].unique()
    colors = itertools.cycle([ 'red', 'green', 'blue', 'orange', 'purple', 'brown'])

    for year in years:
        year_data = monthly_sales[monthly_sales['Year'] == year]

        plt.plot(
            year_data['Month'],
            year_data['Gas_Price_CAD_GJ'],
            marker='o',
            color=next(colors),
            label=str(year) )

    plt.xlabel('Month')
    plt.ylabel('Average Gas Price (CAD/GJ)')
    plt.title('Monthly Gas Price Trend')

    month_names = [
        'Jan', 'Feb', 'Mar', 'Apr',
        'May', 'Jun', 'Jul', 'Aug',
        'Sep', 'Oct', 'Nov', 'Dec']

    plt.xticks(range(1, 13), month_names)
    plt.legend(title="Year")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ====================================================
    #  Average Monthly Gas Price
    # ====================================================
    month_order = [
        'January', 'February', 'March',
        'April', 'May', 'June',
        'July', 'August', 'September',
        'October', 'November', 'December']

    monthly_avg = (df.groupby('month_name')['Gas_Price_CAD_GJ'].mean().reindex(month_order))

    bright_palette = sns.color_palette("bright", 12)
    plt.figure(figsize=(14, 7))

    bars = plt.bar(
        monthly_avg.index,
        monthly_avg.values,
        color=bright_palette)

    plt.title(
        'Average Monthly Gas Price (CAD/GJ)',
        fontsize=16,
        fontweight='bold')

    plt.xlabel('Month')
    plt.ylabel('Gas Price (CAD/GJ)')
    plt.xticks(rotation=45)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontweight='bold')

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df

# =========================================================
# BEHAVIORAL & MARKET ANALYSIS
# =========================================================

def behavioral(df, ai_consultant=None):
    df = df.copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df = df.dropna(subset=[
        "AIL_Total_Demand_MW",
        "Electricity_Price_CAD_MWh",
        "Gas_Price_CAD_GJ"])

    # =========================
    # HOURLY DEMAND
    # =========================
    hour_pattern = df.groupby(df.index.hour)["AIL_Total_Demand_MW"].mean()

    plt.figure(figsize=(10, 5))
    hour_pattern.plot(kind="line", marker="o")
    plt.title("Hourly Demand Pattern")
    plt.xlabel("Hour")
    plt.ylabel("Demand (MW)")
    plt.grid(True)
    plt.show()

    # =========================
    # WEEKDAY PATTERN
    # =========================
    df["Weekday_Name"] = df.index.day_name()

    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"]

    weekday_pattern = (df.groupby("Weekday_Name")["AIL_Total_Demand_MW"].mean().reindex(weekday_order))

    plt.figure(figsize=(10, 5))
    weekday_pattern.plot(kind="bar", color="skyblue")
    plt.title("Weekday Consumption Pattern")
    plt.xlabel("Day")
    plt.ylabel("Demand (MW)")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    # =========================
    # CHEAP HOURS
    # =========================
    cheap_hours = df.groupby(df.index.hour)["Electricity_Price_CAD_MWh"].min()

    plt.figure(figsize=(10, 5))
    cheap_hours.plot(kind="bar", color="green")
    plt.title("Electricity Price by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Price (CAD/MWh)")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    # =========================
    # COST HOURS
    # =========================
    df["Total_Cost_Hour"] = (df["AIL_Total_Demand_MW"] * df["Electricity_Price_CAD_MWh"])

    peak_hours = df.groupby(df.index.hour)["Total_Cost_Hour"].mean()

    plt.figure(figsize=(10, 5))
    peak_hours.plot(kind="bar", color="red")
    plt.title("Average Cost by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Cost Impact")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    # =========================
    # PRICE TREND
    # =========================
    price_trend = df.groupby(df.index.year)["Electricity_Price_CAD_MWh"].mean()

    plt.figure(figsize=(8, 4))
    price_trend.plot(marker="o")
    plt.title("Electricity Price Trend")
    plt.xlabel("Year")
    plt.ylabel("Price (CAD/MWh)")
    plt.grid(True)
    plt.show()

    # =========================
    # GAS TREND
    # =========================
    gas_trend = df.groupby(df.index.year)["Gas_Price_CAD_GJ"].mean()

    plt.figure(figsize=(8, 4))
    gas_trend.plot(marker="o", color="orange")
    plt.title("Gas Price Trend")
    plt.xlabel("Year")
    plt.ylabel("Gas Price (CAD/GJ)")
    plt.grid(True)
    plt.show()

    # =========================
    # GAS vs ELECTRICITY
    # =========================
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=df["Gas_Price_CAD_GJ"],
        y=df["Electricity_Price_CAD_MWh"])
    plt.title("Gas vs Electricity Price Relationship")
    plt.xlabel("Gas Price (CAD/GJ)")
    plt.ylabel("Electricity Price (CAD/MWh)")
    plt.grid(True)
    plt.show()

    # =========================
    # DEMAND vs PRICE
    # =========================
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=df["Electricity_Price_CAD_MWh"],
        y=df["AIL_Total_Demand_MW"])
    plt.title("Demand vs Price")
    plt.xlabel("Price (CAD/MWh)")
    plt.ylabel("Demand (MW)")
    plt.grid(True)
    plt.show()

    # =========================================================
    # LLM INSIGHT LAYER (OLLAMA / LOCAL MODEL)
    # =========================================================
    ai_insight = None

    if ai_consultant is not None:


        payload = {
            "hourly_demand_pattern": hour_pattern.to_dict(),
            "weekday_pattern": weekday_pattern.to_dict(),
            "cheap_hours": cheap_hours.nsmallest(5).to_dict(),
            "expensive_hours": peak_hours.nlargest(5).to_dict(),
            "price_trend": price_trend.to_dict(),
            "gas_trend": gas_trend.to_dict(),
            "price_demand_correlation": df["Electricity_Price_CAD_MWh"].corr(df["AIL_Total_Demand_MW"])}

        prompt = f"""
You are a senior energy market analyst (Calgary electricity system).

Analyze this behavioral energy dataset:

DATA:
{payload}

TASK:
1. Peak demand behavior
2. Cheapest vs expensive operating hours
3. Demand-price relationship
4. Gas vs electricity dependency
5. Inefficiencies in consumption
6. Cost optimization strategy for industry

Keep it concise and executive-level.
"""

        ai_insight = ai_consultant.chat.completions.create(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

    return df, ai_insight




# =========================================================
# STATISTICAL analysis
# =========================================================
def run_complete_statistical_tests(df, ai_consultant=None):
    results = {}
    
    # Prepare data
    df['hour'] = df.index.hour
    df['weekday'] = df.index.dayofweek
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # ANOVA
    groups = [
        df[df["weekday"] == d]["Electricity_Price_CAD_MWh"].values
        for d in range(7)
        if len(df[df["weekday"] == d]) > 0]
    
    f_stat, p_val = f_oneway(*groups) if len(groups) > 1 else (0, 1)
    
    results["anova"] = {
        "f_stat": f_stat,
        "p_value": p_val,
        "significant": p_val < 0.05}
    



    # Chi-square (Weekend vs High Price)
    df2 = df.copy()
    df2["high_price"] = (df2["Electricity_Price_CAD_MWh"] > df2["Electricity_Price_CAD_MWh"].quantile(0.75)).astype(int)
    
    table = pd.crosstab(df2["is_weekend"], df2["high_price"])
    chi2, p_chi, _, _ = chi2_contingency(table)
    
    results["chi_square_weekend"] = {
        "chi2": chi2,
        "p_value": p_chi,
        "significant": p_chi < 0.05}
    


    # LINEAR REGRESSION
    reg_df = df[["Electricity_Price_CAD_MWh", "Temperature_C", "AIL_Total_Demand_MW"]].dropna()
    
    X = reg_df[["Temperature_C", "AIL_Total_Demand_MW"]].values
    y = reg_df["Electricity_Price_CAD_MWh"].values
    
    lin = LinearRegression()
    lin.fit(X, y)
    
    y_pred = lin.predict(X)
    r2 = r2_score(y, y_pred)
    
    results["linear_regression"] = {
        "r2": r2,
        "temp_coef": lin.coef_[0],
        "load_coef": lin.coef_[1],
        "intercept": lin.intercept_}
    
    # strongest driver
    results["linear_regression"]["strongest_driver"] = (
        "temperature"
        if abs(lin.coef_[0]) > abs(lin.coef_[1])
        else "system_load")
    


    # LOGISTIC REGRESSION
    threshold = df["Electricity_Price_CAD_MWh"].quantile(0.75)
    y_bin = (df["Electricity_Price_CAD_MWh"] > threshold).astype(int)
    
    features = ["Temperature_C", "AIL_Total_Demand_MW", "hour", "is_weekend"]
    log_df = df[features].dropna()
    y_bin = y_bin.loc[log_df.index]
    
    log = LogisticRegression(max_iter=1000)
    log.fit(log_df.values, y_bin.values)
    
    results["logistic_regression"] = {
        "accuracy": log.score(log_df.values, y_bin.values),
        "threshold": threshold,
        "coefficients": dict(zip(features, log.coef_[0]))}


    
# ============================================================
# SECTION: COMPLETE MARKET INDICATORS WITH ANOMALY & RISK ANALYSIS
# ============================================================

def calculate_market_indicators(df, price_col='Electricity_Price_CAD_MWh', ai_consultant=None):
    
    df['hour'] = df.index.hour
    df['weekday'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # ============================================================
    # Anomaly Detection
    # ============================================================
    prices = df[price_col]
    
  
    z_scores = np.abs((prices - prices.mean()) / prices.std())
    anomaly_mask_z = z_scores > 3
    

    q1, q3 = prices.quantile([0.25, 0.75])
    iqr = q3 - q1
    anomaly_mask_iqr = (prices < q1 - 1.5 * iqr) | (prices > q3 + 1.5 * iqr)
    

    anomaly_mask = anomaly_mask_z | anomaly_mask_iqr
    anomalies = prices[anomaly_mask]
    
  
    anomaly_stats = {
        "total_anomalies": int(anomaly_mask.sum()),
        "anomaly_ratio_pct": round(anomaly_mask.mean() * 100, 2),
        "mean_anomaly_price": round(anomalies.mean(), 2) if len(anomalies) > 0 else 0,
        "max_anomaly_price": round(anomalies.max(), 2) if len(anomalies) > 0 else 0,
        "min_anomaly_price": round(anomalies.min(), 2) if len(anomalies) > 0 else 0}
    

    if len(anomalies) > 0:
        df_anom = df.copy()
        df_anom['is_anomaly'] = anomaly_mask
        anomaly_stats["common_anomaly_hours"] = (
            df_anom[df_anom['is_anomaly']]['hour']
            .value_counts().head(3).index.tolist())
    else:
        anomaly_stats["common_anomaly_hours"] = []
    
    # ============================================================
    # 2. Spark Spread Analysis
    # ============================================================
    if 'Gas_Price_CAD_GJ' in df.columns:
        heat_rate = 7.0  # نرخ حرارتی فرضی (MMBtu/MWh)
        df['Spark_Spread'] = df[price_col] - (df['Gas_Price_CAD_GJ'] * heat_rate)
        
        spark_stats = {
            "avg_spark_spread": round(df['Spark_Spread'].mean(), 2),
            "max_spark_spread": round(df['Spark_Spread'].max(), 2),
            "min_spark_spread": round(df['Spark_Spread'].min(), 2),
            "negative_spread_pct": round((df['Spark_Spread'] < 0).mean() * 100, 2),
            "positive_spread_pct": round((df['Spark_Spread'] > 0).mean() * 100, 2)}
        
     
        profitable_hours = (
            df[df['Spark_Spread'] > 0]
            .groupby('hour')
            .size()
            .nlargest(3)
            .index.tolist())
        spark_stats["profitable_hours"] = profitable_hours
        
  
        spark_by_hour = df.groupby('hour')['Spark_Spread'].mean()
        spark_stats["best_generation_hours"] = spark_by_hour.nlargest(3).index.tolist()
        spark_stats["worst_generation_hours"] = spark_by_hour.nsmallest(3).index.tolist()
        
    else:
        spark_stats = {
            "avg_spark_spread": 0,
            "max_spark_spread": 0,
            "min_spark_spread": 0,
            "negative_spread_pct": 0,
            "positive_spread_pct": 0,
            "profitable_hours": [],
            "best_generation_hours": [],
            "worst_generation_hours": []}
    
    # ============================================================
    # 3.Volatility & Risk Analysis
    # ============================================================
    # ضریب تغییرات (CV)
    cv = (df[price_col].std() / df[price_col].mean()) * 100
    
    # سطح ریسک
    if cv < 15:
        risk_level = "LOW"
        risk_color = "🟢"
    elif cv < 30:
        risk_level = "MODERATE"
        risk_color = "🟡"
    else:
        risk_level = "HIGH"
        risk_color = "🔴"
    
    # نوسان درون‌روز و بین‌روز
    intraday_volatility = df.groupby('hour')[price_col].std().mean()
    daily_avg = df.groupby(df.index.date)[price_col].mean()
    interday_volatility = daily_avg.std()
    volatility_ratio = intraday_volatility / (interday_volatility + 1e-6)
    
    # نوسان‌پذیری تاریخی
    rolling_volatility = df[price_col].rolling(window=24).std()
    current_volatility = rolling_volatility.iloc[-1] if len(rolling_volatility) > 0 else 0
    
    volatility_stats = {
        "cv_pct": round(cv, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "intraday_volatility": round(intraday_volatility, 2),
        "interday_volatility": round(interday_volatility, 2),
        "volatility_ratio": round(volatility_ratio, 2),
        "current_volatility": round(current_volatility, 2),
        "volatility_percentile": round((cv / 100) * 100, 1)}
    
    # ============================================================
    # 4.Hourly Pattern
    # ============================================================
    hourly_avg = df.groupby('hour')[price_col].mean()
    peak_hours = hourly_avg.nlargest(3).index.tolist()
    off_peak_hours = hourly_avg.nsmallest(3).index.tolist()
    
    peak_price = hourly_avg.loc[peak_hours].mean()
    off_peak_price = hourly_avg.loc[off_peak_hours].mean()
    mean_price = df[price_col].mean()
    median_price = df[price_col].median()
    
    peak_index = peak_price / mean_price
    off_peak_discount = (mean_price - off_peak_price) / mean_price * 100
    
    # ============================================================
    # 5(Uniformity Index)
    # ============================================================
    uniformity_index = hourly_avg.std() / hourly_avg.mean()
    cv_hourly = (hourly_avg.std() / hourly_avg.mean()) * 100
    
    # ============================================================
    # 6. Weekly Pattern
    # ============================================================
    weekday_price = df[~df['is_weekend'].astype(bool)][price_col].mean()
    weekend_price = df[df['is_weekend'].astype(bool)][price_col].mean()
    weekend_discount = ((weekday_price - weekend_price) / weekday_price) * 100 if weekday_price > 0 else 0
    
    # شاخص تعطیلات
    if 'Holiday' in df.columns:
        if df['Holiday'].dtype in ['int64', 'float64']:
            holiday_price = df[df['Holiday'] == 1][price_col].mean() if 1 in df['Holiday'].values else weekday_price
            normal_price = df[df['Holiday'] == 0][price_col].mean() if 0 in df['Holiday'].values else weekday_price
        else:
            holiday_price = df[df['Holiday'] == 'Yes'][price_col].mean() if 'Yes' in df['Holiday'].values else weekday_price
            normal_price = df[df['Holiday'] == 'No'][price_col].mean() if 'No' in df['Holiday'].values else weekday_price
        holiday_discount = ((normal_price - holiday_price) / normal_price) * 100 if normal_price > 0 else 0
    else:
        holiday_discount = 0
    
    # ============================================================
    # 7. Seasonal Pattern
    # ============================================================
    monthly_avg = df.groupby('month')[price_col].mean()
    peak_month = monthly_avg.idxmax()
    off_peak_month = monthly_avg.idxmin()
    seasonal_amplitude = (monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean() * 100
    seasonal_strength = monthly_avg.std() / monthly_avg.mean()
    
    # ============================================================
    # 8. Grid Stress
    # ============================================================
    if 'Grid_Stress_Level' in df.columns:
        if df['Grid_Stress_Level'].dtype == 'object':
            stress_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
            df['stress_numeric'] = df['Grid_Stress_Level'].map(stress_mapping)
        else:
            df['stress_numeric'] = df['Grid_Stress_Level']
        stress_by_hour = df.groupby('hour')['stress_numeric'].mean()
        peak_stress_hours = stress_by_hour.nlargest(3).index.tolist()
        avg_stress_level = df['stress_numeric'].mean()
        high_stress_pct = (df['stress_numeric'] >= 2).mean() * 100 if 'stress_numeric' in df else 0
    else:
        peak_stress_hours = []
        avg_stress_level = 0
        high_stress_pct = 0
    
    # ============================================================
    # 9.Demand Index
    # ============================================================
    if 'AIL_Total_Demand_MW' in df.columns:
        demand_by_hour = df.groupby('hour')['AIL_Total_Demand_MW'].mean()
        peak_demand_hours = demand_by_hour.nlargest(3).index.tolist()
        demand_price_correlation = df['AIL_Total_Demand_MW'].corr(df[price_col])
        avg_demand = df['AIL_Total_Demand_MW'].mean()
        peak_demand = df['AIL_Total_Demand_MW'].max()
        demand_range = peak_demand - avg_demand
    else:
        peak_demand_hours = []
        demand_price_correlation = 0
        avg_demand = 0
        peak_demand = 0
        demand_range = 0
    
    # ============================================================
    # 10.Opportunity Index
    # ============================================================
    price_vol_by_hour = df.groupby('hour')[price_col].agg(['mean', 'std'])
    price_vol_by_hour['score'] = price_vol_by_hour['mean'] / (price_vol_by_hour['std'] + 1e-6)
    best_buy_hour = price_vol_by_hour['score'].idxmin()
    best_sell_hour = price_vol_by_hour['score'].idxmax()
    

    potential_savings = ((mean_price - off_peak_price) / mean_price) * 100
    
  
    results = {
  
        'anomaly': anomaly_stats,
        'spark_spread': spark_stats,
        'volatility': volatility_stats,

        'peak_hours': peak_hours,
        'off_peak_hours': off_peak_hours,
        'peak_price': peak_price,
        'off_peak_price': off_peak_price,
        'peak_index': peak_index,
        'off_peak_discount_pct': off_peak_discount,
        'uniformity_index': uniformity_index,
        'cv_hourly_pct': cv_hourly,

        'weekday_price': weekday_price,
        'weekend_price': weekend_price,
        'weekend_discount_pct': weekend_discount,
        'holiday_discount_pct': holiday_discount,
        'peak_month': peak_month,
        'off_peak_month': off_peak_month,
        'seasonal_amplitude_pct': seasonal_amplitude,
        'seasonal_strength': seasonal_strength,
        'monthly_pattern': monthly_avg.to_dict(),
        'peak_stress_hours': peak_stress_hours,
        'avg_stress_level': avg_stress_level,
        'high_stress_pct': high_stress_pct,
        'peak_demand_hours': peak_demand_hours,
        'demand_price_correlation': demand_price_correlation,
        'avg_demand': avg_demand,
        'peak_demand': peak_demand,
        'demand_range': demand_range,
        'best_buy_hour': best_buy_hour,
        'best_sell_hour': best_sell_hour,
        'potential_savings_pct': potential_savings,
        'mean_price': mean_price,
        'median_price': median_price,
        'price_std': df[price_col].std(),
        'price_range': df[price_col].max() - df[price_col].min(),
        'data_points': len(df)}
    
    

# ============================================================================
# Financial Analysis
# ============================================================================

def calculate_financial_metrics(df, monthly_consumption, investment_cost, ai_consultant=None):
    """
    Financial analysis: ROI, NPV, Profitability, Investment decision + LLM insights
    """
    
    df = df.copy()
    
    # Price structure
    hourly_price = df.groupby(df.index.hour)['Electricity_Price_CAD_MWh'].mean()
    
    off_peak_price = hourly_price.nsmallest(3).mean()
    mean_price = df['Electricity_Price_CAD_MWh'].mean()
    
    # Energy cost model
    monthly_consumption_mwh = monthly_consumption / 1000
    
    total_revenue = monthly_consumption_mwh * mean_price
    total_cost = monthly_consumption_mwh * mean_price
    
    optimized_cost = monthly_consumption_mwh * (
        0.3 * off_peak_price + 0.7 * mean_price)
    
    # Profitability
    profit = total_revenue - optimized_cost
    profit_margin = (profit / total_revenue) * 100 if total_revenue > 0 else 0
    
    monthly_savings = total_cost - optimized_cost
    yearly_savings = monthly_savings * 12
    
    # ROI / Payback
    roi = (yearly_savings / investment_cost) * 100 if investment_cost else 0
    payback_months = investment_cost / monthly_savings if monthly_savings > 0 else np.inf
    
    # NPV (5 years)
    discount_rate = 0.10
    npv = sum(
        yearly_savings / (1 + discount_rate) ** t
        for t in range(1, 6)) - investment_cost
    
    # Decision logic
    if roi > 30:
        decision = "IMMEDIATE APPROVAL"
    elif roi > 15:
        decision = "APPROVE"
    else:
        decision = "REVIEW"
    
    # Derived Metrics
    df["Total_Cost"] = df["AIL_Total_Demand_MW"] * df["Electricity_Price_CAD_MWh"]
    df["Cost_Per_Unit"] = df["Total_Cost"] / df["AIL_Total_Demand_MW"]
    
    # Visual Analysis
    plt.figure()
    df.groupby(df.index.year)["Total_Cost"].sum().plot(kind="bar")
    plt.title("Annual Energy Cost")
    plt.show()
    
    plt.figure()
    df.groupby(df.index.hour)["Total_Cost"].mean().plot(kind="bar")
    plt.title("Peak Cost Periods")
    plt.show()
    
    plt.figure()
    plt.hist(df["Cost_Per_Unit"], bins=30)
    plt.title("Cost Distribution")
    plt.show()
    
    plt.figure()
    plt.scatter(df["AIL_Total_Demand_MW"], df["Cost_Per_Unit"])
    plt.title("Cost vs Consumption")
    plt.show()
    
    # Results
    results = {
        "total_revenue": total_revenue,
        "total_cost": total_cost,
        "optimized_cost": optimized_cost,
        "profit": profit,
        "profit_margin_pct": profit_margin,
        "monthly_savings": monthly_savings,
        "yearly_savings": yearly_savings,
        "roi_pct": roi,
        "payback_months": payback_months,
        "npv_5y": npv,
        "decision": decision }
    
    # LLM Insight Layer
    ai_insights = None
    
    if ai_consultant is not None:
        prompt = f"""
You are a senior energy market analyst.

Analyze the following financial results:

Revenue: {total_revenue:,.2f}
Cost: {total_cost:,.2f}
Optimized Cost: {optimized_cost:,.2f}
Profit: {profit:,.2f}
Profit Margin: {profit_margin:.2f}%
ROI: {roi:.2f}%
Payback (months): {payback_months:.1f}
NPV (5 years): {npv:,.2f}

Decision: {decision}

Return:
1. Key insight
2. Root cause of performance
3. Financial risk
4. Optimization strategy
"""
        
        ai_insights = ai_consultant._call_llm(prompt)
    
    return results, ai_insights, decision


# =========================================================
# FORECASTING
# =========================================================

 def prepare_data(df, target="Electricity_Price_CAD_MWh", horizon=1, use_last_n=600):
    """Prepare data for ML modeling - uses last N rows for speed"""
    df = df.copy()
    df = df.sort_index()
    
    # OPTIMIZATION: Use only last N rows for faster training
    if len(df) > use_last_n:
        df = df.tail(use_last_n)
        st.info(f" Using last {use_last_n} data points for faster training (total available: {len(df)})")
    
    df = df.ffill().bfill()
    return

target = "Electricity_Price_CAD_MWh"

# Stationarity
def check_stationarity(series):
    series = df[target].dropna()
    result = adfuller(series.dropna())
    return {
        "ADF": result[0],
        "p_value": result[1],
        "stationary": result[1] < 0.05}



lookback = 90
df = df.tail(5000)


# ARIMA


def run_arima_pipeline(series,order=(2,1,1),split_date=None,freq="h"):

   
    series = series.copy()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index().ffill().bfill()

    y = series

    # SPLIT
    if split_date is not None:
        train = y[y.index < split_date]
        test  = y[y.index >= split_date]
    else:
        train_size = int(len(y) * 0.8)
        train = y.iloc[:train_size]
        test  = y.iloc[train_size:]

    print("Train:", train.shape, "Test:", test.shape)

    # FIT MODEL
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    print(model_fit.summary())

    results = {}

    # TEST FORECAST
    if test is not None:

        forecast_test = model_fit.forecast(steps=len(test))
        forecast_test.index = test.index

        mae = mean_absolute_error(test, forecast_test)
        rmse = np.sqrt(mean_squared_error(test, forecast_test))
        mape = np.mean(np.abs((test - forecast_test) / test)) * 100

        print("\n===== METRICS =====")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAPE : {mape:.2f}%")

        plt.figure(figsize=(18,6))
        plt.plot(train.index, train, label="Train")
        plt.plot(test.index, test, label="Test")
        plt.plot(forecast_test.index, forecast_test, label="Forecast")
        plt.axvline(test.index[0], color='red', linestyle='--')
        plt.legend()
        plt.grid()
        plt.show()

        results["test_forecast"] = forecast_test

    # FUTURE FORECAST
    def forecast_future(steps, label):

        future = model_fit.forecast(steps=steps)

        future_index = pd.date_range(
            start=y.index[-1],
            periods=steps + 1,
            freq=freq)[1:]

        future.index = future_index

        plt.figure(figsize=(18,6))
        plt.plot(y.tail(24*14), label="History")
        plt.plot(future, label=f"{label} Forecast")
        plt.title(f"ARIMA - {label} Forecast")
        plt.legend()
        plt.grid()
        plt.show()

        return future

    # HORIZONS
    results["1W"] = forecast_future(24*7, "1 Week")
    results["1M"] = forecast_future(24*30, "1 Month")
    results["3M"] = forecast_future(24*90, "3 Months")

    return model_fit, results




def run_sarima_pipeline(series, order=(2,1,1), seasonal_order=(1,1,1,24), split_date=None, freq="h"):

    series = series.copy()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index().ffill().bfill()

    y = series

    if split_date is not None:
        train = y[y.index < split_date]
        test = y[y.index >= split_date]
    else:
        train_size = int(len(y) * 0.8)
        train = y.iloc[:train_size]
        test = y.iloc[train_size:]

    print("Train:", train.shape)
    print("Test :", test.shape)

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False)

    model_fit = model.fit(disp=False)
    print(model_fit.summary())
    model_fit.plot_diagnostics(figsize=(18,10))
    plt.show()



    results = {}

    forecast_test = model_fit.forecast(steps=len(test))

    forecast_test = pd.Series(
        forecast_test.values,
        index=test.index)



    mae = mean_absolute_error(test, forecast_test)
    rmse = np.sqrt(  mean_squared_error(test, forecast_test))
    mape = np.mean(np.abs((test - forecast_test) / test)) * 100

    print("\n===== METRICS =====")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")


    plt.figure(figsize=(18,6))
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Test")
    plt.plot(forecast_test.index, forecast_test, label="Forecast")

    plt.fill_between(conf_int.index,conf_int.iloc[:,0],   # Lower Bound, 
                     conf_int.iloc[:,1],   # Upper Bound
                     alpha=0.25,label="95% Confidence Interval")

    plt.axvline(test.index[0],color='red',linestyle='--')
    plt.title("SARIMA Test Forecast")
    plt.legend()
    plt.grid()
    plt.show()
    results["test_forecast"] = forecast_test



    def forecast_future(steps, label):

        future = model_fit.forecast(steps=steps)
        future_index = pd.date_range(
            start=y.index[-1],
            periods=steps + 1,
            freq=freq)[1:]

        future = pd.Series( future.values,index=future_index)

        plt.figure(figsize=(18,6))
        plt.plot(y.tail(24*14),label="History")
        plt.plot( future, label=f"{label} Forecast" )

        plt.title( f"SARIMA - {label} Forecast")
        plt.legend()
        plt.grid()
        plt.show()
        return future


    results["1W"] = forecast_future(24*7, "1 Week")
    results["1M"] = forecast_future(24*30, "1 Month")
    results["3M"] = forecast_future(24*90, "3 Months")

    return model_fit, results



def plot_acf_pacf(series, lags=50, title_prefix="Series"):
    """
    Plot ACF and PACF for any time series
    """

    # ensure clean series
    series = series.dropna()

    plt.figure(figsize=(14,6))

    # ACF
    plt.subplot(1,2,1)
    plot_acf(series, lags=lags, ax=plt.gca())
    plt.title(f"ACF - {title_prefix}")

    # PACF
    plt.subplot(1,2,2)
    plot_pacf(series, lags=lags, ax=plt.gca())
    plt.title(f"PACF - {title_prefix}")

    plt.tight_layout()
    plt.show()



plot_acf_pacf(train, lags=50, title_prefix="Electricity Price")


# =========================================================
# Prediction Model + LLM Insight
# =========================================================

TARGET="Electricity_Price_CAD_MWh"
HORIZON=168

# =========================
# PREPARE DATA
# =========================

def prepare_data(df):

    df=df.copy()
    df=df.sort_index()
    df=df.ffill().bfill()

    # target shifted 168h ahead
    y=df[TARGET].shift(-HORIZON)

    # features
    X=df.select_dtypes(include=np.number).copy()

    leakage_cols=[
        "Energy_Demand_MW",
        "Energy_Price_CAD",
        "Electricity_Price_CAD_MWh",
        "Gas_Price_CAD_GJ"]

    X=X.drop(columns=leakage_cols,errors="ignore")

    if TARGET in X.columns:
        X=X.drop(columns=[TARGET])

    # align safely
    data=X.copy()
    data["target"]=y

    data=data.dropna(subset=["target"])

    X=data.drop(columns=["target"])
    y=data["target"]

    # fix inf/nan
    X=X.replace([np.inf,-np.inf],np.nan)
    X=X.fillna(0)

    split=int(len(X)*0.8)

    X_train=X.iloc[:split]
    X_test=X.iloc[split:]

    y_train=y.iloc[:split]
    y_test=y.iloc[split:]

    return X_train,X_test,y_train,y_test,X,data

# =========================
# MODEL
# =========================

def train_model(X_train,y_train,X_test,y_test):

    model=xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42)

    model.fit(X_train,y_train)

    preds=model.predict(X_test)

    mae=mean_absolute_error(y_test,preds)
    rmse=np.sqrt(mean_squared_error(y_test,preds))
    r2=r2_score(y_test,preds)

    print("\n===== TEST RESULTS =====")
    print("MAE :",mae)
    print("RMSE:",rmse)
    print("R2  :",r2)

    return model,preds

# =========================
# FUTURE FORECAST
# =========================

def forecast_future(model,X,df):

    future_X=X.tail(HORIZON)

    future_pred=model.predict(future_X)

    future_dates=pd.date_range(
        start=df.index[-1],
        periods=HORIZON,
        freq="h" )

    future_df=pd.DataFrame(
        {"Forecast":future_pred},
        index=future_dates )

    return future_df

# =========================
# OLLAMA EXPLANATION
# =========================

def explain_with_ollama(future_df):

    first=future_df["Forecast"].iloc[0]
    last=future_df["Forecast"].iloc[-1]

    trend="increasing" if last>first else "decreasing"

    prompt=f"""
You are an energy market analyst.

We forecast electricity prices for next 168 hours.

Start price: {first:.2f}
End price: {last:.2f}
Trend: {trend}

Forecast summary stats:
{future_df["Forecast"].describe().to_string()}

Explain:
1. Market interpretation
2. Risk (spikes/volatility)
3. Possible drivers (demand, gas, weather)
4. Trading insight

Be concise and professional.
"""

    response=requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":"llama3",
            "prompt":prompt,
            "stream":False})

    return response.json()["response"]

# =========================
# PLOT
# =========================

def plot_results(y_test,preds,future_df):

    plt.figure(figsize=(16,6))
    plt.plot(y_test.values,label="Actual")
    plt.plot(preds,label="Predicted",linestyle="--")
    plt.title("Test Performance")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(16,6))
    plt.plot(future_df.index,future_df["Forecast"])
    plt.title("Next 7 Days Forecast")
    plt.grid()
    plt.show()

# =========================
# RUN PIPELINE
# =========================

X_train,X_test,y_train,y_test,X,data=prepare_data(df1)
model,preds=train_model(X_train,y_train,X_test,y_test)
future_df=forecast_future(model,X,data)
plot_results(y_test,preds,future_df)

# =========================
# OLLAMA OUTPUT
# =========================

explanation=explain_with_ollama(future_df)

print("\n===== OLLAMA EXPLANATION =====\n")
print(explanation)




def compare_models_with_forecast(X_train,X_test,y_train,y_test,X,df,horizon=168):

    models={
        "RandomForest":RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42),

        "XGBoost":xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42 )}

    predictions={}
    results=[]
    trained_models={}

    for name,model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train,y_train)
        preds=model.predict(X_test)
        predictions[name]=preds
        trained_models[name]=model


        mae=mean_absolute_error(y_test,preds)
        rmse=np.sqrt( mean_squared_error(y_test,preds))
        r2=r2_score(y_test,preds)

        results[name].append({
            "Model":name,
            "MAE":round(mae,3),
            "RMSE":round(rmse,3),
            "R2":round(r2,3)})

    results_df=pd.DataFrame(results)
    results_df=results_df.sort_values( by="RMSE")


    print("\n===== MODEL COMPARISON =====\n")
    print(results_df)

    plt.figure(figsize=(16,6))
    plt.plot(
        y_test.values,
        label="Actual",
        linewidth=3)

    for name,preds in predictions.items():

        plt.plot(
            preds,
            label=name,
            alpha=0.8)

    plt.title("Electricity Price Forecast Comparison")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,5))
    plt.bar(
        results_df["Model"],
        results_df["RMSE"])

    plt.title("RMSE Comparison")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.show()

    best_model=results_df.iloc[0]
    best_model=trained_models[best_model_name]
    print("\n===== BEST MODEL =====\n")
    print(best_model_name)

    future_X=X.tail(horizon)

    future_preds=best_model.predict(future_X)

    future_dates=pd.date_range(
        start=df.index[-1],
        periods=horizon,
        freq="h")

    future_df=pd.DataFrame({ "Forecast":future_preds},index=future_dates)

    print("\n===== NEXT 7 DAYS FORECAST =====\n")
    print(future_df.head())
    plt.figure(figsize=(16,6))

    plt.plot(
        future_df.index,
        future_df["Forecast"],
        linewidth=2,
        label=best_model_name)

    plt.title("Next 7 Days Electricity Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Forecast Price")
    plt.legend()
    plt.grid(True)
    plt.show()




    prompt=f"""
    You are a senior machine learning expert.

    Interpret the following electricity price forecasting results.

    Explain:
    1. Best model
    2. Why it performed best
    3. Overfitting or underfitting
    4. Meaning of RMSE,MAE,R2
    5. Recommendations
    6. Best model for deployment
    7.Forecast reliability
    8.Suggestions for improvement

    Results:
    {results_df.to_string(index=False)}
    """

    response=ollama.chat(

        model="llama3",
        messages=[
            { "role":"user", "content":prompt}])

    print("\n===== OLLAMA INTERPRETATION =====\n")
    print(response["message"]["content"])
    return results_df,predictions

    results_df,predictions,future_df=compare_models_with_forecast(X_train,X_test,y_train,y_test,
    X,df,horizon=168)





    


