# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ollama
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Energy ML Forecasting Platform",
    layout="wide")

# =========================================================
# HEADER
# =========================================================

st.title(" AI Energy Machine Learning Platform")
st.subheader("Random Forest | Gradient Boosting | XGBoost | Model Comparison | Feature Importance")

st.info("""
**This platform performs machine learning analysis for energy market prediction and decision support.**

** Random Forest** - Energy demand and price prediction, Non-linear pattern learning, Robust ensemble modeling  
** Gradient Boosting** - High-accuracy predictive modeling, Sequential error reduction, Improved forecasting performance  
**XGBoost** - Advanced gradient boosting algorithm, Fast and optimized prediction, Feature interaction learning  
** Model Comparison** - Compare multiple ML models, Evaluate MAE, RMSE and R², Identify the best-performing algorithm  
** Feature Importance** - Rank the most influential variables, Explain model predictions, Support business decision-making  
** AI Decision Support** - Automatic interpretation of model results, Business insights and recommendations, Forecasting readiness assessment
""")

# =========================================================
# DATA UPLOAD
# =========================================================

uploaded_file = st.file_uploader("Upload Energy Dataset", type=["csv", "xlsx", "xls"])

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    
    df.columns = df.columns.astype(str).str.strip()
    
    # Try to parse datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    elif all(col in df.columns for col in ['Year', 'Month', 'Day', 'Hour']):
        df['Date'] = pd.to_datetime({
            'year': df['Year'],
            'month': df['Month'],
            'day': df['Day'],
            'hour': df['Hour']})
        df = df.set_index('Date')
    
    return df

# =========================================================
# LLM CONNECTION
# =========================================================

def call_llm(prompt):
    try:
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2})
        return response["message"]["content"]
    except Exception as e:
        return f"LLM Error: {str(e)}"

def energy_ml_agent(model_name, metrics, feature_importance, predictions_summary):
    prompt = f"""
You are a Senior Energy Data Scientist. Analyze these ML forecasting results:

**Model:** {model_name}
**Performance Metrics:**
- MAE: {metrics['MAE']:.3f}
- RMSE: {metrics['RMSE']:.3f}
- R²: {metrics['R2']:.3f}

**Top Features:**
{json.dumps(feature_importance[:5], indent=2)}

**Prediction Summary:**
- Mean Predicted: {predictions_summary['mean']:.2f}
- Max Predicted: {predictions_summary['max']:.2f}
- Min Predicted: {predictions_summary['min']:.2f}

Provide:
1. Model Performance Assessment
2. Key Business Insights
3. Feature Impact Analysis
4. Forecasting Reliability
5. Operational Recommendations
"""
    return call_llm(prompt)

# =========================================================
# DATA PREPARATION FUNCTIONS - OPTIMIZED WITH LAST 600 ROWS
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
    
    # Target with optional horizon shift
    y = df[target].shift(-horizon)
    
    # Features - use numeric columns
    X = df.select_dtypes(include=np.number).copy()
    
    # Remove leakage columns
    leakage_cols = [
        "Energy_Demand_MW",'ID',
        "Energy_Price_CAD",
        "Electricity_Price_CAD_MWh",
        "Gas_Price_CAD_GJ"]
    X = X.drop(columns=leakage_cols, errors="ignore")
    
    if target in X.columns:
        X = X.drop(columns=[target])
    
    # Align X and y
    data = X.copy()
    data["target"] = y
    data = data.dropna(subset=["target"])
    
    X = data.drop(columns=["target"])
    y = data["target"]
    
    # Handle NaN/Inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Split
    split = int(len(X) * 0.8)
    
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    
    return X_train, X_test, y_train, y_test, X, df

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# =========================================================
# MODEL TRAINING FUNCTIONS - OPTIMIZED FOR SPEED
# =========================================================

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,  # Reduced from 200 for speed
        max_depth=10,      # Reduced from 12 for speed
        n_jobs=-1,         # Use all CPU cores
        random_state=42)
    
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=150,      # Reduced from 300 for speed
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,             # Use all CPU cores
        random_state=42)
    
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(
        n_estimators=100,      # Reduced from 200 for speed
        learning_rate=0.05,
        max_depth=3,
        random_state=42)
    
    model.fit(X_train, y_train)
    return model

# =========================================================
# FORECAST FUNCTIONS
# =========================================================

def forecast_future(model, X, df, horizon=168):
    """Forecast next HORIZON hours"""
    future_X = X.tail(horizon)
    future_pred = model.predict(future_X)
    
    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h")
    
    future_df = pd.DataFrame(
        {"Forecast": future_pred},
        index=future_dates)
    return future_df

# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    st.success(f" Loaded {len(df)} rows of data")
    
    with st.expander(" Data Preview"):
        st.dataframe(df.head(100))
        st.write(f"**Columns:** {', '.join(df.columns)}")
    
    # =========================================================
    # SIDEBAR - Configuration
    # =========================================================
    
    st.sidebar.header(" Configuration")
    
    # OPTIMIZATION: Data sample size selector
    use_last_n = st.sidebar.slider("Number of recent data points to use",
        min_value=100,
        max_value=min(2000, len(df)),
        value=min(600, len(df)),
        step=100,
        help="Using fewer data points = faster training. Recommended: 500-1000 for hourly data.")
    
    # Target selection
    target_options = [
        "Electricity_Price_CAD_MWh",
        "Energy_Demand_MW",
        "Energy_Price_CAD",
        "Gas_Price_CAD_GJ"]
    
    available_targets = [col for col in target_options if col in df.columns]
    
    if not available_targets:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        target = st.sidebar.selectbox("Select Target Variable", numeric_cols)
    else:
        target = st.sidebar.selectbox("Select Target Variable", available_targets)
    
    # Horizon selection
    horizon = st.sidebar.selectbox(
        "Forecast Horizon",
        [1, 24, 48, 168],
        format_func=lambda x: f"{x} hour{'s' if x > 1 else ''} ({x//24}d {'+' if x%24 else ''})")
    
    # Model selection
    models_to_run = st.sidebar.multiselect(
        "Select Models to Train",
        ["Random Forest", "XGBoost", "Gradient Boosting"],
        default=["Random Forest", "XGBoost", "Gradient Boosting"])
    
    run_button = st.sidebar.button("Run ML Models", type="primary")
    
    # =========================================================
    # MAIN CONTENT
    # =========================================================
    
    if run_button:
        # Prepare data - with optimization
        with st.spinner(f"Preparing data (using last {use_last_n} rows)..."):
            X_train, X_test, y_train, y_test, X, df_used = prepare_data(
                df, target=target, horizon=horizon, use_last_n=use_last_n)
            
            if len(X_train) == 0 or len(X_test) == 0:
                st.error("Not enough data for training. Please check your dataset.")
                st.stop()
            
            st.info(f" Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
            st.info(f" Using {len(X_train) + len(X_test)} total samples for faster training")
        
        # Train models
        models = {}
        predictions = {}
        metrics_results = []
        
        # Create columns for model display
        cols = st.columns(len(models_to_run))
        
        for idx, model_name in enumerate(models_to_run):
            with st.spinner(f"Training {model_name}..."):
                # Train model
                if model_name == "Random Forest":
                    model = train_random_forest(X_train, y_train)
                elif model_name == "XGBoost":
                    model = train_xgboost(X_train, y_train)
                else:
                    model = train_gradient_boosting(X_train, y_train)
                
                models[model_name] = model
                
                # Predict
                preds = model.predict(X_test)
                predictions[model_name] = preds
                
                # Evaluate
                mae, rmse, r2 = evaluate(y_test, preds)
                
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = pd.DataFrame({'Feature': X_train.columns,
                        'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
                    
                    top_features = importance.head(10).to_dict('records')
                else:
                    top_features = []
                
                # Store results
                metrics_results.append({
                    "Model": model_name,
                    "MAE": round(mae, 3),
                    "RMSE": round(rmse, 3),
                    "R2": round(r2, 3),
                    "Top Features": top_features})
                
                # Generate AI analysis for each model
                pred_summary = {
                    'mean': float(preds.mean()),
                    'max': float(preds.max()),
                    'min': float(preds.min())}
                
                with st.spinner(f"Generating AI insights for {model_name}..."):
                    ai_response = energy_ml_agent(
                        model_name, 
                        {'MAE': mae, 'RMSE': rmse, 'R2': r2},
                        top_features,
                        pred_summary)
                
                # Display results in columns
                with cols[idx]:
                    st.subheader(f" {model_name}")
                    
                    # Metrics
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    col_metric1.metric("MAE", f"{mae:.3f}")
                    col_metric2.metric("RMSE", f"{rmse:.3f}")
                    col_metric3.metric("R²", f"{r2:.3f}")
                    
                    # Plot predictions
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(y_test.values[:200], label='Actual', linewidth=2)
                    ax.plot(preds[:200], label='Predicted', linestyle='--', linewidth=2)
                    ax.set_title(f'{model_name} - Actual vs Predicted')
                    ax.set_xlabel('Time Steps')
                    ax.set_ylabel(target)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Feature Importance
                    if top_features:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        top_df = pd.DataFrame(top_features[:10])
                        ax.barh(top_df['Feature'], top_df['Importance'], color='skyblue')
                        ax.set_title(f'{model_name} - Top 10 Feature Importance')
                        ax.set_xlabel('Importance')
                        ax.invert_yaxis()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    
        
        
        # =========================================================
        # AI COMPARISON ANALYSIS
        # =========================================================
        
        st.header(" AI Model Comparison Analysis")
        
        with st.spinner("Generating AI comparison insights..."):
            comparison_prompt = f"""
You are a Senior Energy Data Scientist. Compare these ML models for energy forecasting:

**Results:**
{json.dumps(metrics_results, indent=2, default=str)}

**Best Model:** {best_model_name} (RMSE: {best_model_row['RMSE']:.3f})

**Data Used:** Last {use_last_n} data points for faster training

Provide:
1. Executive Summary of Findings
2. Best Model Recommendation
3. Why {best_model_name} Performed Best
4. Business Impact of Using Best Model
5. Feature Engineering Recommendations
6. Operational Implementation Guidance
7. Forecasting Reliability Score (0-100)
8. Next Steps for Improvement
"""
            comparison_ai = call_llm(comparison_prompt)
            st.markdown(comparison_ai)
        
     