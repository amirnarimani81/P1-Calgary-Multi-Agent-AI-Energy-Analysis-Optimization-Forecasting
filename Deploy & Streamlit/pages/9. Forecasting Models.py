# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ollama
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="Energy Forecasting Platform", layout="wide")

st.title(" Energy Forecasting Platform")
st.subheader("ARIMA | SARIMA | LSTM | RNN | Seasonal Decomposition | ADF Test")

st.info("""
**Time Series Models:** ARIMA, SARIMA, Seasonal Decomposition  
**Deep Learning:** LSTM, RNN  
**Statistical Tests:** ADF Stationarity, ACF/PACF Analysis
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
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            st.error(" Could not convert index to datetime.")
    
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

# =========================================================
# STATIONARITY TEST
# =========================================================

def check_stationarity(series):
    series = series.dropna()
    result = adfuller(series)
    return {
        "ADF_Statistic": result[0],
        "p_value": result[1],
        "stationary": result[1] < 0.05,
        "summary": f"ADF: {-result[0]:.4f}, p-value: {result[1]:.4f} - {'Stationary' if result[1] < 0.05 else 'Non-Stationary'}"}

# =========================================================
# ACF & PACF
# =========================================================

def plot_acf_pacf(series, lags=50):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].grid(True, alpha=0.3)
    plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# =========================================================
# ARIMA
# =========================================================

def run_arima(series, order=(2,1,1)):
    series = series.sort_index().ffill().bfill()
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    model = ARIMA(train, order=order).fit()
    
    forecast = model.forecast(len(test))
    forecast.index = test.index
    
    metrics = {
        "MAE": mean_absolute_error(test, forecast),
        "RMSE": np.sqrt(mean_squared_error(test, forecast)),
        "R2": r2_score(test, forecast)}
    
    return {"model": model, "forecast": forecast, "train": train, "test": test, "metrics": metrics, "summary": model.summary().as_text()}

# =========================================================
# SARIMA
# =========================================================

def run_sarima(series, order=(1,1,1), seasonal_order=(1,1,1,24)):
    series = series.sort_index().ffill().bfill()
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, 
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    
    forecast = model.forecast(len(test))
    forecast.index = test.index
    
    metrics = {
        "MAE": mean_absolute_error(test, forecast),
        "RMSE": np.sqrt(mean_squared_error(test, forecast)),
        "R2": r2_score(test, forecast)}
    
    return {"model": model, "forecast": forecast, "train": train, "test": test, "metrics": metrics, "summary": model.summary().as_text()}

# =========================================================
# SEASONAL DECOMPOSITION
# =========================================================

def run_decomposition(series, period=24):
    series = series.sort_index().ffill().bfill()
    decomp = seasonal_decompose(series, model='additive', period=period)
    return {
        "observed": decomp.observed,
        "trend": decomp.trend,
        "seasonal": decomp.seasonal,
        "residual": decomp.resid,
        "stats": {"seasonal_strength": decomp.seasonal.var() / (decomp.seasonal.var() + decomp.resid.var())}}

# =========================================================
# LSTM
# =========================================================

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

def run_lstm(series, seq_len=24, epochs=50):
    series = series.sort_index().ffill().bfill()
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series.values.reshape(-1, 1))
    
    X, y = create_sequences(data, seq_len)
    if len(X) < 100:
        return {"error": "Not enough data"}
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                       validation_split=0.1, callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                       verbose=0)
    
    pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    metrics = {
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": np.sqrt(mean_squared_error(actual, pred)),
        "R2": r2_score(actual, pred)}
    
    return {"model": model, "scaler": scaler, "history": history,
            "predictions": pred.flatten(), "actual": actual.flatten(), "metrics": metrics}

# =========================================================
# RNN
# =========================================================

def run_rnn(series, seq_len=24, epochs=50):
    series = series.sort_index().ffill().bfill()
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series.values.reshape(-1, 1))
    
    X, y = create_sequences(data, seq_len)
    if len(X) < 100:
        return {"error": "Not enough data"}
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    model = Sequential([
        SimpleRNN(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        SimpleRNN(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                       validation_split=0.1, callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                       verbose=0)
    
    pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    metrics = {
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": np.sqrt(mean_squared_error(actual, pred)),
        "R2": r2_score(actual, pred)}
    
    return {"model": model, "scaler": scaler, "history": history,
            "predictions": pred.flatten(), "actual": actual.flatten(), "metrics": metrics}

# =========================================================
# AI INSIGHTS
# =========================================================

def generate_ai_insights(results, target):
    prompt = f"""
ENERGY FORECAST ANALYSIS

Target: {target}

Stationarity: {results['stationarity']['summary']}

Model Performance:
{json.dumps(results['comparison'], indent=2)}

Best Model: {results['best_model']}

Provide:
1. Model Performance Assessment
2. Key Business Insights
3. Forecasting Reliability
4. Operational Recommendations
"""
    return call_llm(prompt)

# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success(f" Loaded {len(df)} rows")
    
    with st.expander(" Data Preview"):
        st.dataframe(df.head(50))
    
    # =========================================================
    # SIDEBAR
    # =========================================================
    
    st.sidebar.header(" Configuration")
    
    use_last_n = st.sidebar.slider("Data points to use", 100, min(2000, len(df)), min(600, len(df)), 100)
    
    target_options = ["Electricity_Price_CAD_MWh", "Energy_Demand_MW", "Energy_Price_CAD", "Gas_Price_CAD_GJ"]
    available_targets = [col for col in target_options if col in df.columns]
    target = st.sidebar.selectbox("Target Variable", available_targets if available_targets else df.select_dtypes(include=np.number).columns.tolist())
    
    st.sidebar.subheader("Models")
    run_arima_cb = st.sidebar.checkbox("ARIMA", True)
    run_sarima_cb = st.sidebar.checkbox("SARIMA", True)
    run_decomp_cb = st.sidebar.checkbox("Seasonal Decomposition", True)
    run_lstm_cb = st.sidebar.checkbox("LSTM", True)
    run_rnn_cb = st.sidebar.checkbox("RNN", True)
    
    lookback = st.sidebar.slider("Lookback (LSTM/RNN)", 12, 72, 24, 12)
    epochs = st.sidebar.slider("Epochs", 20, 100, 50, 10)
    season_period = st.sidebar.number_input("Seasonal Period", 12, 168, 24, 12)
    
    run_button = st.sidebar.button("🚀 Run Pipeline", type="primary")
    
    # =========================================================
    # RUN PIPELINE
    # =========================================================
    
    if run_button:
        df_used = df.tail(use_last_n).copy()
        series = df_used[target]
        
        # Stationarity
        st.header(" Stationarity Test")
        stationarity = check_stationarity(series)
        col1, col2, col3 = st.columns(3)
        col1.metric("ADF Statistic", f"{-stationarity['ADF_Statistic']:.4f}")
        col2.metric("p-value", f"{stationarity['p_value']:.4f}")
        col3.metric("Status", " Stationary" if stationarity['stationary'] else "❌ Non-Stationary")
        st.info(stationarity['summary'])
        
        # ACF/PACF
        st.header(" ACF & PACF")
        st.pyplot(plot_acf_pacf(series))
        
        # Run Models
        st.header(" Model Results")
        results = {}
        metrics_list = []
        
        # ARIMA
        if run_arima_cb:
            with st.spinner("Running ARIMA..."):
                try:
                    results["ARIMA"] = run_arima(series)
                    metrics_list.append({"Model": "ARIMA", **results["ARIMA"]["metrics"]})
                except Exception as e:
                    st.error(f"ARIMA failed: {e}")
        
        # SARIMA
        if run_sarima_cb:
            with st.spinner("Running SARIMA..."):
                try:
                    results["SARIMA"] = run_sarima(series, seasonal_order=(1,1,1,season_period))
                    metrics_list.append({"Model": "SARIMA", **results["SARIMA"]["metrics"]})
                except Exception as e:
                    st.error(f"SARIMA failed: {e}")
        
        # Seasonal Decomposition
        if run_decomp_cb:
            with st.spinner("Running Decomposition..."):
                try:
                    results["Decomposition"] = run_decomposition(series, season_period)
                except Exception as e:
                    st.error(f"Decomposition failed: {e}")
        
        # LSTM
        if run_lstm_cb:
            with st.spinner("Running LSTM..."):
                try:
                    results["LSTM"] = run_lstm(series, lookback, epochs)
                    if "metrics" in results["LSTM"]:
                        metrics_list.append({"Model": "LSTM", **results["LSTM"]["metrics"]})
                except Exception as e:
                    st.error(f"LSTM failed: {e}")
        
        # RNN
        if run_rnn_cb:
            with st.spinner("Running RNN..."):
                try:
                    results["RNN"] = run_rnn(series, lookback, epochs)
                    if "metrics" in results["RNN"]:
                        metrics_list.append({"Model": "RNN", **results["RNN"]["metrics"]})
                except Exception as e:
                    st.error(f"RNN failed: {e}")
        
        # =========================================================
        # COMPARISON - FIXED (removed .style)
        # =========================================================
        
        if metrics_list:
            st.subheader(" Model Comparison")
            df_metrics = pd.DataFrame(metrics_list).sort_values("RMSE")
            
            # Display without .style to avoid jinja2 dependency
            st.dataframe(df_metrics)
            
            # Highlight best model with a success message
            best_model = df_metrics.iloc[0]["Model"]
            st.success(f" Best Model: {best_model} (RMSE: {df_metrics.iloc[0]['RMSE']:.4f})")
            
            # Comparison Plot
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(series.values[-200:], label="Actual", linewidth=2, color="black")
            colors = ["blue", "orange", "green", "red", "purple"]
            for i, (name, res) in enumerate(results.items()):
                if "forecast" in res:
                    ax.plot(res["forecast"].values[:200], label=name, linestyle="--", alpha=0.7, color=colors[i%len(colors)])
                elif "predictions" in res:
                    ax.plot(res["predictions"][:200], label=name, linestyle="--", alpha=0.7, color=colors[i%len(colors)])
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            best_model = "None"
        
        # =========================================================
        # INDIVIDUAL RESULTS
        # =========================================================
        
        st.header(" Detailed Results")
        
        # ARIMA
        if "ARIMA" in results:
            with st.expander("ARIMA Results"):
                res = results["ARIMA"]
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{res['metrics']['MAE']:.4f}")
                col2.metric("RMSE", f"{res['metrics']['RMSE']:.4f}")
                col3.metric("R²", f"{res['metrics']['R2']:.4f}")
                
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(res["train"].index[-100:], res["train"].values[-100:], label="Train")
                ax.plot(res["test"].index, res["test"].values, label="Test")
                ax.plot(res["forecast"].index, res["forecast"].values, label="Forecast")
                ax.axvline(res["test"].index[0], color="red", linestyle="--")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                st.text("Model Summary (first 500 chars):")
                st.text(res["summary"][:500] + "...")
        
        # SARIMA
        if "SARIMA" in results:
            with st.expander("SARIMA Results"):
                res = results["SARIMA"]
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{res['metrics']['MAE']:.4f}")
                col2.metric("RMSE", f"{res['metrics']['RMSE']:.4f}")
                col3.metric("R²", f"{res['metrics']['R2']:.4f}")
                
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(res["train"].index[-100:], res["train"].values[-100:], label="Train")
                ax.plot(res["test"].index, res["test"].values, label="Test")
                ax.plot(res["forecast"].index, res["forecast"].values, label="Forecast")
                ax.axvline(res["test"].index[0], color="red", linestyle="--")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                st.text("Model Summary (first 500 chars):")
                st.text(res["summary"][:500] + "...")
        
   
        # LSTM
        if "LSTM" in results:
            with st.expander("LSTM Results"):
                res = results["LSTM"]
                if "error" in res:
                    st.error(res["error"])
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MAE", f"{res['metrics']['MAE']:.4f}")
                    col2.metric("RMSE", f"{res['metrics']['RMSE']:.4f}")
                    col3.metric("R²", f"{res['metrics']['R2']:.4f}")
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(res["history"].history["loss"], label="Train Loss")
                    ax.plot(res["history"].history["val_loss"], label="Val Loss")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(res["actual"][:200], label="Actual", linewidth=2)
                    ax.plot(res["predictions"][:200], label="Predicted", linestyle="--", linewidth=2)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
        # RNN
        if "RNN" in results:
            with st.expander("RNN Results"):
                res = results["RNN"]
                if "error" in res:
                    st.error(res["error"])
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MAE", f"{res['metrics']['MAE']:.4f}")
                    col2.metric("RMSE", f"{res['metrics']['RMSE']:.4f}")
                    col3.metric("R²", f"{res['metrics']['R2']:.4f}")
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(res["history"].history["loss"], label="Train Loss")
                    ax.plot(res["history"].history["val_loss"], label="Val Loss")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(res["actual"][:200], label="Actual", linewidth=2)
                    ax.plot(res["predictions"][:200], label="Predicted", linestyle="--", linewidth=2)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
    
        # =========================================================
        # EXPORT
        # =========================================================
        
        if metrics_list:
            st.header(" Export")
            csv = pd.DataFrame(metrics_list).to_csv(index=False)
            st.download_button(" Download Results CSV", csv, "forecast_results.csv", "text/csv")
        
        st.success(" Pipeline Complete!")