# =========================================================
# AI-DRIVEN ENERGY INTELLIGENCE PLATFORM
# MULTI-AGENT ARCHITECTURE (LANGCHAIN)
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
import streamlit as st
from datetime import datetime, timedelta
import base64
from io import BytesIO

# Machine Learning
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency

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

# LangChain
from langchain_community.llms import Ollama
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import tool

import ollama
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# PAGE CONFIG - MUST BE FIRST Streamlit Command
# =========================================================
st.set_page_config(
    page_title="Calgary Energy AI Platform", layout="wide", initial_sidebar_state="expanded")


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.df = None
    st.session_state.cleaned_data = None
    st.session_state.results = {}
    st.session_state.chat_history = []
    st.session_state.processing = False
    st.session_state.current_tab = "Dashboard"

# =========================================================
# DATA LOADING AND PREPROCESSING
# =========================================================

@st.cache_data
def load_data(uploaded_file):
    """Load and preprocess uploaded data"""
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Convert Date column if exists
        df['Date'] = pd.to_datetime({
             'year': df['Year'],
             'month': df['Month'],
             'day': df['Day'],
            'hour': df['Hour']})
        df = df.set_index('Date')

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def get_date_range_info(df):
    """Extract date range information from DataFrame"""
    if df is None or not isinstance(df.index, pd.DatetimeIndex):
        return None, None, 0, 0
    
    start_date = df.index.min()
    end_date = df.index.max()
    total_hours = len(df)
    total_days = total_hours / 24
    
    return start_date, end_date, total_hours, total_days


def calculate_market_indicators(df, price_col="Electricity_Price_CAD_MWh"):
    """Calculate comprehensive market indicators"""
    df = df.copy()
    results = {}
    
    # Basic statistics
    if price_col in df.columns:
        prices = df[price_col]
        results["price_stats"] = {
            "mean": round(prices.mean(), 2),
            "median": round(prices.median(), 2),
            "std": round(prices.std(), 2),
            "min": round(prices.min(), 2),
            "max": round(prices.max(), 2),
            "skewness": round(prices.skew(), 3),
            "kurtosis": round(prices.kurtosis(), 3)}
    
    # Demand statistics
    if "AIL_Total_Demand_MW" in df.columns:
        demand = df["AIL_Total_Demand_MW"]
        results["demand_stats"] = {
            "mean": round(demand.mean(), 2),
            "peak": round(demand.max(), 2),
            "min": round(demand.min(), 2),
            "std": round(demand.std(), 2)}
        
        # Correlation
        if price_col in df.columns:
            results["price_demand_correlation"] = round(df[price_col].corr(demand), 3)
    
    # Temporal patterns
    if "Hour" in df.columns and price_col in df.columns:
        hourly_prices = df.groupby("Hour")[price_col].mean()
        results["peak_hour"] = int(hourly_prices.idxmax())
        results["off_peak_hour"] = int(hourly_prices.idxmin())
        results["hourly_pattern"] = hourly_prices.to_dict()
    
    return results


def anomaly_detection(df, price_col="Electricity_Price_CAD_MWh"):
    """Detect anomalies in price data"""
    if price_col not in df.columns:
        return {}
    
    prices = df[price_col]
    z_score = np.abs((prices - prices.mean()) / prices.std())
    q1, q3 = prices.quantile([0.25, 0.75])
    iqr = q3 - q1
    
    anomaly_mask = (z_score > 3) | (prices < q1 - 1.5 * iqr) | (prices > q3 + 1.5 * iqr)
    
    return {
        "total_anomalies": int(anomaly_mask.sum()),
        "anomaly_ratio": round(anomaly_mask.mean() * 100, 2),
        "mean_anomaly_price": round(prices[anomaly_mask].mean(), 2) if anomaly_mask.sum() > 0 else 0,
        "anomaly_indices": df.index[anomaly_mask].tolist() if hasattr(df.index, 'tolist') else []}


def calculate_spark_spread(df, heat_rate=7.0):
    """Calculate spark spread if gas prices available"""
    if "Gas_Price_CAD_GJ" not in df.columns or "Electricity_Price_CAD_MWh" not in df.columns:
        return {}
    
    df["Spark_Spread"] = df["Electricity_Price_CAD_MWh"] - (df["Gas_Price_CAD_GJ"] * heat_rate)
    spark = df["Spark_Spread"]
    
    return {
        "avg_spread": round(spark.mean(), 2),
        "max_spread": round(spark.max(), 2),
        "min_spread": round(spark.min(), 2),
        "positive_pct": round((spark > 0).mean() * 100, 2),
        "negative_pct": round((spark < 0).mean() * 100, 2)}

# =========================================================
# LLM CONFIGURATION AND AGENT
# =========================================================

def call_llm(prompt, context=""):
    """Call the LLM with prompt and context"""
    try:
        full_prompt = f"""
        Context: {context}
        
        User Query: {prompt}
        
        Please provide a professional, detailed analysis with specific recommendations.
        """
        
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": full_prompt}],
            options={"temperature": 0.3, "top_p": 0.9})
        return response["message"]["content"]
    except Exception as e:
        return f" LLM Error: {str(e)}. Please ensure Ollama is running with llama3.2:latest model."


@tool
def analyze_market_trends(query: str) -> str:
    """Analyze current market trends and provide insights"""
    df = st.session_state.df
    if df is None:
        return "No data available for analysis. Please upload a dataset first."
    
    # Get key metrics
    metrics = calculate_market_indicators(df)
    
    context = f"""
    Current Market Metrics:
    - Average Price: ${metrics.get('price_stats', {}).get('mean', 'N/A')}/MWh
    - Price Volatility: ${metrics.get('price_stats', {}).get('std', 'N/A')}/MWh
    - Peak Hour: {metrics.get('peak_hour', 'N/A')}:00
    - Demand-Price Correlation: {metrics.get('price_demand_correlation', 'N/A')}
    
    Please analyze these metrics and provide market insights.
    """
    
    return call_llm(query, context)

@tool
def recommend_optimization_strategy(query: str) -> str:
    """Provide optimization recommendations for energy operations"""
    df = st.session_state.df
    if df is None:
        return "No data available for optimization recommendations."
    
    metrics = calculate_market_indicators(df)
    anomalies = anomaly_detection(df)
    spark = calculate_spark_spread(df)
    
    context = f"""
    Current Operational Data:
    - Peak Demand: {metrics.get('demand_stats', {}).get('peak', 'N/A')} MW
    - Price Anomalies: {anomalies.get('total_anomalies', 'N/A')} detected
    - Spark Spread: ${spark.get('avg_spread', 'N/A')}/MWh
    - Positive Spark Spread: {spark.get('positive_pct', 'N/A')}% of time
    
    Provide optimization recommendations based on this data.
    """
    
    return call_llm(query, context)

@tool
def financial_risk_assessment(query: str) -> str:
    """Assess financial risks and provide mitigation strategies"""
    df = st.session_state.df
    if df is None:
        return "No data available for risk assessment."
    
    metrics = calculate_market_indicators(df)
    anomalies = anomaly_detection(df)
    
    # Calculate Value at Risk
    if "Electricity_Price_CAD_MWh" in df.columns:
        prices = df["Electricity_Price_CAD_MWh"]
        var_95 = prices.mean() - (1.65 * prices.std())
        var_99 = prices.mean() - (2.33 * prices.std())
    else:
        var_95 = var_99 = "N/A"
    
    context = f"""
    Financial Risk Metrics:
    - Value at Risk (95%): ${var_95}/MWh
    - Value at Risk (99%): ${var_99}/MWh
    - Price Volatility: {metrics.get('price_stats', {}).get('std', 'N/A')}
    - Anomaly Ratio: {anomalies.get('anomaly_ratio', 'N/A')}%
    
    Provide financial risk assessment and mitigation recommendations.
    """
    
    return call_llm(query, context)

# Create tools for LangChain agent
tools = [
    Tool(name="Market Analysis", func=analyze_market_trends, description="Analyze energy market trends and patterns"),
    Tool(name="Optimization Strategy", func=recommend_optimization_strategy, description="Recommend optimization strategies for energy operations"),
    Tool(name="Financial Risk Assessment", func=financial_risk_assessment, description="Assess financial risks and provide mitigation strategies")]

# Initialize agent with memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

try:
    llm = Ollama(model="llama3.2:latest", temperature=0.3)
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
        memory=memory,
        verbose=False,
        handle_parsing_errors=True)
except Exception as e:
    st.warning(f" LangChain agent initialization error: {str(e)}. Some features may be limited.")
    agent = None

# =========================================================
# VISUALIZATION FUNCTIONS
# =========================================================
def create_price_dashboard(df):
    """Create interactive price dashboard"""
    if df is None or "Electricity_Price_CAD_MWh" not in df.columns:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Price Trend", "Hourly Price Pattern", "Price Distribution", "Demand vs Price"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]])
    
    # Price trend
    fig.add_trace(go.Scatter(x=df.index, y=df["Electricity_Price_CAD_MWh"], mode='lines', name='Price', line=dict(color='#1f77b4')),
        row=1, col=1)
    
    # Hourly pattern
    if "Hour" in df.columns:
        hourly = df.groupby("Hour")["Electricity_Price_CAD_MWh"].mean()
        fig.add_trace(
            go.Bar(x=hourly.index, y=hourly.values, name='Hourly Avg', marker_color='#2ca02c'),
            row=1, col=2)
    
    # Distribution
    fig.add_trace(
        go.Histogram(x=df["Electricity_Price_CAD_MWh"], nbinsx=50, name='Distribution', marker_color='#ff7f0e'),
        row=2, col=1)
    
    # Demand vs Price
    if "AIL_Total_Demand_MW" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["AIL_Total_Demand_MW"], y=df["Electricity_Price_CAD_MWh"], mode='markers', name='Demand vs Price', marker=dict(color='#d62728', size=5)),
            row=2, col=2)
    
    fig.update_layout(height=600, showlegend=True, template='plotly_white')
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Hour", row=1, col=2)
    fig.update_xaxes(title_text="Price ($/MWh)", row=2, col=1)
    fig.update_xaxes(title_text="Demand (MW)", row=2, col=2)
    fig.update_yaxes(title_text="Price ($/MWh)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($/MWh)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Price ($/MWh)", row=2, col=2)
    
    return fig


def create_anomaly_dashboard(df):
    """Create anomaly detection dashboard"""
    if df is None or "Electricity_Price_CAD_MWh" not in df.columns:
        return None
    
    anomalies = anomaly_detection(df)
    prices = df["Electricity_Price_CAD_MWh"]
    z_score = np.abs((prices - prices.mean()) / prices.std())
    anomaly_mask = z_score > 3
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df.index, y=prices,
        mode='lines', name='Price',
        line=dict(color='#1f77b4', width=2)))
    
    # Anomalies
    if anomaly_mask.sum() > 0:
        fig.add_trace(go.Scatter(
            x=df.index[anomaly_mask], y=prices[anomaly_mask],
            mode='markers', name='Anomalies',
            marker=dict(color='red', size=10, symbol='x')))
    
    fig.update_layout(
        title="Price Anomaly Detection",
        xaxis_title="Date",
        yaxis_title="Price ($/MWh)",
        height=400,
        template='plotly_white')
    
    return fig

def create_seasonal_dashboard(df):
    """Create seasonal analysis dashboard"""
    if df is None or "Electricity_Price_CAD_MWh" not in df.columns:
        return None
    
    if "Month" not in df.columns:
        df["Month"] = df.index.month
    
    # Monthly average
    monthly = df.groupby("Month")["Electricity_Price_CAD_MWh"].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly.index, y=monthly.values,
        name='Monthly Average',
        marker_color='#2ca02c'))
    
    fig.update_layout(
        title="Seasonal Price Pattern",
        xaxis_title="Month",
        yaxis_title="Price ($/MWh)",
        height=400,
        template='plotly_white')
    
    return fig

def create_spark_dashboard(df):
    """Create spark spread dashboard"""
    if df is None:
        return None
    
    spark_data = calculate_spark_spread(df)
    if not spark_data or "Spark_Spread" not in df.columns:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Spark_Spread"],
        mode='lines', name='Spark Spread',
        line=dict(color='#9467bd', width=2),
        fill='tozeroy'))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="Spark Spread Analysis",
        xaxis_title="Date",
        yaxis_title="Spark Spread ($/MWh)",
        height=400,
        template='plotly_white')
    
    return fig

def create_forecast_dashboard(forecast_data, actual_data):
    """Create forecast vs actual dashboard"""
    fig = go.Figure()
    
    # Actual
    fig.add_trace(go.Scatter(
        x=actual_data.index, y=actual_data.values,
        mode='lines', name='Actual',
        line=dict(color='#1f77b4', width=2)))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_data.index, y=forecast_data.values,
        mode='lines', name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash')))
    
    fig.update_layout(
        title="Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($/MWh)",
        height=400,
        template='plotly_white')
    
    return fig

# =========================================================
# DASHBOARD TABS
# =========================================================

def render_dashboard_tab():
    """Render the main dashboard tab"""
    st.markdown('<div class="main-header"> Energy Intelligence Dashboard</div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    
    if df is None:
        st.info(" Please upload an energy dataset to begin analysis.")
        return
    
    # Get metrics
    metrics = calculate_market_indicators(df)
    anomalies = anomaly_detection(df)
    spark_data = calculate_spark_spread(df)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Price</div>
            <div class="metric-value">${metrics.get('price_stats', {}).get('mean', 'N/A')}</div>
            <div class="metric-label">/MWh</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <div class="metric-label">Peak Demand</div>
            <div class="metric-value">{metrics.get('demand_stats', {}).get('peak', 'N/A')}</div>
            <div class="metric-label">MW</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-label">Price Volatility</div>
            <div class="metric-value">${metrics.get('price_stats', {}).get('std', 'N/A')}</div>
            <div class="metric-label">/MWh</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-label">Anomalies Detected</div>
            <div class="metric-value">{anomalies.get('total_anomalies', 0)}</div>
            <div class="metric-label">{anomalies.get('anomaly_ratio', 0)}% of data</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333;">
            <div class="metric-label">Peak Hour</div>
            <div class="metric-value">{metrics.get('peak_hour', 'N/A')}:00</div>
            <div class="metric-label">Highest price</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #333;">
            <div class="metric-label">Demand-Price Correlation</div>
            <div class="metric-value">{metrics.get('price_demand_correlation', 'N/A')}</div>
            <div class="metric-label">Relationship strength</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);">
            <div class="metric-label">Spark Spread</div>
            <div class="metric-value">${spark_data.get('avg_spread', 'N/A')}</div>
            <div class="metric-label">/MWh</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #fccb90 0%, #d57eeb 100%);">
            <div class="metric-label">Positive Spread</div>
            <div class="metric-value">{spark_data.get('positive_pct', 'N/A')}%</div>
            <div class="metric-label">Profitable periods</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    st.markdown("---")
    
    # Price dashboard
    fig_price = create_price_dashboard(df)
    if fig_price:
        st.plotly_chart(fig_price, use_container_width=True)
    
    # Two column charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly dashboard
        fig_anomaly = create_anomaly_dashboard(df)
        if fig_anomaly:
            st.plotly_chart(fig_anomaly, use_container_width=True)
    
    with col2:
        # Seasonal dashboard
        fig_seasonal = create_seasonal_dashboard(df)
        if fig_seasonal:
            st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # Spark spread
    fig_spark = create_spark_dashboard(df)
    if fig_spark:
        st.plotly_chart(fig_spark, use_container_width=True)
    
    # Market insights
    st.markdown("---")
    st.subheader(" Market Insights")
    
    # Generate insights using LLM
    with st.spinner("Generating market insights..."):
        if agent is not None:
            try:
                insight_prompt = "Provide a comprehensive analysis of the current energy market conditions, including key trends, risks, and opportunities."
                insight_response = agent.invoke(insight_prompt)
                st.markdown(f'<div class="insight-box">{insight_response}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not generate insights: {str(e)}")
        else:
            st.info("Agent not available. Please check Ollama configuration.")


def render_consultant_tab():
    """Render the AI consultant tab"""
    st.markdown('<div class="main-header"> AI Energy Consultant</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Welcome to the AI Energy Consultant! I can help you with:
        -  Market analysis and trend identification
        -  Optimization strategies for energy operations
        -  Financial risk assessment and mitigation
        -  Price forecasting and planning
        -  Sustainability recommendations
        """)
    
    with col2:
        if st.button(" Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Quick action buttons
    st.subheader("Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_queries = {
        "Market Trends": "What are the current market trends and price patterns?",
        "Cost Optimization": "How can I optimize energy costs?",
        "Risk Assessment": "What are the key financial risks?",
        "Forecast": "What is the price forecast for the next week?"}
    
    with col1:
        if st.button(" Market Trends"):
            st.session_state.chat_history.append(("user", quick_queries["Market Trends"]))
            with st.spinner("Analyzing market trends..."):
                response = analyze_market_trends(quick_queries["Market Trends"])
                st.session_state.chat_history.append(("assistant", response))
            st.rerun()
    
    with col2:
        if st.button(" Cost Optimization"):
            st.session_state.chat_history.append(("user", quick_queries["Cost Optimization"]))
            with st.spinner("Generating optimization recommendations..."):
                response = recommend_optimization_strategy(quick_queries["Cost Optimization"])
                st.session_state.chat_history.append(("assistant", response))
            st.rerun()
    
    with col3:
        if st.button(" Risk Assessment"):
            st.session_state.chat_history.append(("user", quick_queries["Risk Assessment"]))
            with st.spinner("Assessing financial risks..."):
                response = financial_risk_assessment(quick_queries["Risk Assessment"])
                st.session_state.chat_history.append(("assistant", response))
            st.rerun()
    
    with col4:
        if st.button(" Price Forecast"):
            st.session_state.chat_history.append(("user", quick_queries["Forecast"]))
            with st.spinner("Generating forecast..."):
                response = call_llm("Provide a price forecast for the next week based on historical patterns.")
                st.session_state.chat_history.append(("assistant", response))
            st.rerun()
    
    # Chat interface
    st.markdown("---")
    st.subheader(" Chat with AI Consultant")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state.chat_history[-20:]:  # Show last 20 messages
            if role == "user":
                st.markdown(f'<div class="chat-message user-message">👤 {message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">🤖 {message}</div>', unsafe_allow_html=True)
    

    # Chat input
    user_query = st.chat_input("Ask me anything about energy markets, operations, or finance...")
    
    if user_query:
        # Add user message
        st.session_state.chat_history.append(("user", user_query))
        
        # Process with agent
        with st.spinner("Thinking..."):
            try:
                if agent is not None:
                    response = agent.invoke(user_query)
                else:
                    # Fallback to direct LLM call
                    context = "You are an expert energy consultant providing professional advice."
                    response = call_llm(user_query, context)
                
                st.session_state.chat_history.append(("assistant", response))
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

def render_forecast_tab():
    """Render the forecasting tab"""
    st.markdown('<div class="main-header"> Price Forecasting</div>', unsafe_allow_html=True)
    
    df = st.session_state.df
    
    if df is None:
        st.info(" Please upload an energy dataset to begin forecasting.")
        return
    
    if "Electricity_Price_CAD_MWh" not in df.columns:
        st.warning(" No price data available for forecasting.")
        return
    
    # Select forecast model
    model_type = st.selectbox(
        "Select Forecast Model",
        ["SARIMA", "LSTM", "Prophet (Coming Soon)"])
    
    forecast_horizon = st.slider("Forecast Horizon (hours)", 24, 168, 48)
    
    if st.button("Generate Forecast", type="primary"):
        price_series = df["Electricity_Price_CAD_MWh"].dropna().sort_index()
        
        with st.spinner(f"Generating {model_type} forecast..."):
            if model_type == "SARIMA":
                # Run SARIMA
                train_size = int(len(price_series) * 0.8)
                train = price_series[:train_size]
                test = price_series[train_size:]
                
                try:
                    # Fit SARIMA model
                    model = SARIMAX(
                        train,
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 24),
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)
                    
                    # Generate forecast
                    forecast = model.forecast(forecast_horizon)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(" Forecast Results")
                        forecast_df = pd.DataFrame({
                            "Date": forecast.index,
                            "Forecasted Price": forecast.values})
                        st.dataframe(forecast_df, use_container_width=True)
                        
                        # Statistics
                        st.subheader(" Forecast Statistics")
                        stats_df = pd.DataFrame({
                            "Metric": ["Mean", "Min", "Max", "Std Dev"],
                            "Value": [
                                round(forecast.mean(), 2),
                                round(forecast.min(), 2),
                                round(forecast.max(), 2),
                                round(forecast.std(), 2)]})
                        st.dataframe(stats_df, use_container_width=True)
                    
                    with col2:
                        # Plot forecast
                        fig = create_forecast_dashboard(forecast, test[:forecast_horizon])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Model diagnostics
                    with st.expander("Model Diagnostics"):
                        st.text(model.summary().as_text())
                    
                except Exception as e:
                    st.error(f"Error fitting SARIMA model: {str(e)}")
            
            elif model_type == "LSTM":
                # Run LSTM
                try:
                    # Prepare data
                    scaler = MinMaxScaler()
                    data = scaler.fit_transform(price_series.values.reshape(-1, 1))
                    
                    seq_len = 24
                    X, y = [], []
                    for i in range(len(data) - seq_len):
                        X.append(data[i:i+seq_len])
                        y.append(data[i+seq_len])
                    
                    X = np.array(X)
                    y = np.array(y)
                    
                    # Split data
                    split = int(len(X) * 0.8)
                    X_train, X_test = X[:split], X[split:]
                    y_train, y_test = y[:split], y[split:]
                    
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    
                    # Build model
                    model = Sequential([
                        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                        Dropout(0.2),
                        LSTM(50, return_sequences=False),
                        Dropout(0.2),
                        Dense(1)])
                    
                    model.compile(optimizer='adam', loss='mse')
                    
                    # Train model
                    history = model.fit(
                        X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.1,
                        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                        verbose=0)
                    
                    # Make predictions
                    last_sequence = data[-seq_len:].reshape(1, seq_len, 1)
                    
                    # Multi-step forecast
                    forecast_scaled = []
                    current_seq = last_sequence.copy()
                    
                    for _ in range(forecast_horizon):
                        next_pred = model.predict(current_seq, verbose=0)
                        forecast_scaled.append(next_pred[0, 0])
                        current_seq = np.roll(current_seq, -1, axis=1)
                        current_seq[0, -1, 0] = next_pred[0, 0]
                    
                    # Inverse transform
                    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
                    
                    # Create forecast index
                    last_date = price_series.index[-1]
                    forecast_index = [last_date + timedelta(hours=i+1) for i in range(forecast_horizon)]
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(" Forecast Results")
                        forecast_df = pd.DataFrame({
                            "Date": forecast_index,
                            "Forecasted Price": forecast})
                        st.dataframe(forecast_df, use_container_width=True)
                    
                    with col2:
                        # Plot forecast
                        fig = go.Figure()
                        
                        # Historical data (last 100 points)
                        fig.add_trace(go.Scatter(
                            x=price_series.index[-100:],
                            y=price_series.values[-100:],
                            mode='lines',
                            name='Historical',
                            line=dict(color='#1f77b4', width=2)))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_index,
                            y=forecast,
                            mode='lines',
                            name='LSTM Forecast',
                            line=dict(color='#ff7f0e', width=2, dash='dash')))
                        
                        fig.update_layout(
                            title="LSTM Price Forecast",
                            xaxis_title="Date",
                            yaxis_title="Price ($/MWh)",
                            height=400,
                            template='plotly_white')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error fitting LSTM model: {str(e)}")
            
            else:
                st.warning(f"{model_type} is not yet implemented.")


# =========================================================
# MAIN APP
# =========================================================

# Render the selected tab
current_tab = st.session_state.get("current_tab", "Dashboard")

if current_tab == "Dashboard":
    render_dashboard_tab()
elif current_tab == "Consultant":
    render_consultant_tab()
elif current_tab == "Forecast":
    render_forecast_tab()
elif current_tab == "Data":
    render_data_tab()

