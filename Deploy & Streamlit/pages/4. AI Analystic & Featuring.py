# =========================================================
# SIMPLIFIED ENERGY ANALYSIS APP
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ollama

# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(page_title="Energy Analysis", layout="wide")

st.title(" Feature Engineering (Lag, Rolling)")
st.markdown("Simple tools for energy market intelligence")

# =========================================================
# DATA LOADING
# =========================================================

uploaded_file = st.file_uploader("Upload your energy data (CSV or Excel)", type=["csv", "xlsx", "xls"])

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

# =========================================================
# SIMPLE FEATURE ENGINEERING
# =========================================================

def create_features(df):
    """Add simple features to the dataset"""
    df = df.copy()
    
    # Find the main columns we need
    demand_col = 'AIL_Total_Demand_MW' if 'AIL_Total_Demand_MW' in df.columns else 'Electricity_kWh'
    price_col = 'Electricity_Price_CAD_MWh' if 'Electricity_Price_CAD_MWh' in df.columns else 'Price_CAD_MWh'
    
    # Make sure columns exist
    if demand_col not in df.columns:
        st.error(f"Required column '{demand_col}' not found!")
        return df
    
    # 1. LAG FEATURES - look back in time
    df['lag_1h'] = df[demand_col].shift(1)      # 1 hour ago
    df['lag_24h'] = df[demand_col].shift(24)    # 1 day ago
    df['lag_168h'] = df[demand_col].shift(168)  # 1 week ago
    
    # 2. ROLLING STATISTICS - average and variation
    df['rolling_mean'] = df[demand_col].rolling(24).mean()  # 24-hour average
    df['rolling_std'] = df[demand_col].rolling(24).std()    # 24-hour volatility
    
    # 3. CHANGE FEATURES - how much it changed
    df['change_1h'] = df[demand_col] - df['lag_1h']
    df['change_pct'] = (df['change_1h'] / (df['lag_1h'] + 1)) * 100
    
    # 4. SPIKE DETECTION - extreme values
    threshold = df[demand_col].quantile(0.95)  # Top 5% are "spikes"
    df['spike'] = (df[demand_col] > threshold).astype(int)
    
    # 5. PRICE features (if we have price data)
    if price_col in df.columns:
        df['price_volatility'] = df[price_col].rolling(24).std()
        df['price_spike'] = (df[price_col] > df[price_col].quantile(0.95)).astype(int)
    
    # Remove empty rows from rolling calculations
    return df.dropna()

# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:
    # Load the data
    df = load_data(uploaded_file)
    st.success(f" Loaded {len(df)} rows of data")
    
    # Show raw data
    with st.expander(" View Raw Data"):
        st.dataframe(df.head(50))
    
    # Create features
    with st.spinner("Creating features..."):
        df_feat = create_features(df)
    
    st.success(f" Created {len(df_feat.columns)} features")
    
    # =========================================================
    # ANALYSIS TABS
    # =========================================================
    
    tab1, tab2, tab3 = st.tabs([" Visualize", " Spike Detection", " AI Insights"])
    
    # TAB 1: VISUALIZATIONS
    with tab1:
        st.subheader("Energy Demand Over Time")
        
        # Get last 500 rows for cleaner plots
        plot_data = df_feat.tail(500)
        
        # Plot 1: Demand with Lags
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(plot_data.index, plot_data.iloc[:, 0], label='Actual', linewidth=2)
        ax.plot(plot_data.index, plot_data['lag_24h'], label='1 Day Ago', alpha=0.7)
        ax.plot(plot_data.index, plot_data['lag_168h'], label='1 Week Ago', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Demand (MW)')
        ax.set_title('Energy Demand - Compare with Past')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Plot 2: Rolling Statistics
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(plot_data.index, plot_data.iloc[:, 0], label='Actual', alpha=0.5)
        ax.plot(plot_data.index, plot_data['rolling_mean'], label='24-hour Average', linewidth=2)
        ax.plot(plot_data.index, plot_data['rolling_std'], label='Volatility', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Demand (MW)')
        ax.set_title('Demand with Rolling Statistics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Plot 3: Price Volatility (if available)
        if 'price_volatility' in df_feat.columns:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(plot_data.index, plot_data['price_volatility'], color='red', label='Price Volatility')
            ax.set_xlabel('Time')
            ax.set_ylabel('Volatility')
            ax.set_title('Electricity Price Volatility')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    # TAB 2: SPIKE DETECTION
    with tab2:
        st.subheader("Detect Extreme Events (Spikes)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            demand_spikes = df_feat['spike'].sum()
            spike_pct = df_feat['spike'].mean() * 100
            st.metric("Demand Spikes", f"{demand_spikes} events", f"{spike_pct:.1f}% of data")
            
            # Plot spikes
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_feat.index, df_feat.iloc[:, 0], alpha=0.7)
            spikes = df_feat[df_feat['spike'] == 1]
            ax.scatter(spikes.index, spikes.iloc[:, 0], color='red', s=30, label='Spikes')
            ax.set_title('Demand Spikes')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            if 'price_spike' in df_feat.columns:
                price_spikes = df_feat['price_spike'].sum()
                price_pct = df_feat['price_spike'].mean() * 100
                st.metric("Price Spikes", f"{price_spikes} events", f"{price_pct:.1f}% of data")
                
                # Plot price spikes
                fig, ax = plt.subplots(figsize=(10, 4))
                price_col = 'Electricity_Price_CAD_MWh' if 'Electricity_Price_CAD_MWh' in df_feat.columns else 'Price_CAD_MWh'
                if price_col in df_feat.columns:
                    ax.plot(df_feat.index, df_feat[price_col], alpha=0.7, color='green')
                    price_spikes_data = df_feat[df_feat['price_spike'] == 1]
                    ax.scatter(price_spikes_data.index, price_spikes_data[price_col], color='red', s=30)
                    ax.set_title('Price Spikes')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            else:
                st.info("Price data not available for spike detection")
    
    # TAB 3: AI INSIGHTS
    with tab3:
        st.subheader("AI Energy Analyst")
        
        # Prepare summary for AI
        summary = {
            "rows": len(df_feat),
            "features": list(df_feat.columns),
            "demand": {
                "mean": float(df_feat.iloc[:, 0].mean()),
                "max": float(df_feat.iloc[:, 0].max()),
                "min": float(df_feat.iloc[:, 0].min())},
            "spikes": int(df_feat['spike'].sum())}
        
        if 'price_spike' in df_feat.columns:
            summary['price_spikes'] = int(df_feat['price_spike'].sum())
        
        st.json(summary)
        
        # Ask questions
        question = st.selectbox( "Ask the AI about your data:", ["What are the key insights from this energy data?",
                "When are the highest demand periods?",
                "What causes spikes in energy demand?",
                "How volatile is the energy market?",
                "Give me a market summary"])
        
        if st.button("Get AI Analysis"):
            with st.spinner("Analyzing with AI..."):
                response = ask_ai(question, json.dumps(summary, indent=2))
                st.markdown(response)
    
