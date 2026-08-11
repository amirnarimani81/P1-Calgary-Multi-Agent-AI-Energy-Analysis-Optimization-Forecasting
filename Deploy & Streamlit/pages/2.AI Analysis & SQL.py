# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import ollama
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="Energy SQL Analytics", layout="wide")

st.title(" Energy SQL Analytics")
st.markdown("### SQL Queries | AI Insights | Interactive Dashboard")

st.info("""
**This platform performs SQL-based analysis on energy data with AI-powered insights.**

** SQL Analytics:**
- Peak/Off-Peak Hours Analysis
- Seasonal & Temperature Impact
- Anomaly Detection
- Weekend vs Weekday Pricing
- Price Index Calculations

**AI Features:**
- Automated Query Interpretation
- Business Insights Generation
- Pattern Recognition
- Recommendations
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
        df['datetime'] = pd.to_datetime(df['Date'])
    else:
        df['datetime'] = pd.to_datetime(df.index)
    
    return df

# =========================================================
# LLM HELPER
# =========================================================

def get_ai_insight(query_name, data):
    """Get AI insights for query results"""
    try:
        prompt = f"""
You are an Energy Analyst. Analyze this {query_name} data:

{data.to_string()}

Provide 2-3 key business insights.
"""
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}
        )
        return response["message"]["content"]
    except:
        return " AI insights unavailable"

# =========================================================
# SQL QUERY FUNCTIONS
# =========================================================

def run_queries(df):
    """Run all SQL queries and return results"""
    
    # Prepare data
    sql_df = df.reset_index().copy()
    sql_df['hour'] = sql_df['datetime'].dt.hour
    sql_df['month'] = sql_df['datetime'].dt.month
    sql_df['weekday'] = sql_df['datetime'].dt.weekday
    sql_df['is_weekend'] = (sql_df['weekday'] >= 5).astype(int)
    
    # Create database
    conn = sqlite3.connect(":memory:")
    sql_df.to_sql("energy", conn, index=False)
    
    results = {}
    
    # Helper to run query
    def run_query(name, query):
        try:
            df_result = pd.read_sql_query(query, conn)
            results[name] = df_result
            return df_result
        except Exception as e:
            results[name] = pd.DataFrame({"Error": [str(e)]})
            return results[name]
    
    # 1. PEAK HOURS
    run_query(
        "peak_hours",
        """
        SELECT hour, 
               ROUND(AVG(Electricity_Price_CAD_MWh), 2) as avg_price,
               COUNT(*) as samples
        FROM energy 
        WHERE Electricity_Price_CAD_MWh IS NOT NULL
        GROUP BY hour 
        ORDER BY avg_price DESC 
        LIMIT 5
        """)
    
    # 2. CHEAP HOURS
    run_query(
        "cheap_hours",
        """
        SELECT hour, 
               ROUND(AVG(Electricity_Price_CAD_MWh), 2) as avg_price,
               COUNT(*) as samples
        FROM energy 
        WHERE Electricity_Price_CAD_MWh IS NOT NULL
        GROUP BY hour 
        ORDER BY avg_price ASC 
        LIMIT 5
        """)
    
    # 3. WEEKEND VS WEEKDAY
    run_query(
        "weekend_vs_weekday",
        """
        SELECT 
            CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END as day_type,
            ROUND(AVG(Electricity_Price_CAD_MWh), 2) as avg_price,
            COUNT(*) as samples
        FROM energy 
        WHERE Electricity_Price_CAD_MWh IS NOT NULL
        GROUP BY is_weekend
        """)
    
    # 4. SEASONAL PATTERN
    run_query(
        "seasonal",
        """
        SELECT month, 
               ROUND(AVG(Electricity_Price_CAD_MWh), 2) as avg_price
        FROM energy 
        WHERE Electricity_Price_CAD_MWh IS NOT NULL
        GROUP BY month 
        ORDER BY month
        """)
    
    # 5. TEMPERATURE IMPACT (if column exists)
    if 'Temperature_C' in sql_df.columns:
        run_query(
            "temperature",
            """
            SELECT 
                CASE 
                    WHEN Temperature_C < -10 THEN 'Very Cold'
                    WHEN Temperature_C < 0 THEN 'Cold'
                    WHEN Temperature_C < 10 THEN 'Mild'
                    WHEN Temperature_C < 20 THEN 'Warm'
                    ELSE 'Hot'
                END as temp_range,
                ROUND(AVG(Electricity_Price_CAD_MWh), 2) as avg_price
            FROM energy 
            WHERE Electricity_Price_CAD_MWh IS NOT NULL
            GROUP BY temp_range
            ORDER BY avg_price DESC
            """ )
    
    # 6. PRICE ANOMALIES
    run_query(
        "anomalies",
        """
        WITH stats AS (
            SELECT AVG(Electricity_Price_CAD_MWh) as mean_price,
                   STDEV(Electricity_Price_CAD_MWh) as std_price
            FROM energy 
            WHERE Electricity_Price_CAD_MWh IS NOT NULL
        )
        SELECT datetime, hour,
               ROUND(Electricity_Price_CAD_MWh, 2) as price,
               ROUND((Electricity_Price_CAD_MWh - mean_price) / std_price, 2) as z_score
        FROM energy, stats
        WHERE Electricity_Price_CAD_MWh IS NOT NULL
        AND ABS(Electricity_Price_CAD_MWh - mean_price) > 2.5 * std_price
        LIMIT 10
        """)
    
    # 7. PEAK INDEX
    run_query(
        "peak_index",
        """
        WITH peak AS (
            SELECT AVG(Electricity_Price_CAD_MWh) as peak_price 
            FROM energy 
            WHERE hour IN (7,8,9,17,18,19)
        ),
        offpeak AS (
            SELECT AVG(Electricity_Price_CAD_MWh) as offpeak_price 
            FROM energy 
            WHERE hour BETWEEN 2 AND 5
        )
        SELECT ROUND(peak_price, 2) as peak_price,
               ROUND(offpeak_price, 2) as offpeak_price,
               ROUND(peak_price / offpeak_price, 2) as peak_index
        FROM peak, offpeak
        """)
    
    conn.close()
    return results

# =========================================================
# PLOTTING FUNCTIONS
# =========================================================

def plot_data(df, title, x_col, y_col, kind='bar'):
    """Simple plotting function"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if kind == 'bar':
        ax.bar(df[x_col], df[y_col], color='skyblue', alpha=0.7)
    elif kind == 'line':
        ax.plot(df[x_col], df[y_col], marker='o', linewidth=2)
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# =========================================================
# DISPLAY FUNCTIONS
# =========================================================

def display_query_result(name, df, plot_func=None):
    """Display query results with AI insights"""
    
    if df is None or df.empty or 'Error' in df.columns:
        st.warning(f"No data for {name}")
        return
    
    with st.expander(f" {name.replace('_', ' ').title()}", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df, use_container_width=True)
            
            if plot_func:
                fig = plot_func(df)
                st.pyplot(fig)
        
        with col2:
            st.subheader("AI Insights")
            with st.spinner("Getting insights..."):
                insight = get_ai_insight(name, df)
                st.markdown(insight)
            
            # Show SQL
            with st.expander(" View SQL"):
                st.code("""
                -- Query for {}
                """.format(name), language="sql")

# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    st.success(f" Loaded {len(df)} rows")
    
    # Preview
    with st.expander(" Data Preview"):
        st.dataframe(df.head(50))
        st.write(f"**Columns:** {', '.join(df.columns)}")
    
    # =========================================================
    # SIDEBAR
    # =========================================================
    
    st.sidebar.header(" Settings")
    
    # Target selection
    default_targets = ['Electricity_Price_CAD_MWh', 'Energy_Demand_MW']
    available = [col for col in default_targets if col in df.columns]
    
    if available:
        target = st.sidebar.selectbox("Target Variable", available)
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        target = st.sidebar.selectbox("Target Variable", numeric_cols)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Click below to run all queries")
    run_btn = st.sidebar.button(" Run Analysis", type="primary", use_container_width=True)
    
    # =========================================================
    # RUN ANALYSIS
    # =========================================================
    
    if run_btn:
        st.header(" Analysis Results")
        
        with st.spinner("Running SQL queries..."):
            results = run_queries(df)
        
        # Display each result
        tabs = st.tabs(["Peak Hours", "Cheap Hours", "Weekend vs Weekday", 
                       "Seasonal", "Temperature", "Anomalies", "Peak Index"])
        
        # Tab 1: Peak Hours
        with tabs[0]:
            if 'peak_hours' in results:
                df_peak = results['peak_hours']
                display_query_result(
                    "Peak Hours", 
                    df_peak,
                    lambda x: plot_data(x, "Top 5 Peak Hours", "hour", "avg_price"))
        
        # Tab 2: Cheap Hours
        with tabs[1]:
            if 'cheap_hours' in results:
                df_cheap = results['cheap_hours']
                display_query_result(
                    "Cheap Hours",
                    df_cheap,
                    lambda x: plot_data(x, "Top 5 Cheapest Hours", "hour", "avg_price"))
        
        # Tab 3: Weekend vs Weekday
        with tabs[2]:
            if 'weekend_vs_weekday' in results:
                df_weekend = results['weekend_vs_weekday']
                
                col1, col2 = st.columns(2)
                with col1:
                    display_query_result("Weekend vs Weekday", df_weekend)
                
                with col2:
                    try:
                        weekday = df_weekend[df_weekend['day_type'] == 'Weekday']['avg_price'].values[0]
                        weekend = df_weekend[df_weekend['day_type'] == 'Weekend']['avg_price'].values[0]
                        discount = ((weekday - weekend) / weekday) * 100
                        st.metric("Weekend Discount", f"{discount:.1f}%")
                    except:
                        pass
        
        # Tab 4: Seasonal
        with tabs[3]:
            if 'seasonal' in results:
                df_seasonal = results['seasonal']
                display_query_result(
                    "Seasonal Pattern",
                    df_seasonal,
                    lambda x: plot_data(x, "Seasonal Price Pattern", "month", "avg_price", 'line'))
        
        # Tab 5: Temperature
        with tabs[4]:
            if 'temperature' in results:
                df_temp = results['temperature']
                display_query_result(
                    "Temperature Impact",
                    df_temp,
                    lambda x: plot_data(x, "Temperature Impact on Prices", "temp_range", "avg_price"))
            else:
                st.info("Temperature data not available in this dataset")
        
        # Tab 6: Anomalies
        with tabs[5]:
            if 'anomalies' in results:
                df_anomalies = results['anomalies']
                display_query_result("Price Anomalies", df_anomalies)
                
                # Plot anomalies
                if not df_anomalies.empty:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.scatter(range(len(df_anomalies)), df_anomalies['price'], 
                             color='red', s=100, alpha=0.7)
                    ax.axhline(y=df_anomalies['price'].mean(), color='blue', 
                              linestyle='--', label='Mean Price')
                    ax.set_xlabel('Anomaly Index')
                    ax.set_ylabel('Price (CAD/MWh)')
                    ax.set_title('Price Anomalies')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
        # Tab 7: Peak Index
        with tabs[6]:
            if 'peak_index' in results:
                df_index = results['peak_index']
                display_query_result("Peak Index", df_index)
                
                if not df_index.empty:
                    row = df_index.iloc[0]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Peak Price", f"{row['peak_price']:.2f} CAD/MWh")
                    col2.metric("Off-Peak Price", f"{row['offpeak_price']:.2f} CAD/MWh")
                    col3.metric("Peak Index", f"{row['peak_index']:.2f}x")
        
        # =========================================================
        # DASHBOARD
        # =========================================================
        
        st.header(" Dashboard")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        if 'peak_hours' in results and not results['peak_hours'].empty:
            col1.metric("Highest Peak Price", 
                       f"{results['peak_hours']['avg_price'].max():.2f} CAD/MWh")
            col2.metric("Peak Hour", 
                       f"{results['peak_hours'].iloc[0]['hour']}:00")
        
        if 'weekend_vs_weekday' in results and not results['weekend_vs_weekday'].empty:
            try:
                weekday = results['weekend_vs_weekday'][results['weekend_vs_weekday']['day_type'] == 'Weekday']['avg_price'].values[0]
                weekend = results['weekend_vs_weekday'][results['weekend_vs_weekday']['day_type'] == 'Weekend']['avg_price'].values[0]
                col3.metric("Weekend Discount", f"{((weekday - weekend) / weekday * 100):.1f}%")
            except:
                pass
        
        if 'peak_index' in results and not results['peak_index'].empty:
            col4.metric("Peak Index", f"{results['peak_index'].iloc[0]['peak_index']:.2f}x")
        
        # =========================================================
        # EXPORT
        # =========================================================
        
        st.header(" Export Results")
        
        # Combine all results
        all_data = pd.DataFrame()
        for name, df_result in results.items():
            if not df_result.empty and 'Error' not in df_result.columns:
                df_result['query_name'] = name
                all_data = pd.concat([all_data, df_result])
        
        if not all_data.empty:
            csv = all_data.to_csv(index=False)
            st.download_button(
                label=" Download All Results (CSV)",
                data=csv,
                file_name="energy_analysis.csv",
                mime="text/csv")
        
        st.success("✅ Analysis Complete!")
        
else:
    st.info(" Please upload an energy dataset to begin")

st.markdown("---")
st.caption("Built with Streamlit • SQLite • Ollama")