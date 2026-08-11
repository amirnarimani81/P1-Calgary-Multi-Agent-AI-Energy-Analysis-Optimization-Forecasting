# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ollama



# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Energy Statistical Intelligence Platform",
    layout="wide")



# =========================================================
# HEADER
# =========================================================

st.title(" AI Energy  Intelligence Platform")


st.subheader("Anomaly Detection | Spark Spread | Volatility | LLM Decision Support")


st.info(
"""
This platform performs advanced energy market intelligence analysis.

Modules:


 Market Anomaly Detection
- Electricity price anomaly identification
- Extreme price event detection
- Peak risk analysis


 Spark Spread Analysis
- Gas-to-power profitability
- Generation opportunity detection
- Profitable operating hours


Volatility & Risk Analysis
- Market uncertainty measurement
- Price fluctuation analysis
- Risk classification


 AI Energy Analyst
- Executive interpretation
- Market risk analysis
- Operational recommendations
- Procurement strategy
- Forecasting readiness

""")


# =========================================================
# LLM CONNECTION
# =========================================================
def call_llm(prompt):
    try:
        response = ollama.chat( model="llama3.2:latest",
            messages=[{"role":"user", "content":prompt}],
            options={"temperature":0.2})
        return response["message"]["content"]
    except Exception as e:

        return str(e)



def energy_analysis_agent(module, results):

    prompt=f"""

You are a Senior Energy Market Data Scientist.
Analyze the following energy intelligence results.


Analysis Module:

{module}

Statistical Results:

{json.dumps(results,indent=2,default=str)}

Provide:

1. Executive Summary
2. Statistical Interpretation
3. Important Energy Drivers
4. Electricity Price Risk Analysis
5. Spark Spread Opportunity Analysis
6. Operational Impact
7. Feature Engineering Recommendations
8. Forecasting Readiness
9. Procurement Strategy
10. Data Quality Score (0-100)

Write for energy executives.

"""
    return call_llm(prompt)



# =========================================================
# MARKET INDICATOR ENGINE
# =========================================================
def calculate_market_indicators(df,price_col="Electricity_Price_CAD_MWh"):

    df = df.copy()

    # -----------------------------
    # TIME FEATURES
    # -----------------------------

    df.index = pd.to_datetime(df.index)


    df["hour"] = ( df.index.hour)
    df["weekday"] = (df.index.dayofweek )
    df["month"] = ( df.index.month)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    results={}

    # =====================================================
    # ANOMALY DETECTION
    # =====================================================


    prices=df[price_col]
    z_score = np.abs(( prices - prices.mean())/ prices.std())

    q1,q3 = prices.quantile([0.25,0.75])
    iqr=q3-q1

    anomaly_mask = ((z_score>3)| ( prices < q1-1.5*iqr ) | ( prices > q3+1.5*iqr ))

    df["Price_Anomaly"] = ( anomaly_mask )
    anomaly_price = (prices[anomaly_mask])

    results["anomaly"]={
        "total_anomalies":  int(anomaly_mask.sum()),
        "anomaly_ratio_pct": round(anomaly_mask.mean()*100, 2),
        "mean_anomaly_price": round( anomaly_price.mean(), 2)  if len(anomaly_price)>0  else 0,
        "max_anomaly_price":round(anomaly_price.max(), 2)  if len(anomaly_price)>0  else 0,
        "common_anomaly_hours":df[ anomaly_mask]["hour"].value_counts().head(5).to_dict()}
    

    # =====================================================
    # SPARK SPREAD ANALYSIS
    # =====================================================


    if "Gas_Price_CAD_GJ" in df.columns:

        heat_rate = 7.0
        df["Spark_Spread"] = ( df[price_col] - (df["Gas_Price_CAD_GJ"] * heat_rate ))
        spark = df["Spark_Spread"]

        results["spark_spread"] = {"avg_spark_spread": round(spark.mean(),2),
            "max_spark_spread":round(spark.max(),2),
            "min_spark_spread":round( spark.min(), 2),
            "positive_spread_pct": round((spark > 0).mean()*100, 2),
            "negative_spread_pct":round((spark < 0).mean()*100,2),
            "profitable_hours":df[df["Spark_Spread"] > 0].groupby("hour").size().nlargest(5).index.tolist(),
            "best_generation_hours":df.groupby("hour")["Spark_Spread"].mean().nlargest(5).index.tolist(),
            "worst_generation_hours":df.groupby("hour")["Spark_Spread"].mean().nsmallest(5).index.tolist()}


    else:
        results["spark_spread"] = {
            "avg_spark_spread":0,
            "max_spark_spread":0,
            "min_spark_spread":0,
            "positive_spread_pct":0,
            "negative_spread_pct":0,
            "profitable_hours":[],
            "best_generation_hours":[],
            "worst_generation_hours":[]}

    # =====================================================
    # VOLATILITY & RISK
    # =====================================================
    cv = ( df[price_col].std()/ df[price_col].mean()) * 100

    if cv < 15:
        risk_level="LOW"
    elif cv < 30:
        risk_level="MODERATE"
    else:
        risk_level="HIGH"


    rolling_vol = (df[price_col] .rolling(24) .std())
    results["volatility"]={
        "coefficient_variation_pct":round(cv,2),
        "risk_level":risk_level,
        "current_volatility":round( rolling_vol.iloc[-1], 2)}

    return df, results
# =========================================================
# ANOMALY DASHBOARD
# =========================================================
def anomaly_dashboard(df, results,price_col):
    st.subheader(" Electricity Price Anomaly Detection")
    anomaly = results["anomaly"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Anomalies",anomaly["total_anomalies"])
    c2.metric("Anomaly %",anomaly["anomaly_ratio_pct"])
    c3.metric("Maximum Price", anomaly["max_anomaly_price"])
    c4.metric("Mean Anomaly Price", anomaly["mean_anomaly_price"])


    fig,ax = plt.subplots(figsize=(14,5))
    ax.plot(df.index, df[price_col], label="Electricity Price")
    mask=df["Price_Anomaly"]
    ax.scatter(df.index[mask],df.loc[mask, price_col],s=60,label="Anomaly")
    ax.set_title("Electricity Price Anomalies")
    ax.legend()
    st.pyplot(fig)
    plt.close()

# =========================================================
# SPARK SPREAD DASHBOARD
# =========================================================
def spark_dashboard(df, results):
    st.subheader(" Spark Spread Analysis")
    spark = results["spark_spread"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Average Spread", spark["avg_spark_spread"])
    c2.metric("Maximum Spread",spark["max_spark_spread"])
    c3.metric("Positive Spread %",spark["positive_spread_pct"])
    c4.metric( "Negative Spread %", spark["negative_spread_pct"])


    if "Spark_Spread" in df.columns:
        fig,ax = plt.subplots(figsize=(14,5))
        ax.plot(df.index,df["Spark_Spread"] )
        ax.axhline(0, linestyle="--")
        ax.set_title("Spark Spread Trend")
        st.pyplot(fig)
        plt.close()


# =========================================================
# VOLATILITY DASHBOARD
# =========================================================
def volatility_dashboard(results): 
    st.subheader( " Market Volatility Risk")
    vol = results["volatility"]

    c1,c2,c3 = st.columns(3)
    c1.metric( "CV %", vol["coefficient_variation_pct"])
    c2.metric( "Risk Level", vol["risk_level"])
    c3.metric( "Current Volatility", vol["current_volatility"])

# =========================================================
# AI INTERPRETATION DASHBOARD
# =========================================================
def ai_interpretation_dashboard(results):

    st.subheader( " AI Energy Market Analyst" )

    with st.spinner("Generating executive energy report..."):
        report = energy_analysis_agent(
            "Energy Market Intelligence: "
            "Anomaly + Spark Spread + Volatility",

            results)


    st.markdown( "### Executive Intelligence Report")
    st.write(report)



# =========================================================
# APPLICATION TABS
# =========================================================
tab1,tab2,tab3,tab4 = st.tabs([" Anomaly Detection"," Spark Spread"," Volatility Risk"," AI Interpretation"])
with tab1:
    anomaly_dashboard( df, results,price_col)

with tab2:
    spark_dashboard(df,results)

with tab3:
    volatility_dashboard(results)

with tab4:
    ai_interpretation_dashboard(results)