import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ollama


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Analysis & Exploratory Data Analysis (EDA) for Energy Markets",
    layout="wide")


# =========================================================
# HEADER
# =========================================================

st.title(" AI Analysis & Exploratory Data Analysis (EDA) for Energy Markets")

st.subheader(" Energy Analyst AI Assistant")


st.info(
"""
### About This Platform

This AI-powered Energy Intelligence Platform transforms raw energy market data 
into actionable insights using data analytics, statistical modeling, 
visualization, and Large Language Models (LLMs).

The platform helps energy analysts evaluate market behavior, identify price 
drivers, detect risks, and assess forecasting readiness.

### Analysis Modules

 **Statistical Summary**
- Calculates electricity price, demand, volatility, and market relationships.

 **Exploratory Data Analysis (EDA)**
- Discovers hidden patterns, distributions, correlations, and demand-price behavior.

 **Gas Market Behavior Analysis**
- Evaluates hourly and monthly gas price trends and market volatility.

 **AI Energy Analyst**
- Uses LLM reasoning to generate executive-level insights, data quality 
assessment, forecasting recommendations, and business actions.

""")

# =========================================================
# DATA UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload Energy Dataset",
    type=["csv", "xlsx", "xls"])

@st.cache_data
def load_data(uploaded_file):

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file )
    else:
        df = pd.read_excel( uploaded_file)

    return df


# =========================================================
# LLM ENERGY ANALYST
# =========================================================

def call_llm(prompt):
    try:
        response = ollama.chat(model="llama3.2:latest",
                               messages=[{"role":"user", "content":prompt}],
            options={"temperature":0.2})
        return response["message"]["content"]
    except Exception as e:
        return f"LLM Error: {e}"



def energy_analysis_agent( module, results):
    prompt=f"""
You are a Senior Energy Market Data Scientist.
Analyze the following energy market analysis results.
Analysis Module:

{module}
Results:

{json.dumps(results,indent=2,default=str)}


Provide:

1. Executive Summary
2. Market Overview
3. Data Quality Assessment
4. Key Findings
5. Statistical Interpretation
6. Trading Insights
7. Seasonal Analysis
8. Cost Optimization Opportunities
9. Risk Analysis
10. Forecasting Readiness
11. Feature Engineering Recommendations
12. Business Recommendations
13. Procurement Strategy
14. Operational Recommendations
15. Data Quality Score (0-100)

Write professional recommendations for energy executives.
"""
    return call_llm( prompt)



# =========================================================
# MODULE 1
# STATISTICAL SUMMARY
# =========================================================


def statistical_summary(df):

   
    price = df["Electricity_Price_CAD_MWh"]
    demand = df["AIL_Total_Demand_MW"]


    mean_price = price.mean()
    median_price = price.median()
    min_price = price.min()
    max_price = price.max()
    std_price = price.std()
    mean_demand = demand.mean()
    median_demand = demand.median()
    min_demand = demand.min()
    max_demand = demand.max()
    std_demand = demand.std()
    cv_price = std_price / mean_price if mean_price > 0 else 0
    cv_demand = std_demand / mean_demand if mean_demand > 0 else 0

    skew_price = price.skew()
    kurtosis_price = price.kurtosis()

    var_95 = mean_price - (1.65 * std_price)
    var_99 = mean_price - (2.33 * std_price)

    correlation = price.corr(demand)


    if pd.isna(skew_price):
        distribution = "No distribution information available"

    elif skew_price > 0.5:
        distribution = (
            "Right-skewed market (high-price spikes occur occasionally)")

    elif skew_price < -0.5:
        distribution = (
            "Left-skewed market (extended low-price periods)" )

    else:
        distribution = (
            "Approximately symmetric market behavior")

 
    results = {
        "Average Electricity Price ($/MWh)": round(mean_price, 2),
        "Median Electricity Price ($/MWh)": round(median_price, 2),
        "Maximum Electricity Price ($/MWh)": round(max_price, 2),
        "Minimum Electricity Price ($/MWh)": round(min_price, 2),
        "Price Range ($/MWh)": round(max_price - min_price, 2),
        "Price Volatility ($/MWh)": round(std_price, 2),
        "Price CV (%)": round(cv_price * 100, 2),
        "Average Demand (MW)": round(mean_demand, 2),
        "Median Demand (MW)": round(median_demand, 2),
        "Peak Demand (MW)": round(max_demand, 2),
        "Minimum Demand (MW)": round(min_demand, 2),
        "Demand Range (MW)": round(max_demand - min_demand, 2),
        "Demand Volatility (MW)": round(std_demand, 2),
        "Demand CV (%)": round(cv_demand * 100, 2),
        "Price Skewness": round(skew_price, 3),
        "Price Kurtosis": round(kurtosis_price, 3),
        "VaR 95% ($/MWh)": round(var_95, 2),
        "VaR 99% ($/MWh)": round(var_99, 2),
        "Demand-Price Correlation": round(correlation, 3),
        "Distribution Type": distribution,
        "Descriptive Statistics": df.describe().round(2).to_dict()}


    kpi_results = {
        k: v
        for k, v in results.items()
        if k != "Descriptive Statistics"}

    st.subheader(" Key Statistical Metrics")
    st.dataframe(pd.DataFrame( kpi_results.items(), columns=["Metric", "Value"]),use_container_width=True)

    st.subheader(" Descriptive Statistics")
    st.dataframe(df.describe().round(2),use_container_width=True)

    return results

# =========================================================
# MODULE 2
# EDA ANALYSIS
# =========================================================

def exploratory_data_analysis(df):
    results={}
    results["Mean Price"] = (df["Electricity_Price_CAD_MWh"].mean())
    results["Price Standard Deviation"] = ( df["Electricity_Price_CAD_MWh"].std())
    results["Demand Price Correlation"] = (df[ "AIL_Total_Demand_MW" ].corr( df[ "Electricity_Price_CAD_MWh" ]))

    fig,ax = plt.subplots(figsize=(10,5))
    sns.histplot( df[
        "Electricity_Price_CAD_MWh" ], bins=50, kde=True, ax=ax)

    ax.set_title( "Electricity Price Distribution")
    st.pyplot(fig)


    fig,ax = plt.subplots( figsize=(8,5))
    sns.scatterplot(data=df, x="AIL_Total_Demand_MW",
        y="Electricity_Price_CAD_MWh",ax=ax)
    ax.set_title( "Demand vs Electricity Price")
    st.pyplot(fig)

    return results


# =========================================================
# MODULE 3
# GAS MARKET ANALYSIS
# =========================================================


def gas_behavior_plot(df):


      # Date processing
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df.set_index( "Date", inplace=True)



    df = df.copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=["Gas_Price_CAD_GJ"])

    # Time features
    #df["Hour"] = df.index.hour
    #df["Day"] = df.index.day
    #df["Weekday"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["Year"] = df.index.year
    #df["Weekend"] = ( df["Weekday"].isin([5,6]).astype(in
    df["Hour"] = df.index.hour
    df["DayOfWeek"] = df.index.day_name()
    df["month_name"] = df.index.month_name()
    gas = df["Gas_Price_CAD_GJ"]

    results = {
        "Average Gas Price (CAD/GJ)": round(gas.mean(),2),
        "Median Gas Price (CAD/GJ)": round(gas.median(),2),
        "Maximum Gas Price (CAD/GJ)": round(gas.max(),2),
        "Minimum Gas Price (CAD/GJ)": round(gas.min(),2),
        "Gas Price Range": round(gas.max()-gas.min(),2),
        "Gas Volatility": round( gas.std(),2),
        "Coefficient Variation (%)": round((gas.std()/gas.mean())*100,2),
        "Skewness": round(gas.skew(),3),
        "Kurtosis": round(gas.kurtosis(),3),
        "VaR 95%": round( gas.mean()-(1.65*gas.std()),2),
        "VaR 99%": round(gas.mean()-(2.33*gas.std()),2)}

    hourly_day = (df.groupby( ["DayOfWeek","Hour"] )["Gas_Price_CAD_GJ"].mean().reset_index())
    peak_hour = ( hourly_day.loc[hourly_day["Gas_Price_CAD_GJ"].idxmax()])
    low_hour = (hourly_day.loc[ hourly_day["Gas_Price_CAD_GJ"].idxmin()])

    results["Peak Price Hour"] = int( peak_hour["Hour"])
    results["Peak Price Day"] = (peak_hour["DayOfWeek"])
    results["Peak Hour Price"] = round(peak_hour["Gas_Price_CAD_GJ"],2)
    results["Lowest Price Hour"] = int(low_hour["Hour"])
    results["Lowest Price Day"] = (low_hour["DayOfWeek"])



    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(
        data=hourly_day,
        x="Hour",
        y="Gas_Price_CAD_GJ",
        hue="DayOfWeek",
        ax=ax)
    ax.set_title("Gas Price by Hour and Day of Week")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel( "Gas Price (CAD/GJ)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)


    monthly_year = ( df.groupby(["Year","Month"] )["Gas_Price_CAD_GJ"] .mean() .reset_index())
    fig, ax = plt.subplots( figsize=(12,6))

    for year in monthly_year["Year"].unique():
        data = monthly_year[ monthly_year["Year"]==year]
        ax.plot(
            data["Month"],
            data["Gas_Price_CAD_GJ"],
            marker="o",
            label=str(year))
    ax.set_title("Monthly Gas Price Trend by Year")
    ax.set_xlabel( "Month")
    ax.set_ylabel( "Gas Price (CAD/GJ)")
    ax.legend(title="Year")
    ax.grid(alpha=0.3)
    st.pyplot(fig)



    # Average Monthly Gas Price
    month_order=[
        'January','February','March','April',
        'May','June','July','August','September',
        'October','November','December']


    monthly_avg = (df.groupby( "month_name")["Gas_Price_CAD_GJ"].mean().reindex(month_order))


    results["Highest Price Month"] = ( monthly_avg.idxmax())
    results["Highest Monthly Price"] = round(  monthly_avg.max(),2)
    results["Lowest Price Month"] = (monthly_avg.idxmin())
    results["Lowest Monthly Price"] = round( monthly_avg.min(),2)


    fig, ax = plt.subplots(figsize=(14,7))
    bars=ax.bar( monthly_avg.index, monthly_avg.values)
    ax.set_title("Average Monthly Gas Price (CAD/GJ)")
    ax.set_xlabel( "Month")
    ax.set_ylabel("Gas Price (CAD/GJ)")
    plt.xticks( rotation=45)


    for bar in bars:
        height=bar.get_height()
        ax.text(
            bar.get_x()+bar.get_width()/2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom")

    ax.grid( axis="y", alpha=0.3)
    st.pyplot(fig)


    # LLM READY SUMMARY
    results["Hourly Pattern"] = ( hourly_day .round(2) .to_dict("records"))
    results["Monthly Pattern"] = ( monthly_avg .round(2) .to_dict())

    # KPI Display
    st.subheader(  " Gas Market Statistical Summary")

    st.dataframe(pd.DataFrame( results.items(),columns=["Metric","Value" ]),use_container_width=True)
    return results


# =========================================================
# CONTROL PANEL
# =========================================================
st.divider()
if uploaded_file is None:

    st.warning("Please upload a dataset to start analysis.")
    st.stop()

# df is now guaranteed to exist
df = load_data(uploaded_file)

option = st.selectbox( "Choose Analysis Module",["Statistical Summary", "EDA Analysis", "Gas Market Behavior"])

if option == "Statistical Summary":
    analysis_results = statistical_summary( df)

elif option == "EDA Analysis":
    analysis_results = exploratory_data_analysis(df)

elif option == "Gas Market Behavior":
    analysis_results = gas_behavior_plot(df)


# =========================================================
# AI REPORT
# =========================================================


st.divider()

if st.button( " Generate AI Energy Analyst Report"):

    with st.spinner("AI Analyst is analyzing..."):
        report = energy_analysis_agent(option,analysis_results)

        st.subheader( " AI Energy Analyst Report")
        st.write(report)