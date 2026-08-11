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
    page_title="AL Analystic, Seasonal, Time Series",
    layout="wide")

# =========================================================
# HEADER
# =========================================================

st.title(" AL Analystic, Seasonal, Time Series")

st.subheader("AI-Powered Energy Analytics & LLM interpretation")

st.info("""
### About This Platform

This AI-powered Energy Intelligence Platform transforms raw energy market data into actionable insights using advanced analytics, visualization, statistical modeling, and Large Language Models (LLMs).

### Advanced Analytics Modules

 Time Series Analysis
- Hourly, weekly, monthly, and yearly energy market trends.

 Seasonal Market Analysis
- Seasonal demand, pricing, temperature impacts, and energy costs.

 Behavioral Energy Analysis
- Consumer demand behavior, peak-hour usage, and cost-driving patterns.

 AI Energy Analyst
- Generates executive-level insights, risk assessment,
forecast readiness, and business recommendations.
""")

# =========================================================
# FILE UPLOAD
# =========================================================


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

Write professionally for executives.

"""
    return call_llm(prompt)

# =========================================================
# TIME SERIES ANALYSIS
# =========================================================
def time_series_analysis(df):

    df=df.copy()
    results={}

    # -------------------------
    # Hourly Price Pattern
    # -------------------------
    # FIX: Use the "Hour" column instead of df.index.hour
    hourly_avg = (df.groupby("Hour")["Electricity_Price_CAD_MWh"].mean())

    fig,ax = plt.subplots( figsize=(10,5))

    sns.lineplot(
        x=hourly_avg.index,
        y=hourly_avg.values,
        marker="o",
        ax=ax)

    ax.set_title("Average Hourly Price Pattern")
    ax.set_xlabel( "Hour")
    ax.set_ylabel("Price CAD/MWh")
    ax.set_xticks(range(24))
    st.pyplot(fig)
    plt.close()

    # -------------------------
    # Weekly Price Pattern
    # -------------------------
    weekday_avg=(df.groupby("Weekday") ["Electricity_Price_CAD_MWh"].mean())
    fig,ax=plt.subplots( figsize=(10,5))
    weekday_avg.plot(marker="o", ax=ax)
    ax.set_title("Weekly Electricity Price Pattern")
    st.pyplot(fig)
    plt.close()

    # -------------------------
    # Year Demand
    # -------------------------
    yearly=( df.groupby("Year") ["AIL_Total_Demand_MW"] .sum())
    fig,ax=plt.subplots( figsize=(10,5))
    yearly.plot( marker="o",ax=ax)
    ax.set_title( "Yearly Energy Demand")
    st.pyplot(fig)
    plt.close()

    # -------------------------
    # Demand vs Price
    # -------------------------
    fig,ax=plt.subplots( figsize=(8,5))
    sns.scatterplot(
        data=df,
        x="AIL_Total_Demand_MW",
        y="Electricity_Price_CAD_MWh",
        ax=ax)
    ax.set_title( "Demand vs Electricity Price")
    st.pyplot(fig)
    plt.close()

    # -------------------------
    # Monthly Consumption
    # -------------------------
    monthly=(df.groupby("Month")["AIL_Total_Demand_MW"].mean())
    fig,ax=plt.subplots( figsize=(10,5))
    monthly.plot( kind="bar", ax=ax)
    ax.set_title( "Monthly Energy Consumption")
    st.pyplot(fig)
    plt.close()

    # -------------------------
    # Gas Trend
    # -------------------------

    if "Gas_Price_CAD_GJ" in df.columns:

        gas=(df.groupby("Month") ["Gas_Price_CAD_GJ"].mean())
        fig,ax=plt.subplots( figsize=(10,5))
        gas.plot(marker="o",ax=ax)
        ax.set_title( "Gas Price Trend")
        st.pyplot(fig)
        plt.close()

    results["Average Price"]=round(df["Electricity_Price_CAD_MWh"].mean(),2)
    results["Average Demand"]=round(df["AIL_Total_Demand_MW"].mean(),2)
    results["Peak Demand"]=round( df["AIL_Total_Demand_MW"].max(),2)
    results["Demand Price Correlation"]=round(df[ "AIL_Total_Demand_MW"].corr(df["Electricity_Price_CAD_MWh"]), 3)

    return results
# =========================================================
# SEASONAL ANALYSIS
# =========================================================

def seasonal_market_analysis(df):

    df=df.copy()
    results={}


    # CREATE SEASON
    if "Season" not in df.columns:

        def get_season(month):
            if month in [12,1,2]:
                return "Winter"

            elif month in [3,4,5]:
                return "Spring"

            elif month in [6,7,8]:
                return "Summer"

            else:
                return "Fall"

        df["Season"]=df["Month"].apply(get_season)

    # =====================================================
    # PRICE BY SEASON
    # =====================================================
    season_price=(df.groupby("Season")["Electricity_Price_CAD_MWh"].mean())
    fig,ax=plt.subplots(figsize=(10,6))
    season_price.plot(marker="o",ax=ax)
    ax.set_title( "Average Electricity Price by Season")
    ax.set_ylabel( "CAD/MWh")
    ax.grid(True)
    st.pyplot(fig)
    plt.close()


    # TEMPERATURE IMPACT

    if "Temperature_C" in df.columns:
        temp_bins=pd.cut(df["Temperature_C"], bins=10)
        temp_avg=(df.groupby(temp_bins)[ "Electricity_Price_CAD_MWh"] .mean())
        fig,ax=plt.subplots( figsize=(10,6))
        ax.plot( range(len(temp_avg)),temp_avg.values,marker="o")
        ax.set_title( "Price vs Temperature")
        ax.set_xlabel("Temperature Range")
        ax.set_ylabel("Average Price")
        ax.set_xticks( range(len(temp_avg)))
        ax.set_xticklabels([str(x.left)  for x in temp_avg.index],rotation=45)
        st.pyplot(fig)
        plt.close()

    # COST FEATURE
    if "Total_Cost" not in df.columns:
        df["Total_Cost"]=(df["AIL_Total_Demand_MW"] * df["Electricity_Price_CAD_MWh"])

    season_consumption=(df.groupby("Season")["AIL_Total_Demand_MW"].mean())
    season_peak=( df.groupby("Season")["AIL_Total_Demand_MW"].max())
    season_cost_mean=(df.groupby("Season")["Total_Cost"].mean())
    season_cost_total=(df.groupby("Season")["Total_Cost"].sum())

    charts=[(season_consumption,"Seasonal Energy Consumption"),
            (season_peak,"Seasonal Peak Load"),
            (season_cost_mean,"Seasonal Average Cost"),
            (season_cost_total,"Seasonal Total Energy Cost")]



    for data,title in charts:
        fig,ax=plt.subplots(figsize=(8,5))
        data.plot(kind="bar",ax=ax)
        ax.set_title(title)
        st.pyplot(fig)
        plt.close()

    # PIE DISTRIBUTION
    season_count=df["Season"].value_counts()
    fig,ax=plt.subplots(1, 2, figsize=(14,7))
    ax[0].pie(season_count, labels=season_count.index, autopct="%1.1f%%")

    ax[0].set_title("Season Distribution")

    if "Peak_Hour" in df.columns:
        peak=df["Peak_Hour"].value_counts()
        ax[1].pie(peak,labels=peak.index, autopct="%1.1f%%")

        ax[1].set_title("Peak Hour Distribution")

    st.pyplot(fig)
    plt.close()



    # LLM RESULTS
    results["Seasonal Price"] = (season_price.to_dict())
    results["Seasonal Demand"] = (season_consumption.to_dict())
    results["Seasonal Cost"] = (season_cost_mean.to_dict())

    return results


# BEHAVIORAL ANALYSIS
def behavioral_analysis(df):

    df=df.copy()
    results={}

    # HOURLY DEMAND BEHAVIOR
    hourly=(df.groupby("Hour")["AIL_Total_Demand_MW" ] .mean())
    fig,ax=plt.subplots( figsize=(10,5))

    hourly.plot( marker="o", ax=ax )
    ax.set_title( "Hourly Demand Behavior")
    ax.set_xlabel("Hour")
    ax.set_ylabel( "Demand MW")
    ax.set_xticks( range(24))
    ax.grid(True)
    st.pyplot(fig)
    plt.close()


    # WEEKDAY BEHAVIOR
    weekday=( df.groupby("Weekday")["AIL_Total_Demand_MW"].mean())
    fig,ax=plt.subplots( figsize=(10,5))
    weekday.plot(kind="bar", ax=ax)
    ax.set_title( "Weekly Consumption Behavior")
    st.pyplot(fig)
    plt.close()



    # YEAR PRICE TREND
    yearly_price=( df.groupby("Year") [ "Electricity_Price_CAD_MWh"].mean())
    fig,ax=plt.subplots( figsize=(8,4))
    yearly_price.plot(marker="o", ax=ax)
    ax.set_title("Electricity Price Trend")
    st.pyplot(fig)
    plt.close()


    # GAS PRICE TREND
    if "Gas_Price_CAD_GJ" in df.columns:
        gas=(df.groupby("Year")["Gas_Price_CAD_GJ"].mean())
        fig,ax=plt.subplots( figsize=(8,4))
        gas.plot( marker="o",ax=ax)
        ax.set_title("Gas Price Trend")
        st.pyplot(fig)
        plt.close()

    # GAS VS ELECTRICITY
    if "Gas_Price_CAD_GJ" in df.columns:
        fig,ax=plt.subplots( figsize=(8,5))
        sns.scatterplot(data=df,
            x="Gas_Price_CAD_GJ",
            y="Electricity_Price_CAD_MWh", ax=ax)
        ax.set_title( "Gas vs Electricity Price")
        st.pyplot(fig)
        plt.close()


    # RESULTS
    results["Average Demand"]=round(df["AIL_Total_Demand_MW"].mean(),2)
    results["Average Price"]=round(df["Electricity_Price_CAD_MWh"].mean(), 2)
    results["Peak Demand"]=round( df["AIL_Total_Demand_MW"].max(), 2 )
    results["Average Cost"]=round(df["Total_Cost_Hour"].mean(),2)

    return results

# =========================================================
# CONTROL PANEL
# =========================================================
option = st.selectbox("Choose Analysis Module",["Time Series Analysis","Seasonal Market Analysis", "Behavioral Energy Analysis"])

if option == "Time Series Analysis":
    analysis_results = time_series_analysis(df)

elif option == "Seasonal Market Analysis":
    analysis_results = seasonal_market_analysis(df)

elif option == "Behavioral Energy Analysis":
    analysis_results = behavioral_analysis(df)



# AI REPORT
st.divider()

st.subheader(" Statistical Analysis Results")
st.json(analysis_results)


st.divider()
if st.button(" Generate AI Energy Analyst Report"):

    with st.spinner( "Generating executive analysis..."):
        report = energy_analysis_agent(option,analysis_results)
        st.subheader("AI Energy Analyst Report")
        st.write(report)



# DOWNLOAD REPORT DATA
json_results = json.dumps(analysis_results,indent=4,default=str)

st.download_button(
    label="Download Analysis JSON",
    data=json_results,
    file_name="energy_analysis_results.json",
    mime="application/json")    