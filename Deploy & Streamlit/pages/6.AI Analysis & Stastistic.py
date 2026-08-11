import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ollama

from scipy import stats
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Energy Statistical Intelligence Platform",
    layout="wide")


# =========================================================
# HEADER
# =========================================================

st.title(" AI Energy Statistical Intelligence Platform")

st.subheader(
    "Correlation | Regression | ANOVA | Chi-Square | LLM Decision Support")


st.info(
"""
This platform performs advanced statistical analysis on energy market data.

Modules:

 Correlation Analysis
- Relationship between energy variables
- Price drivers
- Demand drivers

 Regression Analysis
- Select target variable
- Select feature variables
- Statistical modeling
- R² and significance testing

 ANOVA Analysis
- Compare categorical groups
- Test statistical differences

 Chi-Square Analysis
- Test categorical relationships

 AI Energy Analyst
- Executive interpretation
- Market risks
- Business recommendations
""")



# =========================================================
# DATA UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Energy Dataset",
    type=["csv","xlsx","xls"])


@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = (
        df.columns
        .astype(str)
        .str.strip())

    return df


# LLM FUNCTION
def call_llm(prompt):
    try:
        response = ollama.chat(
            model="llama3.2:latest",messages=[{"role":"user","content":prompt}],options={"temperature":0.2})
        return response["message"]["content"]
    except Exception as e:
        return str(e)



def energy_analysis_agent(module, results):

    prompt=f"""
You are a Senior Energy Market Data Scientist.
Analyze this statistical energy analysis.

Analysis Module:
{module}


Statistical Results:
{json.dumps(results,indent=2,default=str)}


Provide:
1. Executive Summary
2. Statistical Interpretation
3. Important Energy Drivers
4. Market Risk Analysis
5. Operational Impact
6. Feature Engineering Recommendations
7. Forecasting Readiness
8. Business Recommendations
9. Procurement Strategy
10. Data Quality Score (0-100)

Write for energy executives.
"""
    return call_llm(prompt)


# =========================================================
# CHECK FILE
# =========================================================

if uploaded_file is None:

    st.warning("Please upload energy dataset.")
    st.stop()

df = load_data(uploaded_file)

# =========================================================
# DATASET INFORMATION
# =========================================================

st.subheader("Dataset Information")

c1,c2,c3,c4 = st.columns(4)

c1.metric( "Rows",df.shape[0])
c2.metric( "Columns",df.shape[1])
c3.metric("Duplicate Rows",int(df.duplicated().sum()))
c4.metric("Missing Values", int(df.isna().sum().sum()))


with st.expander( "View Dataset"):
    st.dataframe(df.head(5),use_container_width=True)

# =========================================================
# MAIN MODULE SELECTOR
# =========================================================
analysis_type = st.selectbox("Select Statistical Analysis", [
        "Correlation Analysis",
        "Regression Analysis",
        "ANOVA Analysis",
        "Chi-Square Analysis"])

# =========================================================
# CORRELATION ANALYSIS
# =========================================================
def correlation_analysis(df):

    results = {}
    st.subheader( " Correlation Analysis")
    numeric_cols = (df.select_dtypes( include=np.number) .columns.tolist())

    if len(numeric_cols)<2:
        st.warning("Not enough numeric variables.")
        return {}


    filtered_df = (df[numeric_cols].dropna())
    corr = (filtered_df.corr() )

    # ==============================
    # FULL CORRELATION MATRIX
    # ==============================

    fig,ax = plt.subplots(figsize=(14,8) )
    sns.heatmap( corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Energy Variable Correlation Matrix")
    st.pyplot(fig)
    plt.close()

    # ==============================
    # SELECT TARGET CORRELATION
    # ==============================

    target = st.selectbox("Select Variable for Correlation Ranking",numeric_cols,key="corr_target")

    ranking = (corr[target] .sort_values( ascending=False) )
    st.subheader( f"Top Correlations With {target}")
    fig,ax = plt.subplots(figsize=(8,6))
    sns.heatmap( ranking .head(15) .to_frame(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    plt.close()

    # ==============================
    # LLM RESULTS
    # ==============================


    results["Analysis"] = "Correlation"
    results["Target Variable"] = target
    results["Correlation Ranking"] = ( ranking .round(3) .to_dict())
    strongest_positive = ( ranking .drop(target) .head(5) )
    strongest_negative = (ranking .drop(target) .sort_values() .head(5))
    results["Strong Positive Drivers"] = (strongest_positive.round(3).to_dict())
    results["Strong Negative Drivers"] = ( strongest_negative .round(3) .to_dict())
    results["Average Correlation"] = round(corr.abs() .mean() .mean(), 3)

    return results


# =========================================================
# REGRESSION ANALYSIS
# =========================================================
def regression_analysis(df):
    results={}

    st.subheader( " Regression Statistical Analysis")
    numeric_cols=(df.select_dtypes(include=np.number).columns.tolist())
    if len(numeric_cols)<2:
        st.warning( "Need numeric variables." )

        return {}


    # ==============================
    # SELECT TARGET
    # ==============================
    target = st.selectbox( "Select Target Variable", numeric_cols, key="reg_target")

    features = [
        x for x in numeric_cols
        if x != target ]


    selected_features = st.multiselect( "Select Feature Variables", features,default=features[:5],key="reg_features")
    regression_results=[]

    
    results["Regression Target"]=target
    results["Regression Results"]= (result_df.to_dict( orient="records" ))

    return results

# =========================================================
# ANOVA ANALYSIS
# =========================================================


def anova_analysis(df):

    results={}

    st.subheader(" ANOVA Statistical Analysis")
    categorical_cols=( df.select_dtypes( include=["object","category"]).columns.tolist())
    numeric_cols=(df.select_dtypes(include=np.number).columns.tolist())

    if len(categorical_cols)==0:
        st.warning("No categorical variables found.")
        return {}



    group_col=st.selectbox("Select Group Variable",categorical_cols,key="anova_group")
    target_col=st.selectbox("Select Target Variable",numeric_cols,key="anova_target")
    clean= (df[[group_col,target_col]].dropna())

    groups=[]
    for g in clean[group_col].unique():
        values=(clean[ clean[group_col]==g][target_col].values)
        if len(values)>1:groups.append(values)



    if len(groups)<2:
        st.warning( "ANOVA requires multiple groups." )

        return {}



    f_stat,p_value=f_oneway(*groups)
    if p_value<0.05:
        significance="Statistically Significant"
    else:
        significance="Not Significant"

    group_stats=(clean.groupby(group_col)[target_col].agg(["mean","std","count"]).reset_index())
    st.dataframe(group_stats, use_container_width=True)

    # Boxplot
    fig,ax=plt.subplots(figsize=(10,6))
    sns.boxplot(data=clean, x=group_col,y=target_col,ax=ax)
    ax.set_title(f"{target_col} by {group_col}")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()


    return results


# =========================================================
# CHI-SQUARE ANALYSIS
# =========================================================

def chi_square_analysis(df):


    results={}

    st.subheader( "Chi-Square Statistical Analysis")
    categorical_cols=(df.select_dtypes(include=["object","category"]).columns .tolist())

    for col in df.columns:
        if ( df[col].nunique()<=10 and col not in categorical_cols):
            categorical_cols.append(col)

    if len(categorical_cols)<2:
        st.warning( "Need two categorical variables.")

        return {}



    col1=st.selectbox( "Select First Variable", categorical_cols, key="chi1")
    col2=st.selectbox( "Select Second Variable", [
            x for x in categorical_cols    if x!=col1], key="chi2")

    clean=( df[[col1,col2]] .dropna())
    table=pd.crosstab(clean[col1],clean[col2])

    st.write( "Contingency Table")
    st.dataframe( table)
    chi,p,dof,expected=(chi2_contingency( table ))

    n=table.sum().sum()
    min_dim=min(table.shape)-1

    if min_dim>0:
        cramers_v=np.sqrt(chi /(n * min_dim))

    else:
        cramers_v=0

    if p<0.05:
        result="Significant Association"
    else:
        result="No Significant Association"



    c1,c2,c3=st.columns(3)
    c1.metric( "Chi Square",round(chi,4))
    c2.metric( "P Value", round(p,6))
    c3.metric( "Cramer's V",round(cramers_v,3))

    results["Chi Square"]={
        "Variable 1":col1,
        "Variable 2":col2,
        "Chi2":round(chi,4),
        "P Value":round(p,6),
        "Degrees Freedom":dof,
        "Cramers V":round(cramers_v,3),
        "Interpretation":result}
    
    return results

# =========================================================
# RUN SELECTED ANALYSIS
# =========================================================


if analysis_type=="Correlation Analysis":
    analysis_results=correlation_analysis(df)
elif analysis_type=="Regression Analysis":
    analysis_results=regression_analysis(df)

elif analysis_type=="ANOVA Analysis":
    analysis_results=anova_analysis(df)

else:
    analysis_results=chi_square_analysis(df)



# =========================================================
# AI REPORT
# =========================================================
st.divider()
if st.button(" Generate AI Statistical Energy Report"):
    with st.spinner(
        "AI analyzing statistical results..."):
        report=energy_analysis_agent(
            analysis_type,
            analysis_results)
        st.subheader(
            "AI Energy Analyst Interpretation")
        st.write(report)