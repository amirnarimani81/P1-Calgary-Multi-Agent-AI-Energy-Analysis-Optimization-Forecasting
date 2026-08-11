
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ollama

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title=" Data Preview & AI Analysis",
    layout="wide")

st.title(" Data Preview & AI AnalysisCD")
st.markdown("Upload a dataset and receive automated EDA, data quality assessment, and AI-powered recommendations.")

# =====================================================
# OLLAMA FUNCTION
# =====================================================

def call_llm(prompt):
    try:
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[{"role": "user","content": prompt}],
            options={"temperature": 0.2 })

        return response["message"]["content"]

    except Exception as e:

        return f"LLM Error: {e}"


# =====================================================
# DATA CHECK FUNCTION
# =====================================================

def check_df(df):

    report = {}

    report["shape"] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1])}

    report["dtypes"] = (df.dtypes.astype(str).to_dict())
    report["missing_values"] = (df.isnull().sum().to_dict())
    report["duplicate_rows"] = int( df.duplicated().sum())
    report["describe"] = (df.describe(include="all").fillna("").astype(str) .to_dict())

    return report


# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls"])


if uploaded_file is not None:

    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset Loaded Successfully")

# =====================================================
# MAIN APP
# =====================================================


    st.success("Dataset Loaded Successfully")

    # -------------------------
    # DATA PREVIEW
    # -------------------------

    st.subheader(" Data Preview")

    st.dataframe(df.head(),use_container_width=True)

    
# =========================================================
# DATASET INFORMATION + DATE RANGE ANALYSIS
# =========================================================

st.subheader(" Dataset Information")


col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows",f"{df.shape[0]:,}")
col2.metric("Columns", df.shape[1])
col3.metric( "Duplicate Rows",int(df.duplicated().sum()))
col4.metric("Missing Values", f"{df.isna().sum().sum():,}")


# =========================================================
# DATE RANGE ANALYSIS
# =========================================================


st.subheader(" Date Range Analysis")


# Ensure datetime index

if isinstance(df.index, pd.DatetimeIndex):

    start_date = df["Date"].min()
    end_date = df["Date"].max()
    total_hours = len(df)
    total_days = total_hours / 24
    total_months = total_days / 30.4
    total_years = total_days / 365.25


    # Missing timestamp analysis

    expected_hours = pd.date_range(start=start_date,end=end_date, freq="h")
    missing_hours = expected_hours.difference(df.index)
    missing_percentage = ( len(missing_hours) / len(expected_hours)* 100)


    c1, c2, c3, c4 = st.columns(4)
    c1.metric( "Start Date",start_date.strftime("%Y-%m-%d %H:%M") )
    c2.metric("End Date",end_date.strftime("%Y-%m-%d %H:%M"))
    c3.metric("Total Hours",f"{total_hours:,}")
    c4.metric( "Total Days", f"{total_days:,.1f}")



    c5, c6, c7 = st.columns(3)
    c5.metric( "Total Months", f"{total_months:,.1f}")
    c6.metric( "Total Years", f"{total_years:,.2f}")
    c7.metric( "Missing Hours", f"{len(missing_hours):,}")

    st.info(
        f"""
        **Dataset Coverage**
        - Start date: {start_date}
        - End date: {end_date}
        - Total observations: {total_hours:,} hours
        - Missing timestamps: {len(missing_hours):,}
        - Missing percentage: {missing_percentage:.2f}%
        """)


else:
    st.warning(
        """
        Date range analysis requires a datetime index.
        Your dataset currently uses a normal index. """)


# DATA TYPES

    st.subheader("Column Data Types")

    st.dataframe(pd.DataFrame(df.dtypes,columns=["Data Type"]))

# MISSING VALUES

    st.subheader("Missing Values")
    missing_df = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing %": round(df.isnull().mean() * 100, 2)})

    st.dataframe(missing_df)

    fig, ax = plt.subplots(figsize=(14,5))

    sns.heatmap(
        df.isnull(),
        yticklabels=False,
        cbar=False,
        cmap="Blues",
        ax=ax)

    ax.set_title(
        "Missing Value Heatmap")

    st.pyplot(fig)

    # -------------------------
    # DESCRIPTIVE STATS
    # -------------------------

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(),use_container_width=True)

    # -------------------------
    # OUTLIER ANALYSIS
    # -------------------------

    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:

        st.subheader( " Outlier Detection")

        target_col = st.selectbox( "Select Numeric Column",numeric_cols)

        q1 = df[target_col].quantile(0.25)
        q3 = df[target_col].quantile(0.75)

        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((df[target_col] < lower)|(df[target_col] > upper))

        st.metric(
            "Outlier Count", int(outliers.sum()))

        fig, ax = plt.subplots(figsize=(14,5))
        ax.scatter(
            df.index,
            df[target_col],
            alpha=0.6,
            label="Normal")

        ax.scatter(
            df.index[outliers],
            df[target_col][outliers],
            s=80,
            label="Outlier")

        ax.set_title(
            f"IQR Outlier Analysis - {target_col}")

        ax.legend()
        st.pyplot(fig)

    # -------------------------
    # CREATE REPORT
    # -------------------------

    report = check_df(df)

    # -------------------------
    # AI DATA QUALITY ANALYSIS
    # -------------------------

    st.subheader(" AI Data Quality Consultant")

    if st.button(
        "Generate AI Analysis"):

        with st.spinner(
            "Analyzing Dataset..."):

            prompt = f"""
You are a Senior Energy Market Analyst.

Analyze the dataset summary below.

{json.dumps(report, indent=2)}

Provide:

1. Data quality issues
2. Missing value assessment
3. Duplicate row assessment
4. Outlier concerns
5. Feature engineering suggestions
6. Forecasting readiness
7. Recommended data cleaning
8. Business insights
9.Data quality score (0-100)
10. Forecasting readiness score (0-100)

Provide professional recommendations.
"""

            result = call_llm(prompt)
            st.markdown(result)

