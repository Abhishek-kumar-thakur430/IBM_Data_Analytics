"""
E-Commerce Sales Analytics & Forecasting Project
=================================================
Single-file project combining:
  1. Data cleaning
  2. Exploratory Data Analysis (EDA) + KPI computation
  3. A machine learning model (Linear Regression) to forecast monthly sales
  4. A simple front-end (Streamlit dashboard) to present results to a
     non-technical / business audience

Dataset: "Sample - Superstore" (Kaggle) — 9,994 e-commerce orders, 2014-2017.
Source link: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

Run the dashboard with:
    streamlit run Abhishek Thakur_ecommerce_sales_analytics.py

Run just the analysis/model (no dashboard) with:
    python Abhishek Thakur_ecommerce_sales_analytics.py --cli
"""

import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/superstore.csv"


# ---------------------------------------------------------------------------
# 1. DATA CLEANING
# ---------------------------------------------------------------------------
def load_and_clean_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and apply basic cleaning steps."""
    df = pd.read_csv(path, encoding="latin1")

    # Standardize column names (no spaces, consistent casing is easier to code against)
    df.columns = [c.strip() for c in df.columns]

    # Parse dates
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y", errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y", errors="coerce")

    # Drop exact duplicate rows, if any
    df = df.drop_duplicates()

    # Drop rows with a missing order date (can't be used in a time-based analysis)
    df = df.dropna(subset=["Order Date"])

    # Basic sanity: remove negative/zero quantity or sales rows (data entry errors)
    df = df[(df["Sales"] > 0) & (df["Quantity"] > 0)]

    # Feature engineering used later for KPIs / modeling
    df["Order Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    df["Order Year"] = df["Order Date"].dt.year
    df["Profit Margin"] = df["Profit"] / df["Sales"]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. KPI / EDA
# ---------------------------------------------------------------------------
def compute_kpis(df: pd.DataFrame) -> dict:
    """Top-line KPIs for the executive overview page."""
    total_revenue = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Order ID"].nunique()
    total_customers = df["Customer ID"].nunique()
    avg_order_value = total_revenue / total_orders

    monthly = df.groupby("Order Month")["Sales"].sum().sort_index()
    growth_pct = (
        (monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100
        if len(monthly) >= 2 else np.nan
    )

    return {
        "Total Revenue": total_revenue,
        "Total Profit": total_profit,
        "Total Orders": total_orders,
        "Total Customers": total_customers,
        "Average Order Value": avg_order_value,
        "Latest Month-over-Month Growth %": growth_pct,
    }


def sales_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Category")[["Sales", "Profit"]]
        .sum()
        .sort_values("Sales", ascending=False)
        .reset_index()
    )


def sales_by_region(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Region")[["Sales", "Profit"]]
        .sum()
        .sort_values("Sales", ascending=False)
        .reset_index()
    )


def top_customers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.groupby("Customer Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )


def churn_risk_customers(df: pd.DataFrame, months_inactive: int = 6) -> pd.DataFrame:
    """Flag customers whose last order is more than `months_inactive` months
    before the last date in the dataset — a simple recency-based churn-risk signal."""
    last_date = df["Order Date"].max()
    last_order = df.groupby("Customer Name")["Order Date"].max().reset_index()
    last_order["Months Since Last Order"] = (
        (last_date - last_order["Order Date"]).dt.days / 30
    )
    at_risk = last_order[last_order["Months Since Last Order"] > months_inactive]
    return at_risk.sort_values("Months Since Last Order", ascending=False)


# ---------------------------------------------------------------------------
# 3. PREDICTION MODEL — monthly sales forecast
# ---------------------------------------------------------------------------
def build_forecast_model(df: pd.DataFrame):
    """
    Train a simple Linear Regression model on monthly aggregated sales,
    using lag features + month-of-year seasonality, then forecast the
    next 3 months.
    """
    monthly = df.groupby("Order Month")["Sales"].sum().reset_index()
    monthly = monthly.sort_values("Order Month").reset_index(drop=True)

    # Feature engineering: lag-1, lag-2, month number (for seasonality)
    monthly["lag_1"] = monthly["Sales"].shift(1)
    monthly["lag_2"] = monthly["Sales"].shift(2)
    monthly["month_num"] = monthly["Order Month"].dt.month
    model_df = monthly.dropna().reset_index(drop=True)

    features = ["lag_1", "lag_2", "month_num"]
    X = model_df[features]
    y = model_df["Sales"]

    # Time-ordered train/test split (80/20) — never shuffle time series data
    split = int(len(model_df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    metrics = {}
    if len(X_test) > 0:
        preds = model.predict(X_test)
        metrics = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": mean_squared_error(y_test, preds) ** 0.5,
            "R2": r2_score(y_test, preds),
        }

    # Refit on all data for the actual forward forecast
    model.fit(X, y)

    # Iteratively forecast next 3 months
    history = monthly["Sales"].tolist()
    last_month = monthly["Order Month"].max()
    forecasts = []
    for step in range(1, 4):
        lag_1, lag_2 = history[-1], history[-2]
        future_month = (last_month + pd.DateOffset(months=step))
        row = pd.DataFrame(
            [[lag_1, lag_2, future_month.month]], columns=features
        )
        pred = model.predict(row)[0]
        forecasts.append({"Month": future_month, "Forecasted Sales": pred})
        history.append(pred)

    return model, metrics, pd.DataFrame(forecasts), monthly


# ---------------------------------------------------------------------------
# 4a. CLI MODE — prints results to console, used for quick checks / grading
# ---------------------------------------------------------------------------
def run_cli():
    df = load_and_clean_data()
    print(f"Loaded {len(df):,} clean order rows.\n")

    kpis = compute_kpis(df)
    print("== Executive KPIs ==")
    for k, v in kpis.items():
        print(f"  {k}: {v:,.2f}")

    print("\n== Sales & Profit by Category ==")
    print(sales_by_category(df).to_string(index=False))

    print("\n== Sales & Profit by Region ==")
    print(sales_by_region(df).to_string(index=False))

    print("\n== Top 10 Customers by Sales ==")
    print(top_customers(df).to_string(index=False))

    at_risk = churn_risk_customers(df)
    print(f"\n== Churn-Risk Customers (no order in 6+ months): {len(at_risk)} customers ==")

    model, metrics, forecast_df, monthly = build_forecast_model(df)
    print("\n== Forecast Model Performance (holdout set) ==")
    for k, v in metrics.items():
        print(f"  {k}: {v:,.2f}")

    print("\n== Next 3 Months Sales Forecast ==")
    print(forecast_df.to_string(index=False))


# ---------------------------------------------------------------------------
# 4b. DASHBOARD MODE (Streamlit front-end)
# ---------------------------------------------------------------------------
def run_dashboard():
    import streamlit as st
    import plotly.express as px

    st.set_page_config(page_title="E-Commerce Sales Analytics", layout="wide")
    st.title("📊 E-Commerce Sales Analytics & Forecasting")
    st.caption("Dataset: Sample - Superstore (Kaggle) | 9,994 orders, 2014-2017")

    df = load_and_clean_data()
    kpis = compute_kpis(df)

    # ---- Page selector (keeps each audience finding their answer fast) ----
    page = st.sidebar.radio(
        "Go to",
        ["Executive Overview", "Sales & Product Analysis", "Customer & Risk Analysis", "Forecast"],
    )

    if page == "Executive Overview":
        st.header("Executive Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Revenue", f"${kpis['Total Revenue']:,.0f}")
        c2.metric("Total Profit", f"${kpis['Total Profit']:,.0f}")
        c3.metric("Total Orders", f"{kpis['Total Orders']:,}")
        c4.metric("Avg Order Value", f"${kpis['Average Order Value']:,.2f}")

        monthly = df.groupby("Order Month")["Sales"].sum().reset_index()
        fig = px.line(monthly, x="Order Month", y="Sales", title="Monthly Revenue Trend")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Fact → Insight → Opportunity → Action")
        st.markdown(
            """
            - **Fact:** Revenue trends and category mix are shown above.
            - **Insight:** Technology and top regions drive a disproportionate share of profit.
            - **Opportunity:** Reallocate marketing spend toward the highest-margin categories/regions.
            - **Risk:** A subset of customers show high churn risk (see Customer & Risk Analysis).
            - **Action:** Launch a targeted retention offer for at-risk, high-value customers.
            """
        )

    elif page == "Sales & Product Analysis":
        st.header("Sales & Product Analysis")
        col1, col2 = st.columns(2)
        with col1:
            cat = sales_by_category(df)
            st.plotly_chart(px.bar(cat, x="Category", y="Sales", title="Sales by Category"), use_container_width=True)
        with col2:
            reg = sales_by_region(df)
            st.plotly_chart(px.bar(reg, x="Region", y="Sales", title="Sales by Region"), use_container_width=True)

        st.subheader("Top Sub-Categories by Sales")
        sub = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).reset_index()
        st.plotly_chart(px.bar(sub, x="Sub-Category", y="Sales"), use_container_width=True)

    elif page == "Customer & Risk Analysis":
        st.header("Customer & Risk Analysis")
        st.subheader("Top 10 Customers by Sales")
        st.dataframe(top_customers(df), use_container_width=True)

        st.subheader("Churn-Risk Customers")
        at_risk = churn_risk_customers(df)
        st.write(f"**{len(at_risk)} customers** have not ordered in 6+ months.")
        st.dataframe(at_risk.head(20), use_container_width=True)

    elif page == "Forecast":
        st.header("Sales Forecast (Linear Regression)")
        model, metrics, forecast_df, monthly = build_forecast_model(df)

        if metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{metrics['MAE']:,.0f}")
            c2.metric("RMSE", f"{metrics['RMSE']:,.0f}")
            c3.metric("R²", f"{metrics['R2']:.2f}")

        combined = pd.concat(
            [
                monthly[["Order Month", "Sales"]].rename(columns={"Order Month": "Month", "Sales": "Actual"}),
                forecast_df.rename(columns={"Forecasted Sales": "Forecast"}),
            ]
        )
        st.plotly_chart(
            px.line(combined, x="Month", y=["Actual", "Forecast"], title="Actual vs Forecasted Monthly Sales"),
            use_container_width=True,
        )
        st.dataframe(forecast_df, use_container_width=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run analysis in console instead of the dashboard")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        try:
            run_dashboard()
        except Exception as e:
            print("Could not launch Streamlit dashboard (are you running `streamlit run`?).")
            print("Falling back to CLI mode.\n")
            run_cli()
