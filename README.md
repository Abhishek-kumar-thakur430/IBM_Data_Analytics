# E-Commerce Sales Analytics & Forecasting

An end-to-end analytics project that turns raw e-commerce order data into
business decisions: an executive KPI overview, sales/product/customer
analysis, a churn-risk flag, and a 3-month sales forecast — all in a single
Python file with an interactive dashboard front-end.

## Dataset

**Sample - Superstore** (Kaggle): 9,994 e-commerce orders placed between
2014 and 2017, across the US, with columns for order/ship dates, customer,
region, product category, sales, quantity, discount, and profit.

- Kaggle source: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final
- Local copy used in this project: `data/superstore.csv`

## What the model does

1. **Cleans** the raw data (parses dates, removes duplicates and invalid
   rows, engineers a few analysis-ready fields).
2. **Computes KPIs**: total revenue, profit, orders, customers, average
   order value, and month-over-month growth.
3. **Analyzes** sales and profit by category, region, and sub-category, and
   surfaces the top 10 customers by sales.
4. **Flags churn risk**: customers with no order in the last 6+ months.
5. **Forecasts** the next 3 months of total sales using a Linear Regression
   model trained on lagged monthly sales + month-of-year seasonality
   (time-ordered 80/20 train/test split, evaluated with MAE, RMSE, R²).
6. **Presents results** through a 4-page Streamlit dashboard: Executive
   Overview, Sales & Product Analysis, Customer & Risk Analysis, and
   Forecast — so each audience (exec, sales manager, customer success)
   can find their answer quickly.

## How to run it

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full analysis in the console (quick check, no browser needed)
python Abhishek Thakur_ecommerce_sales_analytics.py --cli

# 3. Or launch the interactive dashboard
streamlit run Abhishek Thakur_ecommerce_sales_analytics.py
```

## Project structure

```
.
├── ecommerce_sales_analytics.py   # single file: cleaning + EDA + model + dashboard
├── requirements.txt                # Python dependencies
├── README.md                       # this file
├── E-Commerce_Sales_Analytics_Report.docx   # written project report
└── data/
    └── superstore.csv              # raw dataset
```

## Key findings (see full report for details)

- Technology is the highest-profit category despite Furniture generating
  comparable revenue — Furniture is heavily discounted, eroding margin.
- The West and East regions drive the largest share of revenue and profit.
- ~200 customers (out of 793) have not ordered in 6+ months and are
  flagged as churn risk.
- The 3-month sales forecast projects a seasonal dip into Q1, consistent
  with the historical pattern in the data.

## Notes on the model

The forecasting model is intentionally simple (Linear Regression on lag
features) so it is transparent and easy to explain to a non-technical
audience. Its R² on the holdout set is modest, which is expected given the
short time series (48 months) — the value of the project is in the KPI
and segmentation analysis as much as in the forecast itself.
