---
# Generate Data Tech Challenge — Ellie Meltzer

This repository contains everything needed to understand my submission of the tech challenge.
The goal was to analyze international flight data, identify high/low traffic routes, detect geographic growth patterns, and build a simple predictive model for passenger demand.

---

## Process Overview

### 1. Data Preparation
To begin I Loaded the provided Excel dataset (`TechChallenge_Data.xlsx`).
Once the provided excel was loaded I:
- Cleaned the data by:
  - Converting `Month` to a proper datetime.
  - Ensuring passenger counts were numeric.
  - Creating a `Route` identifier (`AustralianPort-ForeignPort`).
  - Cleaned data: https://drive.google.com/file/d/1nQo3JaXSmOULpsn1wza79h98Pl8jN_2N/view?usp=sharing 

### 2. Analysis
**Task 1a. Top/Bottom Routes:**  
  For this task I summed total passengers per route to identify the highest and lowest trafficked routes.  
  File Outputs: `top10_routes.csv`, `bottom10_routes_nonzero.csv`
  High traffic routes indicate markets worth investing in, on the other hand, low traffic routes highlight where resoruces   may be underutilized.

**Task 1b. Growth Patterns:**  
  Calculated year-over-year (YoY) growth at the country level and averaged the last 12 months to highlight fast-growing and declining markets.  
  File Output: `avg_yoy_last12_by_country.csv`
  Found that strong growth in markets like Japan (+23.6% YoY) and Thailand (+34.4% YoY).
  Findings can indentify where additional flights may be warranted versus not need

**Task 1c. Visualizations:**  
  Two visualizations were created using matplotlib.pyplot
  - Top 10 routes bar chart (`top_routes.png`)
    -Bar chart ranking the busiest routes
    -Highlights Sydney-Auckland as the dominant market.
    
  - Top 5 routes monthly passenger trends (`time_series_top5.png`)
    -Line plot showing seasonal patterns and upward trends
    -Illustrates both seasonalility and long-term growth in routes

### 3. Forecasting Model 
- Implemented a seasonal-naive baseline: each forecasted month = same month last year...used as a benchmark
- Evaluated on a 12-month holdout with RMSE, MAE, and MAPE.  
- Also tested a Holt–Winters exponential smoothing model for comparison.  
- File Outputs: `model_summary.json` (and `model_comparison.csv` if Holt–Winters is run).

  all file outputs can be found in the repo:)
---
