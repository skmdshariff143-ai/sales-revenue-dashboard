# 📊 Data Analytics Portfolio — Sales, Segmentation, Forecasting & Data Quality Automation

A data analytics portfolio featuring four projects: an interactive Sales & Revenue Dashboard, a Customer Segmentation analysis using K-Means clustering, a Predictive Revenue Forecasting model, and an Automated Data Quality & Cleaning Pipeline. All four include interactive web-based dashboards deployed via GitHub Pages.

## 🌐 Live Demo

**[View Portfolio Hub →](https://skmdshariff143-ai.github.io/sales-revenue-dashboard/)**

| Project | Live Link |
|---------|-----------|
| Task 1 — Sales Dashboard | [Open Dashboard](https://skmdshariff143-ai.github.io/sales-revenue-dashboard/task1-sales-dashboard/sales_dashboard.html) |
| Task 2 — Customer Segmentation | [Open Dashboard](https://skmdshariff143-ai.github.io/sales-revenue-dashboard/task2-customer-segmentation/segmentation_dashboard.html) |
| Task 3 — Predictive Revenue Forecasting | [Open Dashboard](https://skmdshariff143-ai.github.io/sales-revenue-dashboard/task3-predictive-analytics/predictive_dashboard.html) |
| Task 4 — Data Quality & Cleaning Automation | [Open Dashboard](https://skmdshariff143-ai.github.io/sales-revenue-dashboard/task4-data-cleaning-automation/cleaning_dashboard.html) |

---

## 📁 Project Structure

```
sales-revenue-dashboard/
├── index.html                                  # Landing page linking all tasks
├── README.md                                   # This file
│
├── task1-sales-dashboard/
│   └── sales_dashboard.html                    # Sales & Revenue Analytics Dashboard
│
├── task2-customer-segmentation/
│   ├── segmentation_dashboard.html             # Customer Segmentation Dashboard
│   ├── customer_segmentation.py                # Python clustering script
│   ├── data/
│   │   ├── segmented_customers.csv             # 600 customers with segment labels
│   │   └── segment_profile_summary.csv         # Aggregated segment profiles
│   └── charts/
│       ├── pca_segments.png                    # PCA cluster visualization
│       ├── elbow_silhouette.png                # Elbow method + silhouette analysis
│       ├── segment_profile_heatmap.png         # Feature heatmap by segment
│       └── segment_sizes.png                   # Segment distribution bar chart
│
├── task3-predictive-analytics/
│   ├── predictive_dashboard.html               # Predictive Analytics Dashboard
│   ├── predictive_analytics.py                 # Time-series forecasting script
│   ├── data/
│   │   ├── historical_cleaned.csv              # Historical cleaned sales data
│   │   ├── forecast_next_6_months.csv          # Future 6-month forecast values
│   │   └── model_evaluation.csv                # Model comparison metrics
│   └── charts/
│       ├── forecast_overview.png               # Forecast trend line chart
│       ├── residuals.png                       # Residual distribution plot
│       ├── model_comparison.png                # Model performance comparison
│       └── feature_importance.png              # Random Forest feature weights
│
└── task4-data-cleaning-automation/
    ├── cleaning_dashboard.html                 # Data Cleaning Audit Dashboard
    ├── data_cleaning_automation.py             # Data cleaning pipeline script
    ├── build_excel_report.py                   # Excel report generator script
    ├── data_cleaning_report.xlsx               # Automated Excel report
    └── data/
        ├── raw_messy_orders.csv                # Original messy dataset
        ├── cleaned_orders.csv                  # Cleaned output dataset
        ├── cleaning_log.csv                    # Audit cleaning execution log
        └── cleaning_summary.json               # Pipeline metrics summary JSON
```

---

## Task 1 — Sales & Revenue Dashboard

An interactive, self-contained sales analytics dashboard built with **HTML**, **CSS**, and **Chart.js**. It features 6 KPI cards with sparkline micro-charts (Total Revenue, Orders, AOV, Conversion Rate, New Customers, Net Profit), 5 interactive chart types (line, doughnut, horizontal bar, vertical bar, radar), advanced filtering by time period, region, and product category, and a searchable/sortable recent orders table — all with a premium dark glassmorphism design, smooth entry animations, and full responsiveness. No build step required.

### Tech Stack
| Technology | Purpose |
|-----------|---------|
| HTML5 | Structure & Semantic Markup |
| CSS3 | Styling, Glassmorphism, Animations |
| JavaScript | Data Generation, Interactivity |
| [Chart.js 4.x](https://www.chartjs.org/) | Charts & Visualizations |

---

## Task 2 — Customer Segmentation

A machine learning project that performs K-Means customer segmentation on 200 customers using **Python**, **pandas**, and **scikit-learn**. The pipeline generates synthetic customer data with 6 behavioral features (Annual Income, Spending Score, Purchase Frequency, Avg Order Value, Customer Tenure, Total Purchases), applies StandardScaler normalization, determines the optimal cluster count (K=4) via Elbow Method and Silhouette Analysis, and segments customers into 4 distinct groups: *Premium Loyalists*, *Budget Shoppers*, *New High-Value*, and *At-Risk Customers*. Results are visualized through PCA scatter plots, profile heatmaps, and a companion interactive Chart.js dashboard.

### Tech Stack
| Technology | Purpose |
|-----------|---------|
| Python 3 | Core scripting |
| pandas | Data manipulation |
| NumPy | Numerical computing |
| scikit-learn | KMeans, PCA, StandardScaler, Silhouette |
| Matplotlib & Seaborn | Static chart generation |
| HTML/CSS/Chart.js | Interactive dashboard |

---

## Task 3 — Predictive Revenue Forecasting

A machine learning and time-series forecasting project that predicts future monthly sales revenue for the next 6 months using **Python**, **pandas**, and **scikit-learn**. The pipeline engineers lag features, rolling averages, and seasonal indices from historical actuals to train and compare a Linear Regression model against a Random Forest Regressor using a time-based holdout validation split. Performance is evaluated using standard regression metrics (MAE, RMSE, MAPE, and R² Score), and the final selected Random Forest model is deployed recursively to generate predictions and residual distribution charts. A companion interactive dark glassmorphism dashboard allows users to adjust future marketing spend parameters and dynamically observe simulated revenue forecast updates.

### Tech Stack
| Technology | Purpose |
|-----------|---------|
| Python 3 | Core scripting |
| pandas & NumPy | Feature engineering & data manipulation |
| scikit-learn | Linear Regression, RandomForestRegressor, Train/Test split |
| Matplotlib & Seaborn | Static evaluation chart generation |
| HTML/CSS/Chart.js | Interactive dashboard |

---

## Task 4 — Data Cleaning & Reporting Automation

An automated data engineering and reporting pipeline built with **Python**, **pandas**, and **xlsxwriter**. The pipeline implements an 11-step automated data cleaning process that handles missing values, removes duplicates, parses inconsistent date formats, standardizes text fields, and aligns computed fields (Quantity, UnitPrice, and Sales). The system logs all modifications to an audit CSV file, generates a metrics summary JSON file, and automatically creates a highly formatted multi-sheet Excel report complete with native formulas, auto-adjusted column dimensions, custom header styles, and embedded charts. By resolving all structural and content anomalies, the data quality score was successfully improved from 97.8% initially to a perfect 100.0%. A companion interactive dark glassmorphism dashboard allows users to filter, search, and visualize the audit logs and progress metrics in real time.

### Tech Stack
| Technology | Purpose |
|-----------|---------|
| Python 3 | Core scripting |
| pandas & NumPy | Data manipulation & cleaning pipeline |
| xlsxwriter | Excel workbook formatting, styling, formulas, and charts |
| HTML/CSS/Chart.js | Interactive dashboard & log search UI |

---

## 🏃 Getting Started

### View Dashboards (No install needed)
```bash
git clone https://github.com/skmdshariff143-ai/sales-revenue-dashboard.git
cd sales-revenue-dashboard
# Open index.html in your browser
start index.html
```

### Run the Python Segmentation Script
```bash
cd task2-customer-segmentation
pip install pandas numpy scikit-learn matplotlib seaborn
python customer_segmentation.py
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

Built with ❤️ by [Shaik Mahammad Shariff](https://github.com/skmdshariff143-ai)

<!-- Verification pass completed successfully -->
