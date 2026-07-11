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
│   ├── sales_dashboard.html                    # Sales & Revenue Analytics Dashboard
│   └── data/
│       └── online_retail.csv                   # Real public UK Online Retail dataset (~540k rows)
│
├── task2-customer-segmentation/
│   ├── segmentation_dashboard.html             # Customer Segmentation Dashboard
│   ├── customer_segmentation.py                # Python clustering script
│   ├── data/
│   │   ├── segmented_customers.csv             # 4,338 customers with segment labels
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
│   │   ├── forecast_next_3_months.csv          # Future 3-month forecast values
│   │   └── model_evaluation.csv                # Model comparison metrics
│   └── charts/
│       ├── forecast_overview.png               # Forecast trend line chart
│       ├── residuals.png                       # Residual distribution plot
│       ├── model_comparison.png                # Model performance comparison
│       └── feature_importance.png              # Random Forest feature weights
│
└── task4-data-cleaning-automation/
    ├── cleaning_dashboard.html                 # Data Cleaning Audit Dashboard
    ├── universal_data_cleaner.html             # Reusable Standalone CSV Cleaner Tool
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

An interactive, self-contained sales analytics dashboard built with **HTML**, **CSS**, and **Chart.js** running on the real public **UCI UK Online Retail dataset** (~540,000 transaction rows). It features 6 KPI cards with sparkline micro-charts (Total Revenue, Orders, AOV, Repeat Purchase Rate, New Customers, Net Profit), 5 interactive chart types (line, doughnut, horizontal bar, vertical bar, radar), advanced filtering by time period, region, and product category, and a searchable/sortable recent orders table — all with a premium dark glassmorphism design, smooth entry animations, and full responsiveness. Includes a full-screen dynamic progress loading overlay to parse the 45MB CSV file asynchronously.

### Tech Stack
| Technology | Purpose |
|-----------|---------|
| HTML5 | Structure & Semantic Markup |
| CSS3 | Styling, Glassmorphism, Animations |
| JavaScript | Data Loading, PapaParse Parser, Interactivity |
| [Chart.js 4.x](https://www.chartjs.org/) | Charts & Visualizations |

---

## Task 2 — Customer Segmentation

A machine learning project that performs K-Means customer segmentation using **Python**, **pandas**, and **scikit-learn** on **4,338 unique customers** derived from the real UK Online Retail dataset. The pipeline aggregates transactions to build RFM (Recency, Frequency, Monetary) metrics per customer, applies StandardScaler normalization, determines the optimal cluster count (K=4) via Elbow Method and Silhouette Analysis, and segments customers into 4 groups (*Premium Loyalists*, *Budget Shoppers*, *New High-Value*, and *At-Risk*) dynamically mapped based on centroid metrics. Results are visualized through PCA scatter plots, profile heatmaps, and a companion interactive Chart.js dashboard showing real RFM segments.

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

A machine learning and time-series forecasting project that predicts future monthly sales revenue using **Python**, **pandas**, and **scikit-learn** on the monthly aggregated timeline from the real Online Retail dataset. Because a 12-month historical timeline (Dec 2010 to Nov 2011) is too short for a reliable 6-month prediction, the validation split was adjusted to **2 months** and the forecast window set to **3 months** (Dec 2011 to Feb 2012) for model stability. The pipeline compares a Linear Regression model (Optimal, R²: 0.93) against a Random Forest Regressor using a holdout split. A companion interactive dashboard allows users to adjust future marketing spend parameters and dynamically observe simulated revenue forecast updates.

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

An automated data engineering and reporting pipeline built with **Python**, **pandas**, and **xlsxwriter**. The pipeline implements an 11-step automated data cleaning process that handles missing values, removes duplicates, parses inconsistent date formats, standardizes text fields, and aligns computed fields (Quantity, UnitPrice, and Sales). The system logs all modifications to an audit CSV file, generates a metrics summary JSON file, and automatically creates a formatted Excel report. By resolving all structural anomalies, the data quality score was successfully improved from 97.8 to a perfect 100.0%. A companion interactive dashboard displays the audit log and progress metrics. 

Additionally, this task includes **Universal Reusable CSV Cleaner**, a standalone client-side tool where anyone can drag-and-drop or upload their own CSV file to auto-detect types, remove duplicates, trim spaces, standardize dates, impute nulls, and export a cleaned CSV without any backend requirements.

### Tech Stack
| Technology | Purpose |
|-----------|---------|
| HTML5 / CSS3 | Standalone CSV Cleaner UI |
| JavaScript / PapaParse | Client-side CSV parsing & cleaning logic |
| Python 3 | Pipeline core scripting |
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
