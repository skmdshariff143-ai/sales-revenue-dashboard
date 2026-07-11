# 📊 Data Analytics Portfolio — Sales & Customer Segmentation

A data analytics portfolio featuring two projects: an interactive Sales & Revenue Dashboard and a Customer Segmentation analysis using K-Means clustering. Both include interactive web-based dashboards deployed via GitHub Pages.

## 🌐 Live Demo

**[View Portfolio →](https://skmdshariff143-ai.github.io/sales-revenue-dashboard/)**

| Project | Live Link |
|---------|-----------|
| Task 1 — Sales Dashboard | [Open Dashboard](https://skmdshariff143-ai.github.io/sales-revenue-dashboard/task1-sales-dashboard/sales_dashboard.html) |
| Task 2 — Customer Segmentation | [Open Dashboard](https://skmdshariff143-ai.github.io/sales-revenue-dashboard/task2-customer-segmentation/segmentation_dashboard.html) |

---

## 📁 Project Structure

```
sales-revenue-dashboard/
├── index.html                                  # Landing page linking both tasks
├── README.md                                   # This file
│
├── task1-sales-dashboard/
│   └── sales_dashboard.html                    # Sales & Revenue Analytics Dashboard
│
└── task2-customer-segmentation/
    ├── segmentation_dashboard.html             # Customer Segmentation Dashboard
    ├── customer_segmentation.py                # Python clustering script
    ├── data/
    │   ├── segmented_customers.csv             # 200 customers with segment labels
    │   └── segment_profile_summary.csv         # Aggregated segment profiles
    └── charts/
        ├── pca_segments.png                    # PCA cluster visualization
        ├── elbow_silhouette.png                # Elbow method + silhouette analysis
        ├── segment_profile_heatmap.png         # Feature heatmap by segment
        └── segment_sizes.png                   # Segment distribution bar chart
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
