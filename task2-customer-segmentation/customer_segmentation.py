"""
Customer Segmentation using K-Means Clustering on Real RFM Data
================================================================
This script performs customer segmentation analysis using K-Means clustering
on RFM (Recency, Frequency, Monetary) metrics calculated from the real
Online Retail transaction dataset.

Author: Shaik Mahammad Shariff
Tech Stack: Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
RANDOM_SEED = 42
K_RANGE = range(2, 9)       # Test K = 2..8
OPTIMAL_K = 4
FEATURES = ['Recency', 'Frequency', 'Monetary']

# Output directories (relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(SCRIPT_DIR, 'charts')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# Chart style configuration
DARK_BG = '#0a0e17'
CARD_BG = '#111827'
TEXT_COLOR = '#f1f5f9'
TEXT_SECONDARY = '#94a3b8'
GRID_COLOR = '#1e293b'
SEGMENT_COLORS = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444']

def load_and_build_rfm():
    """
    Load real transaction data and build RFM summary per CustomerID.
    """
    csv_path = os.path.join(SCRIPT_DIR, '..', 'task1-sales-dashboard', 'data', 'online_retail.csv')
    print(f"[INFO] Loading transaction data from {csv_path}...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Preprocess
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df.dropna(subset=['CustomerID', 'InvoiceDate'])
    
    # Filter out returns and negative quantities/prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # Compute spend per transaction line
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    
    # Max date in dataset
    max_date = df['InvoiceDate'].max()
    print(f"[INFO] Maximum Date in Dataset: {max_date}")
    
    # Aggregate per CustomerID
    print("[INFO] Computing RFM metrics per CustomerID...")
    rfm = df.groupby('CustomerID').agg(
        LastInvoiceDate=('InvoiceDate', 'max'),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('Revenue', 'sum')
    )
    
    # Calculate Recency in days relative to max_date
    rfm['Recency'] = (max_date - rfm['LastInvoiceDate']).dt.days
    
    # Reorder columns
    rfm = rfm.reset_index()[['CustomerID', 'Recency', 'Frequency', 'Monetary']]
    
    # Add synthetic Names/Age/Gender for dashboard visual completeness
    np.random.seed(RANDOM_SEED)
    first_names = [
        'Alice', 'Bob', 'Carol', 'David', 'Eva', 'Frank', 'Grace', 'Henry',
        'Iris', 'Jack', 'Karen', 'Leo', 'Mia', 'Noah', 'Olivia', 'Peter',
        'Quinn', 'Ryan', 'Sara', 'Tom', 'Uma', 'Victor', 'Wendy', 'Xavier',
        'Yara', 'Zach', 'Amber', 'Brian', 'Clara', 'Derek', 'Ella', 'Felix',
        'Gina', 'Hugo', 'Ivy', 'James', 'Kira', 'Liam', 'Maya', 'Nate',
        'Opal', 'Paul', 'Rosa', 'Sean', 'Tina', 'Umar', 'Vera', 'Will'
    ]
    last_names = [
        'Johnson', 'Smith', 'Williams', 'Brown', 'Martinez', 'Lee', 'Kim',
        'Wilson', 'Chen', 'Taylor', 'Davis', 'Anderson', 'Thomas', 'White',
        'Harris', 'Clark', 'Lewis', 'Walker', 'Hall', 'Young', 'Allen',
        'King', 'Wright', 'Scott', 'Green', 'Baker', 'Adams', 'Nelson',
        'Hill', 'Moore', 'Campbell', 'Mitchell', 'Roberts', 'Carter'
    ]
    
    names = []
    ages = []
    genders = []
    
    for _ in range(len(rfm)):
        names.append(f"{np.random.choice(first_names)} {np.random.choice(last_names)}")
        ages.append(int(np.random.randint(18, 70)))
        genders.append(np.random.choice(['M', 'F']))
        
    rfm['Name'] = names
    rfm['Age'] = ages
    rfm['Gender'] = genders
    
    print(f"[INFO] RFM table built: {len(rfm)} unique customers.")
    return rfm

def scale_features(df, features=FEATURES):
    """Apply StandardScaler to the RFM features."""
    print("[INFO] Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return X_scaled, scaler

def find_optimal_k(X_scaled, k_range=K_RANGE):
    """Evaluate K-Means for different values of K."""
    print("[INFO] Evaluating K values from {} to {}...".format(k_range.start, k_range.stop - 1))
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        print(f"  K={k}: Inertia={km.inertia_:.1f}, Silhouette={sil:.4f}")

    optimal_k = list(k_range)[np.argmax(silhouettes)]
    print(f"[INFO] Optimal K by silhouette: {optimal_k} (score={max(silhouettes):.4f})")
    return {'inertias': inertias, 'silhouettes': silhouettes, 'optimal_k': optimal_k}

def run_kmeans(X_scaled, k=OPTIMAL_K):
    """Run K-Means clustering with K=4."""
    print(f"[INFO] Running K-Means with K={k}...")
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"[INFO] Final silhouette score: {sil:.4f}")
    return km, labels, sil

def map_segment_labels(df, labels, k=OPTIMAL_K):
    """
    Dynamically map cluster indices to RFM profile names:
    - Premium Loyalists: Highest Monetary spend.
    - At-Risk: Highest Recency (days inactive) among the remaining.
    - New High-Value: Remaining cluster with higher Monetary spend.
    - Budget Shoppers: Remaining cluster with lower Monetary spend.
    """
    df['Segment'] = labels
    cluster_means = df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
    print("\n[INFO] Cluster Centroids (Raw Means):")
    print(cluster_means)
    
    # 1. Premium Loyalists
    premium_idx = cluster_means['Monetary'].idxmax()
    
    # 2. At-Risk
    remaining_indices = [i for i in range(k) if i != premium_idx]
    at_risk_idx = cluster_means.loc[remaining_indices, 'Recency'].idxmax()
    
    # 3. New High-Value & Budget Shoppers
    last_two = [i for i in remaining_indices if i != at_risk_idx]
    if cluster_means.loc[last_two[0], 'Monetary'] > cluster_means.loc[last_two[1], 'Monetary']:
        new_hv_idx = last_two[0]
        budget_idx = last_two[1]
    else:
        new_hv_idx = last_two[1]
        budget_idx = last_two[0]
        
    segment_labels = {
        premium_idx: 'Premium Loyalists',
        budget_idx: 'Budget Shoppers',
        new_hv_idx: 'New High-Value',
        at_risk_idx: 'At-Risk'
    }
    
    df['SegmentLabel'] = df['Segment'].map(segment_labels)
    return segment_labels

def apply_pca(X_scaled, n_components=2):
    """Reduce dimensions for PCA visual scatter."""
    print("[INFO] Applying PCA (2 components)...")
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca

def plot_pca_segments(X_pca, labels, segment_labels, save_path):
    """Scatter plot of clusters in PCA space."""
    print("[CHART] Generating PCA segments scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    for i in sorted(segment_labels.keys()):
        mask = labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=SEGMENT_COLORS[i], label=segment_labels[i],
                   alpha=0.6, s=40, edgecolors='none')

    ax.set_xlabel('PCA Component 1', color=TEXT_SECONDARY, fontsize=12)
    ax.set_ylabel('PCA Component 2', color=TEXT_SECONDARY, fontsize=12)
    ax.set_title('Customer Segments — PCA 2D Cluster Visual', color=TEXT_COLOR, fontsize=16, fontweight='bold', pad=16)
    ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, alpha=0.1, color=GRID_COLOR)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print(f"[CHART] Saved: {save_path}")

def plot_elbow_silhouette(k_range, inertias, silhouettes, save_path):
    """Inertia and silhouette dual-axis plot."""
    print("[CHART] Generating Elbow + Silhouette plot...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax1.set_facecolor(DARK_BG)

    ks = list(k_range)
    ax1.plot(ks, inertias, 'o-', color=SEGMENT_COLORS[0], linewidth=2, markersize=8, label='Inertia')
    ax1.set_xlabel('Number of Clusters (K)', color=TEXT_SECONDARY, fontsize=12)
    ax1.set_ylabel('Inertia', color=SEGMENT_COLORS[0], fontsize=12)
    ax1.tick_params(axis='y', labelcolor=SEGMENT_COLORS[0])
    ax1.tick_params(colors=TEXT_SECONDARY)

    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, 's-', color=SEGMENT_COLORS[1], linewidth=2, markersize=8, label='Silhouette Score')
    ax2.set_ylabel('Silhouette Score', color=SEGMENT_COLORS[1], fontsize=12)
    ax2.tick_params(axis='y', labelcolor=SEGMENT_COLORS[1])

    optimal_idx = np.argmax(silhouettes)
    ax1.axvline(x=ks[optimal_idx], linestyle='--', color='#f43f5e', alpha=0.7, linewidth=1.5)
    ax1.text(ks[optimal_idx] + 0.15, max(inertias) * 0.95, f'Optimal K={ks[optimal_idx]}',
             color='#f43f5e', fontsize=11, fontweight='bold')

    ax1.set_title('Elbow Method & Silhouette Analysis', color=TEXT_COLOR, fontsize=16, fontweight='bold', pad=16)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=10, loc='center right')

    for spine in ax1.spines.values():
        spine.set_color(GRID_COLOR)
    for spine in ax2.spines.values():
        spine.set_color(GRID_COLOR)
    ax1.grid(True, alpha=0.1, color=GRID_COLOR)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print(f"[CHART] Saved: {save_path}")

def plot_segment_heatmap(df, features, save_path):
    """Heatmap of normalized cluster centroids."""
    print("[CHART] Generating segment profile heatmap...")
    profile = df.groupby('SegmentLabel')[features].mean()

    # Normalize 0-1 per feature
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min())

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    sns.heatmap(profile_norm, annot=profile.round(1).values, fmt='', cmap='viridis',
                linewidths=2, linecolor=DARK_BG, ax=ax,
                cbar_kws={'label': 'Normalized Score'},
                annot_kws={'color': TEXT_COLOR, 'fontsize': 10, 'fontweight': 'bold'})

    ax.set_title('Segment Profile Heatmap (Raw Mean Overlays)', color=TEXT_COLOR, fontsize=16, fontweight='bold', pad=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, color=TEXT_SECONDARY, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color=TEXT_COLOR, fontsize=11)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=TEXT_SECONDARY)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_SECONDARY)
    cbar.set_label('Normalized Score', color=TEXT_SECONDARY)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print(f"[CHART] Saved: {save_path}")

def plot_segment_sizes(df, segment_labels, save_path):
    """Bar chart of customer segment counts."""
    print("[CHART] Generating segment sizes bar chart...")
    labels_order = ['Premium Loyalists', 'Budget Shoppers', 'New High-Value', 'At-Risk']
    counts = df.groupby('SegmentLabel').size().reindex(labels_order).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Color index mapping based on labels
    label_to_index = {v: k for k, v in segment_labels.items()}
    colors_ordered = [SEGMENT_COLORS[label_to_index[name]] for name in labels_order]

    bars = ax.bar(counts.index, counts.values, color=colors_ordered,
                  edgecolor='white', linewidth=0.5, width=0.6)

    for bar, count in zip(bars, counts.values):
        pct = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(counts.values)*0.01),
                f'{int(count)}\n({pct:.1f}%)', ha='center', va='bottom',
                color=TEXT_COLOR, fontsize=11, fontweight='bold')

    ax.set_ylabel('Number of Customers', color=TEXT_SECONDARY, fontsize=12)
    ax.set_title('Segment Size Distribution', color=TEXT_COLOR, fontsize=16, fontweight='bold', pad=16)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.set_xticklabels(counts.index, rotation=0, color=TEXT_SECONDARY, fontsize=10)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(axis='y', alpha=0.1, color=GRID_COLOR)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print(f"[CHART] Saved: {save_path}")

def save_results(df, segment_labels, save_dir):
    """Save segmented customers and profiles to CSV files."""
    print("[DATA] Saving output CSVs...")
    
    # Save raw segmentation table
    customers_path = os.path.join(save_dir, 'segmented_customers.csv')
    df.to_csv(customers_path, index=False)
    print(f"[DATA] Saved: {customers_path} ({len(df)} rows)")
    
    # Aggregate segments profiles
    summary = df.groupby(['Segment', 'SegmentLabel']).agg(
        Count=('CustomerID', 'count'),
        AvgRecency=('Recency', 'mean'),
        AvgFrequency=('Frequency', 'mean'),
        AvgMonetary=('Monetary', 'mean')
    ).reset_index()
    
    total_rev = df['Monetary'].sum()
    rev_by_seg = df.groupby('Segment')['Monetary'].sum()
    summary['RevenueContribution'] = summary['Segment'].map(
        lambda s: round(rev_by_seg.get(s, 0) / total_rev * 100, 1)
    )
    
    summary_path = os.path.join(save_dir, 'segment_profile_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"[DATA] Saved: {summary_path} ({len(summary)} rows)")
    return summary

def main():
    print("=" * 60)
    print("  Customer Segmentation RFM Pipeline")
    print("=" * 60)
    
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Load and build RFM
    df = load_and_build_rfm()
    print()
    
    # 2. Scale features
    X_scaled, scaler = scale_features(df)
    print()
    
    # 3. Find optimal K
    eval_results = find_optimal_k(X_scaled)
    print()
    
    # 4. Run K-Means
    km, labels, sil_score = run_kmeans(X_scaled, k=OPTIMAL_K)
    print()
    
    # 5. Label segments dynamically
    segment_labels = map_segment_labels(df, labels, k=OPTIMAL_K)
    print()
    
    # 6. Apply PCA
    X_scaled_for_pca = X_scaled  # Keep the same
    X_pca, pca = apply_pca(X_scaled_for_pca)
    print()
    
    # 7. Generate charts
    print("[INFO] Generating visualizations...")
    plot_pca_segments(X_pca, labels, segment_labels, os.path.join(CHARTS_DIR, 'pca_segments.png'))
    plot_elbow_silhouette(K_RANGE, eval_results['inertias'], eval_results['silhouettes'],
                          os.path.join(CHARTS_DIR, 'elbow_silhouette.png'))
    plot_segment_heatmap(df, FEATURES, os.path.join(CHARTS_DIR, 'segment_profile_heatmap.png'))
    plot_segment_sizes(df, segment_labels, os.path.join(CHARTS_DIR, 'segment_sizes.png'))
    print()
    
    # 8. Save output datasets
    summary = save_results(df, segment_labels, DATA_DIR)
    print()
    
    print("=" * 60)
    print("  Segmentation pipeline complete!")
    print("=" * 60)
    print(f"  Total Customers:     {len(df)}")
    print(f"  Silhouette Score:    {sil_score:.4f}")
    print(f"\n  Segment Breakdown:")
    for _, row in summary.iterrows():
        print(f"    {row['SegmentLabel']:20s} — {int(row['Count']):4d} customers ({row['RevenueContribution']:.1f}% revenue contribution)")
    print("=" * 60)

if __name__ == '__main__':
    main()
