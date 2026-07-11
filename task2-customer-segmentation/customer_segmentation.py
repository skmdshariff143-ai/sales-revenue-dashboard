"""
Customer Segmentation using K-Means Clustering
================================================
This script performs customer segmentation analysis using K-Means clustering
with scikit-learn. It generates synthetic customer data, applies feature scaling,
determines optimal clusters via Elbow Method and Silhouette Analysis, and produces
visualizations and summary reports.

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
matplotlib.use('Agg')  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
RANDOM_SEED = 42
N_CUSTOMERS = 200
K_RANGE = range(2, 9)       # Test K = 2..8
OPTIMAL_K = 4
FEATURES = [
    'AnnualIncome', 'SpendingScore', 'PurchaseFrequency',
    'AvgOrderValue', 'CustomerTenure', 'TotalPurchases'
]
SEGMENT_LABELS = {
    0: 'Premium Loyalists',
    1: 'Budget Shoppers',
    2: 'New High-Value',
    3: 'At-Risk Customers'
}

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
SEGMENT_COLORS = ['#6366f1', '#06b6d4', '#ec4899', '#f59e0b']


def generate_synthetic_data(n_customers=N_CUSTOMERS, seed=RANDOM_SEED):
    """
    Generate realistic synthetic customer data with distinct behavioral patterns
    that naturally form 4 clusters when segmented.

    Returns:
        pd.DataFrame: Customer data with features and metadata.
    """
    np.random.seed(seed)
    print(f"[INFO] Generating synthetic data for {n_customers} customers...")

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

    # Cluster proportions: Premium(28%), Budget(32%), NewHV(18%), AtRisk(22%)
    cluster_sizes = [
        int(n_customers * 0.28),  # 56 Premium Loyalists
        int(n_customers * 0.32),  # 64 Budget Shoppers
        int(n_customers * 0.18),  # 36 New High-Value
    ]
    cluster_sizes.append(n_customers - sum(cluster_sizes))  # Remainder = At-Risk

    records = []
    customer_id = 1

    for cluster_idx, size in enumerate(cluster_sizes):
        for _ in range(size):
            name = f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
            gender = np.random.choice(['M', 'F'])

            if cluster_idx == 0:  # Premium Loyalists
                age = int(np.clip(np.random.normal(42, 8), 28, 65))
                income = int(np.clip(np.random.normal(95000, 15000), 65000, 150000))
                spending = int(np.clip(np.random.normal(82, 10), 60, 100))
                freq = int(np.clip(np.random.normal(28, 6), 15, 45))
                aov = round(np.clip(np.random.normal(185, 35), 110, 300), 2)
                tenure = int(np.clip(np.random.normal(6, 2), 3, 12))
                total = int(np.clip(np.random.normal(320, 70), 150, 500))

            elif cluster_idx == 1:  # Budget Shoppers
                age = int(np.clip(np.random.normal(32, 10), 18, 55))
                income = int(np.clip(np.random.normal(35000, 8000), 18000, 52000))
                spending = int(np.clip(np.random.normal(30, 12), 10, 55))
                freq = int(np.clip(np.random.normal(12, 5), 3, 25))
                aov = round(np.clip(np.random.normal(55, 15), 25, 90), 2)
                tenure = int(np.clip(np.random.normal(3, 1.5), 1, 7))
                total = int(np.clip(np.random.normal(85, 35), 20, 180))

            elif cluster_idx == 2:  # New High-Value
                age = int(np.clip(np.random.normal(28, 5), 20, 40))
                income = int(np.clip(np.random.normal(72000, 12000), 50000, 110000))
                spending = int(np.clip(np.random.normal(78, 12), 55, 98))
                freq = int(np.clip(np.random.normal(18, 5), 8, 30))
                aov = round(np.clip(np.random.normal(145, 30), 85, 230), 2)
                tenure = int(np.clip(np.random.normal(1.2, 0.6), 0, 3))
                total = int(np.clip(np.random.normal(55, 20), 15, 100))

            else:  # At-Risk Customers
                age = int(np.clip(np.random.normal(38, 12), 22, 68))
                income = int(np.clip(np.random.normal(52000, 10000), 30000, 78000))
                spending = int(np.clip(np.random.normal(35, 10), 15, 55))
                freq = int(np.clip(np.random.normal(5, 3), 1, 12))
                aov = round(np.clip(np.random.normal(78, 20), 35, 130), 2)
                tenure = int(np.clip(np.random.normal(4, 2), 1, 9))
                total = int(np.clip(np.random.normal(60, 25), 10, 120))

            records.append({
                'CustomerID': f'CUST-{customer_id:04d}',
                'Name': name,
                'Age': age,
                'Gender': gender,
                'AnnualIncome': income,
                'SpendingScore': spending,
                'PurchaseFrequency': freq,
                'AvgOrderValue': aov,
                'CustomerTenure': tenure,
                'TotalPurchases': total
            })
            customer_id += 1

    df = pd.DataFrame(records)
    print(f"[INFO] Generated {len(df)} customer records with {len(FEATURES)} features.")
    return df


def scale_features(df, features=FEATURES):
    """Apply StandardScaler to the feature columns."""
    print("[INFO] Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return X_scaled, scaler


def find_optimal_k(X_scaled, k_range=K_RANGE):
    """
    Evaluate K-Means for different values of K using inertia (Elbow Method)
    and silhouette score.

    Returns:
        dict: {'inertias': list, 'silhouettes': list, 'optimal_k': int}
    """
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
    """Run K-Means clustering with the specified K."""
    print(f"[INFO] Running K-Means with K={k}...")
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"[INFO] Final silhouette score: {sil:.4f}")
    return km, labels, sil


def apply_pca(X_scaled, n_components=2):
    """Reduce dimensionality to 2D for visualization."""
    print("[INFO] Applying PCA (2 components)...")
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    print(f"[INFO] Explained variance: PC1={explained[0]:.3f}, PC2={explained[1]:.3f}, Total={sum(explained):.3f}")
    return X_pca, pca


def plot_pca_segments(X_pca, labels, save_path):
    """Scatter plot of PCA-projected clusters."""
    print("[CHART] Generating PCA segments scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    for i, label_name in SEGMENT_LABELS.items():
        mask = labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=SEGMENT_COLORS[i], label=label_name,
                   alpha=0.7, s=60, edgecolors='white', linewidths=0.3)

    ax.set_xlabel('PCA Component 1', color=TEXT_SECONDARY, fontsize=12)
    ax.set_ylabel('PCA Component 2', color=TEXT_SECONDARY, fontsize=12)
    ax.set_title('Customer Segments — PCA Visualization', color=TEXT_COLOR, fontsize=16, fontweight='bold', pad=16)
    ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, alpha=0.15, color=GRID_COLOR)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print(f"[CHART] Saved: {save_path}")


def plot_elbow_silhouette(k_range, inertias, silhouettes, save_path):
    """Dual-axis plot: inertia (Elbow) + silhouette score."""
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

    # Mark optimal K
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
    ax1.grid(True, alpha=0.15, color=GRID_COLOR)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print(f"[CHART] Saved: {save_path}")


def plot_segment_heatmap(df, features, save_path):
    """Heatmap of cluster centroids (mean feature values by segment)."""
    print("[CHART] Generating segment profile heatmap...")
    profile = df.groupby('SegmentLabel')[features].mean()

    # Normalize for heatmap (0-1 per feature)
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min())

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    sns.heatmap(profile_norm, annot=profile.round(1).values, fmt='', cmap='viridis',
                linewidths=2, linecolor=DARK_BG, ax=ax,
                cbar_kws={'label': 'Normalized Score'},
                annot_kws={'color': TEXT_COLOR, 'fontsize': 10, 'fontweight': 'bold'})

    ax.set_title('Segment Profile Heatmap', color=TEXT_COLOR, fontsize=16, fontweight='bold', pad=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', color=TEXT_SECONDARY, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color=TEXT_COLOR, fontsize=11)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=TEXT_SECONDARY)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_SECONDARY)
    cbar.set_label('Normalized Score', color=TEXT_SECONDARY)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print(f"[CHART] Saved: {save_path}")


def plot_segment_sizes(df, save_path):
    """Bar chart of cluster sizes."""
    print("[CHART] Generating segment sizes bar chart...")
    counts = df.groupby('SegmentLabel').size().reindex(SEGMENT_LABELS.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    bars = ax.bar(counts.index, counts.values, color=SEGMENT_COLORS,
                  edgecolor='white', linewidth=0.5, width=0.6)

    for bar, count in zip(bars, counts.values):
        pct = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{count}\n({pct:.0f}%)', ha='center', va='bottom',
                color=TEXT_COLOR, fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of Customers', color=TEXT_SECONDARY, fontsize=12)
    ax.set_title('Segment Distribution', color=TEXT_COLOR, fontsize=16, fontweight='bold', pad=16)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.set_xticklabels(counts.index, rotation=15, ha='right', color=TEXT_SECONDARY, fontsize=10)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(axis='y', alpha=0.15, color=GRID_COLOR)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print(f"[CHART] Saved: {save_path}")


def save_results(df, save_dir):
    """Save segmented customers and summary profiles to CSV."""
    print("[DATA] Saving output CSVs...")

    customers_path = os.path.join(save_dir, 'segmented_customers.csv')
    df.to_csv(customers_path, index=False)
    print(f"[DATA] Saved: {customers_path} ({len(df)} rows)")

    # Create segment profile summary
    summary = df.groupby(['Segment', 'SegmentLabel']).agg(
        Count=('CustomerID', 'count'),
        AvgAnnualIncome=('AnnualIncome', 'mean'),
        AvgSpendingScore=('SpendingScore', 'mean'),
        AvgPurchaseFrequency=('PurchaseFrequency', 'mean'),
        AvgOrderValue=('AvgOrderValue', 'mean'),
        AvgTenure=('CustomerTenure', 'mean'),
        AvgTotalPurchases=('TotalPurchases', 'mean')
    ).reset_index()

    # Calculate revenue contribution
    df['EstRevenue'] = df['AvgOrderValue'] * df['TotalPurchases']
    total_rev = df['EstRevenue'].sum()
    rev_by_seg = df.groupby('Segment')['EstRevenue'].sum()
    summary['RevenueContribution'] = summary['Segment'].map(
        lambda s: round(rev_by_seg.get(s, 0) / total_rev * 100, 1)
    )

    summary_path = os.path.join(save_dir, 'segment_profile_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"[DATA] Saved: {summary_path} ({len(summary)} rows)")

    return summary


def main():
    """Main pipeline: generate data → scale → cluster → visualize → save."""
    print("=" * 60)
    print("  Customer Segmentation Pipeline")
    print("=" * 60)

    # Ensure output directories exist
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: Generate or load data
    df = generate_synthetic_data()
    print()

    # Step 2: Scale features
    X_scaled, scaler = scale_features(df)
    print()

    # Step 3: Find optimal K
    eval_results = find_optimal_k(X_scaled)
    print()

    # Step 4: Run K-Means with optimal K
    km, labels, sil_score = run_kmeans(X_scaled, k=OPTIMAL_K)
    df['Segment'] = labels
    df['SegmentLabel'] = df['Segment'].map(SEGMENT_LABELS)
    print()

    # Step 5: PCA for visualization
    X_pca, pca = apply_pca(X_scaled)
    print()

    # Step 6: Generate charts
    print("[INFO] Generating visualizations...")
    plot_pca_segments(X_pca, labels, os.path.join(CHARTS_DIR, 'pca_segments.png'))
    plot_elbow_silhouette(K_RANGE, eval_results['inertias'], eval_results['silhouettes'],
                          os.path.join(CHARTS_DIR, 'elbow_silhouette.png'))
    plot_segment_heatmap(df, FEATURES, os.path.join(CHARTS_DIR, 'segment_profile_heatmap.png'))
    plot_segment_sizes(df, os.path.join(CHARTS_DIR, 'segment_sizes.png'))
    print()

    # Step 7: Save results
    summary = save_results(df, DATA_DIR)
    print()

    # Final summary
    print("=" * 60)
    print("  Segmentation Complete!")
    print("=" * 60)
    print(f"\n  Total Customers:     {len(df)}")
    print(f"  Number of Segments:  {OPTIMAL_K}")
    print(f"  Silhouette Score:    {sil_score:.4f}")
    print(f"\n  Segment Breakdown:")
    for _, row in summary.iterrows():
        print(f"    {row['SegmentLabel']:20s} — {int(row['Count']):3d} customers ({row['RevenueContribution']:.1f}% revenue)")
    print(f"\n  Charts saved to:     {CHARTS_DIR}")
    print(f"  Data saved to:       {DATA_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
