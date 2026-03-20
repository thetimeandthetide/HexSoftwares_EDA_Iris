# ============================================================
#  PROJECT 1 — Exploratory Data Analysis (EDA) on Iris Dataset
#  Hex Softwares Data Science Internship
#  Author: Joseph Fidel KM
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# ── 0. Style ────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({'figure.dpi': 120, 'font.family': 'DejaVu Sans'})

# ── 1. Load Dataset ─────────────────────────────────────────
print("=" * 55)
print("  PROJECT 1 — EDA on Iris Dataset")
print("  Hex Softwares Internship | Joseph Fidel KM")
print("=" * 55)

iris_raw = load_iris()
df = pd.DataFrame(iris_raw.data, columns=iris_raw.feature_names)
df['species'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)

print("\n✅ Dataset Loaded Successfully")
print(f"   Shape : {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Basic Info ────────────────────────────────────────────
print("\n── First 5 Rows ──────────────────────────────────────")
print(df.head().to_string())

print("\n── Data Types & Non-Null Counts ──────────────────────")
print(df.info())

print("\n── Missing Values ────────────────────────────────────")
missing = df.isnull().sum()
print(missing)
print("✅ No missing values found!" if missing.sum() == 0 else f"⚠️  {missing.sum()} missing values detected")

print("\n── Statistical Summary ───────────────────────────────")
print(df.describe().round(2).to_string())

print("\n── Species Distribution ──────────────────────────────")
print(df['species'].value_counts().to_string())

# ── 3. Figure 1: Overview Dashboard ─────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("EDA — Iris Dataset\nHex Softwares Internship  |  Joseph Fidel KM",
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

features = ['sepal length (cm)', 'sepal width (cm)',
            'petal length (cm)', 'petal width (cm)']
colors = sns.color_palette("Set2", 3)
species_list = df['species'].cat.categories.tolist()

# Row 1 — Histograms
for i, feat in enumerate(features[:3]):
    ax = fig.add_subplot(gs[0, i])
    for sp, col in zip(species_list, colors):
        subset = df[df['species'] == sp][feat]
        ax.hist(subset, bins=12, alpha=0.6, color=col, label=sp, edgecolor='white')
    ax.set_title(feat.replace(' (cm)', '').title(), fontsize=10, fontweight='bold')
    ax.set_xlabel("cm", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7)

# Row 2 — Box Plots
for i, feat in enumerate(features[:3]):
    ax = fig.add_subplot(gs[1, i])
    data_to_plot = [df[df['species'] == sp][feat].values for sp in species_list]
    bp = ax.boxplot(data_to_plot, patch_artist=True, notch=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_xticklabels(species_list, fontsize=8)
    ax.set_title(f"{feat.replace(' (cm)', '').title()} — Box Plot",
                 fontsize=10, fontweight='bold')
    ax.set_ylabel("cm", fontsize=8)

# Row 3 — Violin + Scatter
ax_violin = fig.add_subplot(gs[2, 0:2])
df_melt = df.melt(id_vars='species', var_name='Feature', value_name='Value')
sns.violinplot(data=df_melt, x='Feature', y='Value', hue='species',
               palette='Set2', inner='quartile', ax=ax_violin)
ax_violin.set_title("Feature Distribution by Species (Violin)", fontsize=10, fontweight='bold')
ax_violin.set_xticklabels([f.replace(' (cm)', '') for f in features], fontsize=8)
ax_violin.set_xlabel("")
ax_violin.legend(fontsize=8, title='Species')

ax_scatter = fig.add_subplot(gs[2, 2])
for sp, col in zip(species_list, colors):
    sub = df[df['species'] == sp]
    ax_scatter.scatter(sub['petal length (cm)'], sub['petal width (cm)'],
                       c=[col], label=sp, alpha=0.7, edgecolors='white', s=50)
ax_scatter.set_title("Petal Length vs Width", fontsize=10, fontweight='bold')
ax_scatter.set_xlabel("Petal Length (cm)", fontsize=8)
ax_scatter.set_ylabel("Petal Width (cm)", fontsize=8)
ax_scatter.legend(fontsize=7)

plt.savefig('/home/claude/p1_overview.png', bbox_inches='tight')
plt.close()
print("\n✅ Figure 1 saved: p1_overview.png")

# ── 4. Figure 2: Correlation Heatmap ────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("Correlation Analysis — Iris Dataset\nHex Softwares Internship  |  Joseph Fidel KM",
              fontsize=14, fontweight='bold')

corr = df[features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, ax=axes[0], linewidths=0.5,
            cbar_kws={"shrink": 0.8})
axes[0].set_title("Feature Correlation Matrix", fontsize=11, fontweight='bold')
axes[0].set_xticklabels([f.replace(' (cm)', '') for f in features], rotation=30, ha='right', fontsize=9)
axes[0].set_yticklabels([f.replace(' (cm)', '') for f in features], rotation=0, fontsize=9)

# Pairplot-style scatter matrix (manual, 4 features)
pairs = [('sepal length (cm)', 'petal length (cm)'),
         ('sepal length (cm)', 'petal width (cm)')]
for idx, (fx, fy) in enumerate(pairs):
    inset = axes[1].inset_axes([idx * 0.52, 0.1, 0.45, 0.8])
    for sp, col in zip(species_list, colors):
        sub = df[df['species'] == sp]
        inset.scatter(sub[fx], sub[fy], c=[col], label=sp, alpha=0.6, s=30, edgecolors='white')
    inset.set_xlabel(fx.replace(' (cm)', ''), fontsize=7)
    inset.set_ylabel(fy.replace(' (cm)', ''), fontsize=7)
    inset.tick_params(labelsize=6)
    if idx == 0:
        inset.legend(fontsize=6)
axes[1].axis('off')
axes[1].set_title("Key Feature Scatter Pairs", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/p1_correlation.png', bbox_inches='tight')
plt.close()
print("✅ Figure 2 saved: p1_correlation.png")

# ── 5. Key Findings ──────────────────────────────────────────
print("\n── Key Findings ──────────────────────────────────────")
print("1. Petal length & petal width are highly correlated (r ≈ 0.96).")
print("2. Setosa is clearly separable from Versicolor & Virginica.")
print("3. Virginica has the largest petals on average.")
print("4. No missing values — dataset is clean and ready for ML.")
print("\n✅ Project 1 Complete!\n")
