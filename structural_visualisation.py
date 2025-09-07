import os
import matplotlib.pyplot as plt
import pandas as pd

plot_dir = "./plots_c_structural"
os.makedirs(plot_dir, exist_ok=True)

df= pd.read_csv("structural_metrics.csv")
plt.figure(figsize=(9,6))
data = [
    df['MI_Before'].dropna(), df['MI_After'].dropna(),
    df['CC_Before'].dropna(), df['CC_After'].dropna(),
    df['LOC_Before'].dropna(), df['LOC_After'].dropna()
]
labels = ['MI (Before)', 'MI (After)', 'CC (Before)', 'CC (After)', 'LOC (Before)', 'LOC (After)']
plt.boxplot(data, showfliers=False)
plt.xticks(range(1, len(labels)+1), labels, rotation=20, ha='right')
plt.ylabel("Value")
plt.title("Structural Metrics —Before vs After")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "structural_before_after.png"), dpi=200)
plt.close()

# Change Distributions Histograms
changes = [
    ('MI_Change',  'ΔMI (After − Before)'),
    ('CC_Change',  'ΔCC (After − Before)'),
    ('LOC_Change', 'ΔLOC (After − Before)'),
]
for col, title in changes:
    series = df[col].dropna()
    plt.figure(figsize=(8,5))
    plt.hist(series, bins=30, edgecolor='black')
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2, label='0 (no change)')
    plt.title(f"{title} — Distribution")
    plt.xlabel(title)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{col.lower()}_hist.png"), dpi=200)
    plt.close()

x = df['LOC_Change'].astype(float)
y = df['CC_Change'].astype(float)
mask = ~(x.isna() | y.isna())
xv, yv = x[mask], y[mask]

plt.figure(figsize=(7,6))
plt.scatter(xv, yv, s=16, alpha=0.6)
plt.axhline(0, color='gray', linestyle='dashed', linewidth=1)
plt.axvline(0, color='gray', linestyle='dashed', linewidth=1)
plt.title("ΔLOC vs ΔCC (After − Before)")
plt.xlabel("ΔLOC")
plt.ylabel("ΔCC")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "delta_loc_vs_delta.png"), dpi=200)
plt.close()


def helper(name, s):
    s = s.dropna()
    if len(s) == 0:
        return f"{name}: no data"
    return (f"{name}: n={len(s)}, mean={s.mean():.2f}, median={s.median():.2f}, "
            f"min={s.min():.2f}, max={s.max():.2f}")

print(helper("ΔMI",  df["MI_Change"]))
print(helper("ΔCC",  df["CC_Change"]))
print(helper("ΔLOC", df["LOC_Change"]))
