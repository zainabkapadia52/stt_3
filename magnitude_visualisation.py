import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dirp = "./plots_d_magnitude"
os.makedirs(dirp, exist_ok=True)

SEM_THR = 0.995   # CodeBERT (Semantic_Similarity)
TOK_THR = 0.75    # BLEU (Token_Similarity)

df= pd.read_csv("magnitude_metrics.csv")
df= df.copy()

x = pd.to_numeric(df["Semantic_Similarity"], errors="coerce")
y = pd.to_numeric(df["Token_Similarity"], errors="coerce")
mask = x.notna() & y.notna()
x = x[mask].values
y = y[mask].values

if len(x) >= 2:
    r = np.corrcoef(x, y)[0, 1]
else:
    r = np.nan

print(f"Pearson r (Semantic vs Token) = {r:.4f}  (n = {len(x)})")


q_minor_minor = int(((x >= SEM_THR) & (y >= TOK_THR)).sum())
q_minor_major = int(((x >= SEM_THR) & (y <  TOK_THR)).sum())   
q_major_minor = int(((x <  SEM_THR) & (y >= TOK_THR)).sum())   
q_major_major = int(((x <  SEM_THR) & (y <  TOK_THR)).sum())
print("\nQuadrant counts:")
print(f"  Minor/Minor : {q_minor_minor}")
print(f"  Minor/Major : {q_minor_major}")
print(f"  Major/Minor : {q_major_minor}")
print(f"  Major/Major : {q_major_major}")


plt.figure(figsize=(7.2, 6))
plt.scatter(x, y, s=10, alpha=0.25, edgecolors="none")
plt.axvline(SEM_THR, linestyle="--", linewidth=1.5)
plt.axhline(TOK_THR, linestyle="--", linewidth=1.5)
plt.xlabel("Semantic_Similarity (CodeBERT cosine)")
plt.ylabel("Token_Similarity (BLEU)")
plt.title("Semantic vs. Token Similarity")
plt.xlim(0.70, 1.00)
plt.ylim(0.50, 1.00)
plt.text(0.705, 0.955, f"Pearson r = {r:.3f}", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7"))

plt.tight_layout()
out_path = os.path.join(dirp, "semantic_vs_token_scatter.png")
plt.savefig(out_path, dpi=200)
plt.close()
print(f"\nSaved scatter → {out_path}")
