import pandas as pd
import numpy as np
import subprocess
import os
from radon.metrics import mi_visit
from radon.complexity import cc_visit
from radon.raw import analyze as analyze_loc

repo = "/Users/zainab/optuna" 
df = pd.read_csv("diff1.csv")

def git_show(commit_hash: str, filepath: str, parent: bool = False) -> str:
    spec = f"{commit_hash}^:{filepath}" if parent else f"{commit_hash}:{filepath}"
    try:
        res = subprocess.run(
            ["git", "-C", repo, "show", spec],
            capture_output=True, text=True, check=False, encoding="utf-8", errors="ignore"
        )
        return res.stdout if res.returncode == 0 else ""
    except Exception:
        return ""

def run_radon(code: str):
    if not isinstance(code, str) or not code.strip():
        return np.nan, np.nan, np.nan
    # normalize newlines
    code = code.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    try:
        sloc = analyze_loc(code).sloc
        mi = mi_visit(code, multi=True)
        cc_total = sum(b.complexity for b in cc_visit(code))
        return mi, cc_total, sloc
    except Exception:
        return np.nan, np.nan, np.nan


mi_b, cc_b, loc_b = [], [], []
mi_a, cc_a, loc_a = [], [], []
for _, row in df.iterrows():
    h = str(row["Hash"])
    fp = str(row["Filename"])

    before_code = git_show(h, fp, parent=True) 
    after_code  = git_show(h, fp, parent=False) 

    mb, cb, lb = run_radon(before_code)
    ma, ca, la = run_radon(after_code)

    mi_b.append(mb); cc_b.append(cb); loc_b.append(lb)
    mi_a.append(ma); cc_a.append(ca); loc_a.append(la)


df["MI_Before"]  = mi_b
df["CC_Before"]  = cc_b
df["LOC_Before"] = loc_b

df["MI_After"]   = mi_a
df["CC_After"]   = cc_a
df["LOC_After"]  = loc_a

df["MI_Change"]  = df["MI_After"] - df["MI_Before"]
df["CC_Change"]  = df["CC_After"] - df["CC_Before"]
df["LOC_Change"] = df["LOC_After"] - df["LOC_Before"]

metric_cols = ["MI_Before","CC_Before","LOC_Before","MI_After","CC_After","LOC_After"]
cleaned = df.dropna(subset=metric_cols)
cleaned.to_csv("structural_metrics.csv", index=False)
print(f"Structural metrics saved to {"structural_metrics.csv"}")
