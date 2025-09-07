import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path

# lab 2 dataset
df = pd.read_csv("diff1.csv")
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

# Total number of commits and files
total_commits = df['Hash'].nunique()
total_files = df['Filename'].nunique()
print(f"Total commits: {total_commits}")
print(f"Total files: {total_files}")

values= [total_commits, total_files]
labels= ['Commits', 'Files']
colors= ['teal', 'coral']  

plt.bar(labels, values, color=colors)
plt.title("No of commits and Modified files count")
plt.ylabel("Count")
for i, v in enumerate(values):
    plt.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "commits_vs_files.png"))
plt.close()

# Average number of modified files per commit
files_per_commit = df.groupby('Hash')['Filename'].nunique()
avg_files_per_commit = files_per_commit.mean()
print(f"Average number of modified files per commit: {avg_files_per_commit:.2f}")

plt.hist(files_per_commit, bins=20, edgecolor='black', color='skyblue')
plt.title("Distribution of Modified Files per Commit")
plt.xlabel("Files Modified per Commit")
plt.ylabel("Frequency")

plt.axvline(avg_files_per_commit, color='red', linestyle='dashed', linewidth=2, label=f"Mean = {avg_files_per_commit:.2f}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "files_per_commit_hist.png"))
plt.close()



# Fix-type
types = ["add", "update", "remove", "fix", "docs", "test"]
tiebreak = ["fix", "remove", "add", "update", "test", "docs"]  # tie-break

# Anchored verbs at start
pattern = {
    "add"   : [r"^(add|introduce|implement)\b"],
    "update": [r"^(update|modify|change|revise|refactor)\b"],
    "remove": [r"^(remove|delete|drop|eliminate)\b"],
    "fix"   : [r"^(fix|resolve|correct)\b", r"\bbug\s*(fix|fixes)?\b"],
    "docs"  : [r"^(document|docs)\b", r"^add (doc|docs|readme|comment|docstring)\b"],
    "test"  : [r"^(test|add test|add tests|write tests)\b"],
}
extra= {
    "docs":  [r"\breadme\.md\b", r"\bdocstring\b", r"\bcomment(s)?\b", r"\bdocs?\b"],
    "test":  [r"\bpytest\b", r"\bunittest\b", r"\bci\b", r"\btest_"],
    "fix":   [r"\bbug\b", r"\bhotfix\b", r"\bissue\b", r"\bregression\b"],
    "remove":[r"\bdeprecated\b", r"\bdead code\b"],
    "update":[r"\brefactor\b", r"\brename\b", r"\bcleanup\b", r"\bmodify\b", r"\bchange\b"],
    "add":   [r"\bintroduc(e|ing)\b", r"\bimplement\b", r"\bnew feature\b"],
}

def classify_inference(text: str) -> str:

    if not isinstance(text, str) or not text.strip():
        return "update" 
    s = text.strip().lower()
    scores = {b: 0 for b in types}
    for bucket, pats in pattern.items():
        for pat in pats:
            if re.search(pat, s):
                scores[bucket] += 2
    for bucket, pats in extra.items():
        for pat in pats:
            if re.search(pat, s):
                scores[bucket] += 1

    if max(scores.values()) == 0:
        if re.search(r"\bfix|resolve|correct\b", s): scores["fix"] += 1
        elif re.search(r"\bremove|delete|drop\b", s): scores["remove"] += 1
        elif re.search(r"\badd|introduce|implement\b", s): scores["add"] += 1
        elif re.search(r"\bupdate|modify|change|refactor\b", s): scores["update"] += 1
        else:
            scores["update"] += 1

    best = max(scores.values())
    tied = [b for b, sc in scores.items() if sc == best]
    for b in tiebreak:
        if b in tied:
            return b
    return "update"


df["Fix_Bucket"] = df["LLM Inference (fix type)"].apply(classify_inference)
counts = df["Fix_Bucket"].value_counts().reindex(types, fill_value=0)
print("Fix type distribution:")
print(counts)
plt.figure(figsize=(7,7))
vals = counts.values
labs = counts.index.tolist()
wedges, _ = plt.pie(vals, labels=None, startangle=140)
centre = plt.Circle((0,0), 0.55, fc='white')
plt.gcf().gca().add_artist(centre)
plt.legend(wedges, [f"{l}: {v}" for l, v in zip(labs, vals)],
           title="Fix Types", loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("Fix Type Distribution")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fix_types.png"), dpi=200)
plt.close()



# File Extensions
def get_ext(path):
    if not isinstance(path, str) or not path:
        return "none"
    name = os.path.basename(path)
    # handle dotfiles like ".gitignore" (treat as no extension)
    if name.startswith(".") and name.count(".") == 1:
        return "none"
    # handle multi-part extensions like .tar.gz
    sfxs = [s.lower() for s in Path(name).suffixes]
    if not sfxs:
        return "none"
    if "".join(sfxs[-2:]) in {".tar.gz", ".tar.bz2", ".tar.xz"}:
        return "".join(sfxs[-2:])  
    return sfxs[-1]  # last suffix like ".py", ".yml"

if "Extension" not in df.columns:
    df["Extension"] = df["Filename"].apply(get_ext)

ext_counts = df["Extension"].value_counts()

plt.figure(figsize=(10,6))
bars = plt.bar(ext_counts.index, ext_counts.values, color="mediumseagreen", edgecolor="black")
plt.title("Modified File Extensions")
plt.xlabel("File Extension")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval),
             ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "all_extensions.png"), dpi=200)
plt.close()

#top 15 modified file names
file_counts = df["Filename"].value_counts()
top_n = 15
top_files = file_counts.head(top_n).sort_values(ascending=True)

plt.figure(figsize=(12, max(6, 0.5 * len(top_files))))
plt.barh(top_files.index, top_files.values, color="cornflowerblue", edgecolor="black")
plt.title(f"Top {top_n} Modified Filenames")
plt.xlabel("Count")
plt.ylabel("Filename")
for y, v in enumerate(top_files.values):
    plt.text(v + max(top_files.values)*0.01, y, str(int(v)),
             va='center', ha='left', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_filenames.png"), dpi=200)
plt.close()

print("All plots saved in:", os.path.abspath(output_dir))



