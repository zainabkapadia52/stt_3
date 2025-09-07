import pandas as pd

df = pd.read_csv("magnitude_metrics.csv")

# thresholds
semantic = 0.995
token = 0.75

# Classification rules
df["Semantic_Class"] = df["Semantic_Similarity"].apply(
    lambda x: "Minor Fix" if pd.notnull(x) and x >= semantic else "Major Fix"
)

df["Token_Class"] = df["Token_Similarity"].apply(
    lambda x: "Minor Fix" if pd.notnull(x) and x >= token else "Major Fix"
)

# agreement
df["Classes_Agree"] = df.apply(
    lambda row: "YES" if row["Semantic_Class"] == row["Token_Class"] else "NO",
    axis=1
)

df.to_csv("classification.csv", index=False)
print("Added Semantic_Class, Token_Class, and Classes_Agree and saved to classification.csv")

print("Counts:")
print("Semantic Minor:", (df["Semantic_Class"] == "Minor Fix").sum())
print("Semantic Major:", (df["Semantic_Class"] == "Major Fix").sum())
print("Token Minor:", (df["Token_Class"] == "Minor Fix").sum())
print("Token Major:", (df["Token_Class"] == "Major Fix").sum())
print("Agreement YES:", (df["Classes_Agree"] == "YES").sum())
print("Agreement NO :", (df["Classes_Agree"] == "NO").sum())

