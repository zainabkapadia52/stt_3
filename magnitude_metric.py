import os
import subprocess
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sacrebleu.metrics import BLEU

os.environ["TOKENIZERS_PARALLELISM"] = "false"
repo = "/Users/zainab/optuna" 
dst = "magnitude_metrics.csv"
df = pd.read_csv("structural_metrics.csv")
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = SentenceTransformer("microsoft/codebert-base").to(device)
bleu_metric = BLEU(effective_order=True)


def git_show(commit_hash: str, filepath: str) -> str:
    if not commit_hash or not isinstance(filepath, str) or not filepath:
        return ""
    try:
        res = subprocess.run(
            ["git", "-C", repo, "show", f"{commit_hash}:{filepath}"],
            capture_output=True, text=True, check=False, encoding="utf-8", errors="ignore"
        )
        return res.stdout if res.returncode == 0 else ""
    except Exception:
        return ""

def parent_of(commit_hash: str) -> str | None:
    try:
        res = subprocess.run(
            ["git", "-C", repo, "rev-list", "--parents", "-n", "1", commit_hash],
            capture_output=True, text=True, check=False, encoding="utf-8", errors="ignore"
        )
        parts = res.stdout.strip().split()
        return parts[1] if len(parts) > 1 else None
    except Exception:
        return None

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


semantic, token = [], []
cache = {}

for i, row in df.iterrows():
    h = str(row.get("Hash", "")).strip()
    fp = str(row.get("Filename", "")).strip()

    # fetch current and parent code from git
    key_curr = (h, fp, "curr")
    if key_curr not in cache:
        curr_code = git_show(h, fp)
        cache[key_curr] = curr_code
    else:
        curr_code = cache[key_curr]

    ph = parent_of(h)
    key_prev = (ph or "", fp, "prev")
    if key_prev not in cache:
        prev_code = git_show(ph, fp) if ph else ""
        cache[key_prev] = prev_code
    else:
        prev_code = cache[key_prev]

    prev_code = clean_text(prev_code)
    curr_code = clean_text(curr_code)

    # semantic similarity
    try:
        emb = model.encode([prev_code, curr_code], convert_to_tensor=True,
                           normalize_embeddings=True, device=device)
        cos_sim = util.pytorch_cos_sim(emb[0], emb[1]).item()
    except Exception:
        cos_sim = None
    semantic.append(cos_sim)

    #token similarity (BLEU)
    try:
        bleu = bleu_metric.sentence_score(curr_code, [prev_code]).score / 100.0
    except Exception:
        bleu = None
    token.append(bleu)

df["Semantic_Similarity"] = semantic
df["Token_Similarity"] = token
df.to_csv(dst, index=False)
print(f"Added Semantic_Similarity and Token_Similarity")
