#!/usr/bin/env python3
"""
SBERT.py
SBERT-style sentence embeddings using HuggingFace Transformers + torch

Inputs:
- articles_enriched_updated.parquet
  Required columns: outlet, date, text_noboiler (preferred) OR text (fallback)

Outputs (created next to this script in ./sbert_outputs_balanced/):
- sbert_neighbors_seed{seed}.json
- sbert_overlap_seed{seed}.json
- sbert_coverage_seed{seed}.csv
- sbert_overlap_aggregated.json
- cache/sent_table.parquet
- cache/emb_*.npy  (cached embeddings)

Notes:
- Uses GPU if available (CUDA preferred on Bocconi cluster).
- Sentence table is built once and cached; embeddings are cached per outlet/seed.
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# --- Torch ---
try:
    import torch
except ImportError:
    raise SystemExit(
        "\nERROR: PyTorch not installed.\n"
        "Install (CPU):  pip install -U torch torchvision torchaudio\n"
        "Install (CUDA): pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
    )

from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# PATHS (portable / cluster-friendly)
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # ensures relative paths resolve next to this file

PARQUET_PATH = os.path.join(SCRIPT_DIR, "articles_enriched_updated.parquet")

OUTDIR = os.path.join(SCRIPT_DIR, "sbert_outputs_balanced")
os.makedirs(OUTDIR, exist_ok=True)

CACHE_DIR = os.path.join(OUTDIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

SENT_TABLE_CACHE = os.path.join(CACHE_DIR, "sent_table.parquet")

# =========================
# CONFIG
# =========================
FILTER_POST_OCT7 = True
OCT7 = pd.Timestamp("2023-10-07", tz="UTC")

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+|\n+")

TARGET_WORDS = [
    "gaza", "israel", "hamas", "idf", "palestine",
    "terrorist", "palestinian", "israeli"
]

USE_IDF_EXPANSIONS = True
IDF_REGEX = re.compile(
    r"\b("
    r"idf|i\.?d\.?f\.?"
    r"|israel(?:i)?\s+defen[cs]e\s+forces"
    r"|israeli\s+forces"
    r"|israeli\s+military"
    r"|israeli\s+army"
    r")\b",
    flags=re.IGNORECASE
)

# sentence filtering
MIN_SENT_CHARS = 40
MAX_SENT_CHARS = 500

# embedding
BATCH_SIZE = 64
MAX_TOK_LEN = 256

# FORCE GPU
FORCE_GPU = True

# analysis
TOP_SENTENCES_FOR_KEYWORDS = 300
TOP_SENTENCES_TO_SAVE = 30
TOP_KEYWORDS = 25

# robustness
SEEDS = [1, 2, 3, 4, 5]


# Neighbors for Al Jazeera use the whole Al Jazeera corpus;
# neighbors for CNN use the whole CNN corpus.

USE_WHOLE_CORPUS_PER_OUTLET = True


# =========================
# DEVICE (force GPU)
# =========================
def choose_device(force_gpu: bool = True) -> str:
    # Prefer CUDA on cluster; then MPS (Mac); else CPU
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if force_gpu:
        raise SystemExit(
            "ERROR: FORCE_GPU=True but no GPU backend is available.\n"
            "Set FORCE_GPU=False to allow CPU, or request a GPU node in SLURM."
        )
    return "cpu"

DEVICE = choose_device(FORCE_GPU)


# =========================
# HELPERS
# =========================
def stable_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p and p.strip()]
    out = []
    for p in parts:
        if len(p) < MIN_SENT_CHARS:
            continue
        if len(p) > MAX_SENT_CHARS:
            p = p[:MAX_SENT_CHARS]
        out.append(p)
    return out

def contains_target(sentence: str, target: str) -> bool:
    if target == "idf" and USE_IDF_EXPANSIONS:
        return bool(IDF_REGEX.search(sentence))
    return bool(re.search(r"\b" + re.escape(target) + r"\b", sentence, flags=re.IGNORECASE))

def pick_text_column(df: pd.DataFrame) -> str:
    """
    Prefer text_noboiler if present and not all-missing; fallback to text.
    """
    if "text_noboiler" in df.columns:
        # use it if it has at least some non-null / non-empty content
        series = df["text_noboiler"].astype(str)
        if (series.str.len() > 0).any():
            return "text_noboiler"
    if "text" in df.columns:
        return "text"
    raise ValueError("Neither 'text_noboiler' nor 'text' found in the parquet.")

def build_sentence_table_once(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Build sentence table once and cache it to parquet.
    Includes outlet/date/sent and uses selected text column.
    """
    if os.path.exists(SENT_TABLE_CACHE):
        sent_df = pd.read_parquet(SENT_TABLE_CACHE)
        if {"outlet", "date", "sent"}.issubset(sent_df.columns):
            return sent_df

    rows = []
    texts = df[text_col].astype(str).tolist()
    outlets = df["outlet"].astype(str).str.lower().tolist()
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True)

    for i, txt in tqdm(list(enumerate(texts)), desc="Splitting into sentences"):
        d = dates.iloc[i]
        if pd.isna(d):
            continue
        o = outlets[i]
        sents = split_sentences(txt)
        for s in sents:
            rows.append({"outlet": o, "date": d, "sent": s})

    sent_df = pd.DataFrame(rows).dropna(subset=["date"])
    sent_df.to_parquet(SENT_TABLE_CACHE, index=False)
    return sent_df

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts

def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom

def embed_sentences(sentences: List[str], tokenizer, model, cache_key: str) -> np.ndarray:
    """
    Embeds sentences with mean pooling + L2 normalize.
    Caches embeddings to .npy 
    """
    fname = f"emb_{stable_hash(HF_MODEL + '|' + cache_key)}.npy"
    fpath = os.path.join(CACHE_DIR, fname)

    if os.path.exists(fpath):
        return np.load(fpath)

    model.eval()
    all_embs = []

    # autocast fp16 where possible (faster on GPU)
    use_autocast = DEVICE in ("cuda", "mps")
    # For MPS autocast, float16 is usually OK; for CUDA as well.
    autocast_ctx = (
        torch.autocast(device_type=DEVICE, dtype=torch.float16)
        if use_autocast
        else None
    )

    with torch.inference_mode():
        for i in tqdm(range(0, len(sentences), BATCH_SIZE), desc=f"Embedding ({cache_key})"):
            batch = sentences[i:i + BATCH_SIZE]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_TOK_LEN,
                return_tensors="pt"
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            if autocast_ctx is not None:
                with autocast_ctx:
                    out = model(**enc)
                    emb = mean_pooling(out, enc["attention_mask"])
            else:
                out = model(**enc)
                emb = mean_pooling(out, enc["attention_mask"])

            all_embs.append(emb.detach().to("cpu").numpy())

    embs = np.vstack(all_embs)
    embs = l2_normalize(embs)
    np.save(fpath, embs)
    return embs

def tfidf_top_keywords(texts: List[str], topk: int = TOP_KEYWORDS) -> List[Tuple[str, float]]:
    if not texts:
        return []
    vect = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.9, ngram_range=(1, 1))
    X = vect.fit_transform(texts)
    if X.shape[1] == 0:
        return []
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vect.get_feature_names_out())
    idx = np.argsort(scores)[::-1][:topk]
    return [(str(terms[i]), float(scores[i])) for i in idx]

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

def assert_inputs():
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            f"Parquet not found at: {PARQUET_PATH}\n"
            "Put 'articles_enriched_updated.parquet' in the same folder as SBERT.py."
        )


# =========================
# CORE: per-outlet full-corpus neighbors
# =========================
def run_one_seed_whole_corpus(sent_df_full: pd.DataFrame, seed: int, tokenizer, model) -> Dict:
    """
    Uses ALL sentences for each outlet (no balancing / no sampling).
    The 'seed' is kept only for replicability of outputs naming and (optionally)
    future extensions; it does not change the data subset when whole-corpus mode is enabled.
    """
    out_json: Dict[str, Dict] = {}
    coverage_rows = []

    for outlet, g in sent_df_full.groupby("outlet"):
        g = g.reset_index(drop=True)
        sentences = g["sent"].tolist()

        # cache key: outlet + whole + N (seed included only to keep files distinct if you want)
        cache_key = f"{outlet}_WHOLE_seed{seed}_n{len(sentences)}"
        embs = embed_sentences(sentences, tokenizer, model, cache_key=cache_key)

        outlet_results = {}

        for target in TARGET_WORDS:
            mention_mask = np.array([contains_target(s, target) for s in sentences], dtype=bool)
            n_mention = int(mention_mask.sum())

            coverage_rows.append({
                "outlet": outlet,
                "seed": seed,
                "target": target,
                "n_sentences": len(sentences),
                "n_sentences_with_target": n_mention,
                "pct_sentences_with_target": (n_mention / len(sentences) * 100.0) if len(sentences) else 0.0
            })

            if n_mention == 0:
                outlet_results[target] = {
                    "n_mention_sentences": 0,
                    "top_sentences": [],
                    "keywords": []
                }
                continue

            proto = embs[mention_mask].mean(axis=0, keepdims=True)
            proto = l2_normalize(proto)

            sims = (embs @ proto.T).ravel()
            top_idx = np.argsort(sims)[::-1]

            # keyword window + shown sentences
            top_idx_use = top_idx[:TOP_SENTENCES_FOR_KEYWORDS]
            top_sent_texts = [sentences[i] for i in top_idx_use]

            keywords = tfidf_top_keywords(top_sent_texts, topk=TOP_KEYWORDS)
            top_sentences_to_show = [sentences[i] for i in top_idx[:TOP_SENTENCES_TO_SAVE]]

            outlet_results[target] = {
                "n_mention_sentences": n_mention,
                "top_sentences": top_sentences_to_show,
                "keywords": keywords
            }

        out_json[outlet] = outlet_results

    # write outputs
    neighbors_path = os.path.join(OUTDIR, f"sbert_neighbors_seed{seed}.json")
    with open(neighbors_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print("[OK] wrote", neighbors_path)

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_csv = os.path.join(OUTDIR, f"sbert_coverage_seed{seed}.csv")
    coverage_df.to_csv(coverage_csv, index=False)
    print("[OK] wrote", coverage_csv)

    overlap = {}
    if "aljazeera" in out_json and "cnn" in out_json:
        for target in TARGET_WORDS:
            aj_kw = {w for (w, s) in out_json["aljazeera"].get(target, {}).get("keywords", [])}
            cnn_kw = {w for (w, s) in out_json["cnn"].get(target, {}).get("keywords", [])}
            overlap[target] = {
                "topk": TOP_KEYWORDS,
                "jaccard_overlap": jaccard(aj_kw, cnn_kw),
                "common": sorted(list(aj_kw & cnn_kw)),
                "aljazeera_only": sorted(list(aj_kw - cnn_kw)),
                "cnn_only": sorted(list(cnn_kw - aj_kw)),
            }

    overlap_path = os.path.join(OUTDIR, f"sbert_overlap_seed{seed}.json")
    with open(overlap_path, "w", encoding="utf-8") as f:
        json.dump(overlap, f, ensure_ascii=False, indent=2)
    print("[OK] wrote", overlap_path)

    return overlap


def main():
    assert_inputs()

    # Small banner for SLURM logs
    print("cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("gpu:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    df = pd.read_parquet(PARQUET_PATH).copy()

    for col in ["outlet", "date"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Found: {list(df.columns)}")

    text_col = pick_text_column(df)
    print("Using text column:", text_col)

    df["outlet"] = df["outlet"].astype(str).str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])

    if FILTER_POST_OCT7:
        df = df[df["date"] >= OCT7].copy()

    print("Articles after Oct 7, 2023:\n", df["outlet"].value_counts())
    print("Device:", DEVICE)

    # Build sentence table ONCE (cached)
    sent_df_full = build_sentence_table_once(df, text_col=text_col)
    print("Total sentences:", len(sent_df_full))
    print("Sentences per outlet:\n", sent_df_full["outlet"].value_counts())

    # Load model/tokenizer ONCE
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModel.from_pretrained(HF_MODEL).to(DEVICE)

    all_overlaps = []
    for seed in SEEDS:
        overlap = run_one_seed_whole_corpus(sent_df_full, seed, tokenizer, model)
        overlap["seed"] = seed
        all_overlaps.append(overlap)

    # Aggregate overlap (median jaccard across seeds)
    agg = {}
    for target in TARGET_WORDS:
        vals = []
        for ov in all_overlaps:
            if target in ov:
                vals.append(ov[target]["jaccard_overlap"])
        if vals:
            agg[target] = {
                "median_jaccard_overlap": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n_seeds": len(vals)
            }

    agg_path = os.path.join(OUTDIR, "sbert_overlap_aggregated.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print("\n[OK] wrote", agg_path)
    print("\nDONE ✅ Outputs in:", OUTDIR)


if __name__ == "__main__":
    main()