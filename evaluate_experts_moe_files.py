"""
evaluate_experts_moe.py
=======================
Full evaluation pipeline for MoE multimodal malware classification.

Flow:
  1. Load aligned embeddings (E_safe, E_mal, E_n2v) + labels from merged.csv
  2. Build three per-expert FAISS indices
  3. Load trained MoE → extract fused embeddings → build MoE FAISS index
  4. Evaluate all four retrievers on the test split using:
       - Precision@k
       - nDCG@k  (SAFE-paper formulation: log2(rank+1) denominator)
       - F1@k    (from Precision@k and Recall@k)
       - Accuracy@k (majority-vote family label)
  5. Print per-expert and MoE metric tables

Requires: numpy, faiss-cpu (or faiss-gpu), torch, pandas, scikit-learn
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import faulthandler
faulthandler.enable()
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIG  — mirror your MoE_Training_dataset_form.py paths
# ============================================================
MASTER_CSV        = Path("/Users/amangolani/MOE_Paper/moepaper/MoE_project_data/merged.csv")
COLUMN_NAMES_TXT  = Path("/Users/amangolani/MOE_Paper/moepaper/MoE_project_data/column_names_ember.txt")

SAFE_DIR_MAL      = Path("/Users/amangolani/MOE_Paper/moepaper/MoE_project_data/embeddings")
SAFE_DIR_BEN      = Path("/Users/amangolani/MOE_Paper/moepaper/MoE_project_data/embeddings_benign")
MALCONV_DIR_MAL   = Path("/Users/amangolani/MOE_Paper/moepaper/MoE_project_data/malware_embeddings")
MALCONV_DIR_BEN   = Path("/Users/amangolani/MOE_Paper/moepaper/MoE_project_data/benignware_embeddings")
N2V_MALWARE_NPY   = Path("/Users/amangolani/MOE_Paper/moepaper/MoE_project_data/embeddings_with_ids.npy")
N2V_BENIGN_NPY    = Path("/Users/amangolani/MOE_Paper/moepaper/MoE_project_data/benign_embeddings_with_ids.npy")
BEN_TABULAR_CSV   = Path("/Users/amangolani/MOE_Paper/moepaper/MoE_project_data/benign_ember_features.csv")
MALWARE_HASHES_TXT= Path("/Users/amangolani/MOE_Paper/moepaper/MoE_project_data/sha_list.txt")
BENIGN_HASHES_CSV = Path("/Users/amangolani/MOE_Paper/moepaper/MoE_project_data/benign_hashes.csv")

MOE_CKPT          = Path("/Users/amangolani/MOE_Paper/moepaper/best_moe.pt")

D_COMMON   = 256
K_VALUES   = [1, 3, 5, 10]       # which @k values to report
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
SEED       = 42

# Which partition to query with.
#   "val"  -> hyperparameter selection (e.g. the expert-dropout sweep).
#   "test" -> final reporting. Run this ONCE, after the configuration is fixed.
# Selecting a configuration by its test score biases the reported number:
# it becomes the maximum over the sweep rather than an unbiased estimate.
EVAL_SPLIT = "val"

COLS_TYPE = [
    "adware", "flooder", "ransomware", "dropper", "spyware",
    "packed", "crypto_miner", "file_infector", "installer",
    "worm", "downloader",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Re-import the loader helpers from your existing code.
# Paste/import them here, or keep this file next to
# MoE_Training_dataset_form.py and do:
#
#   from MoE_Training_dataset_form import (
#       load_sha_list_txt, load_sha_list_csv,
#       load_safe_dir_to_map, load_safe_benign_nested_to_map,
#       load_pooled_folder_to_map, merge_node2vec,
#       align_embeddings, merge_maps, build_filename_to_sha,
#       remap_malconv_map_to_sha, remap_n2v_to_sha,
#       MoEOnEmbeddings,
#   )
#
# For self-containment the critical ones are inlined below.
# ============================================================

def load_sha_list_txt(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]

def load_sha_list_csv(path):
    df = pd.read_csv(path, header=None)
    return df.iloc[:, -1].astype(str).str.strip().str.lower().tolist()

def align_embeddings(global_ids, emb_map, d):
    E    = np.zeros((len(global_ids), d), dtype=np.float32)
    mask = np.zeros(len(global_ids),      dtype=np.float32)
    for i, sid in enumerate(global_ids):
        v = emb_map.get(sid)
        if v is not None:
            E[i]    = v
            mask[i] = 1.0
    return E, mask

def load_pooled_folder_to_map(folder):
    m = {}
    for f in Path(folder).glob("*.npy"):
        v = np.load(f, allow_pickle=False)
        v = np.asarray(v)
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]
        if v.ndim == 1:
            m[f.stem] = v.astype(np.float32)
    return m

def merge_maps(prefer_first, map_a, map_b):
    out = dict(map_b) if prefer_first else dict(map_a)
    out.update(map_a if prefer_first else map_b)
    return out

def load_ids_emb(path):
    X   = np.load(path, allow_pickle=True)
    ids = X[:, 0].astype(str)
    emb = np.vstack([np.asarray(v, dtype=np.float32) for v in X[:, 1]])
    return ids, emb

def merge_node2vec(mal_path, ben_path):
    ids_m, E_m = load_ids_emb(mal_path)
    ids_b, E_b = load_ids_emb(ben_path)
    assert E_m.shape[1] == E_b.shape[1]
    ids_all = np.concatenate([ids_m, ids_b])
    E_all   = np.vstack([E_m, E_b])
    n2v_map = {}
    for sid, vec in zip(ids_all, E_all):
        if sid not in n2v_map:
            n2v_map[sid] = vec.astype(np.float32)
    return n2v_map, E_all.shape[1]


# ============================================================
# MoE model — must match MoE_Training_dataset_form.py exactly
# ============================================================
def l2_normalize(x, eps=1e-12):
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)

class GatingMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, num_experts=3, dropout=0.2, tau=1.0):
        super().__init__()
        self.tau = tau
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )
    def forward(self, x):
        logits = self.net(x)
        logits = torch.tanh(logits / 5.0) * 5.0   # bound logits to [-5, 5]
        p = F.softmax(logits / self.tau, dim=-1)
        return p, logits

class MoEOnEmbeddings(nn.Module):
    def __init__(self, tab_dim_bin, tab_dim_retr, d_mal, d_safe, d_n2v, d_common=256, num_types=11):
        super().__init__()
        self.gate_bin  = GatingMLP(tab_dim_bin,  hidden=256, num_experts=3)
        self.gate_retr = GatingMLP(tab_dim_retr, hidden=256, num_experts=3)
        self.proj_mal  = nn.Linear(d_mal,  d_common)
        self.proj_safe = nn.Linear(d_safe, d_common)
        self.proj_n2v  = nn.Linear(d_n2v,  d_common)
        self.bin_clf   = nn.Linear(d_common, 2)
        self.retrieval_head = nn.Sequential(
            nn.Linear(d_common, d_common), nn.ReLU(),
            nn.Linear(d_common, d_common),
        )

    def forward(self, x_tab_bin, x_tab_retr, e_mal, e_safe, e_n2v, masks=None):
        p_bin,  _ = self.gate_bin(x_tab_bin)
        p_retr, _ = self.gate_retr(x_tab_retr)
        v_mal  = self.proj_mal(e_mal)
        v_safe = self.proj_safe(e_safe)
        v_n2v  = self.proj_n2v(e_n2v)
        if masks is not None:
            v_mal  = v_mal  * masks[:, 0:1]
            v_safe = v_safe * masks[:, 1:2]
            v_n2v  = v_n2v  * masks[:, 2:3]
            p_retr_masked = p_retr * masks
            p_retr = p_retr_masked / p_retr_masked.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        z_retr = p_retr[:,0:1]*v_mal + p_retr[:,1:2]*v_safe + p_retr[:,2:3]*v_n2v
        h = l2_normalize(self.retrieval_head(z_retr))
        return {"p_retr": p_retr, 
                "z_retr": z_retr, 
                "h": h}

# ============================================================
# FAISS helpers
# ============================================================
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build L2-normalised flat index."""
    E = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(E)
    index = faiss.IndexFlatIP(E.shape[1])   # inner product == cosine after L2-norm
    index.add(E)
    return index

def query_faiss(index, query_emb: np.ndarray, k: int):
    """Return (distances, indices) for a single query vector."""
    q = query_emb.astype(np.float32).reshape(1, -1).copy()
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return D[0], I[0]


# ============================================================
# Metric implementations
# ============================================================
def precision_at_k(retrieved_labels, query_label, k):
    """
    Precision@k: fraction of top-k neighbours sharing the query's family label.
    For binary (malware/benign) use the family string; also works for fine-grained
    malware family names if you set query_label to the family string.
    """
    top = retrieved_labels[:k]
    return np.mean(np.array(top) == query_label)


def ndcg_at_k(retrieved_labels, query_label, k):
    """
    nDCG@k as defined in the SAFE paper:
      DCG@k  = sum_{i=1}^{k} rel_i / log2(i + 1)
      IDCG@k = sum_{i=1}^{min(R,k)} 1 / log2(i + 1)
    where rel_i = 1 if retrieved[i] == query_label else 0,
    and R = total number of positives in the retrieved list
    (upper-bounding at k).

    Note: SAFE paper uses log2(rank+1) with 1-indexed ranks, which
    is the standard formulation above.
    """
    top   = retrieved_labels[:k]
    rels  = (np.array(top) == query_label).astype(float)
    gains = [r / np.log2(i + 2) for i, r in enumerate(rels)]   # i+2 because i is 0-indexed
    dcg   = sum(gains)

    # IDCG: best possible ordering (all positives first)
    n_pos = int(rels.sum())
    ideal_gains = [1.0 / np.log2(i + 2) for i in range(min(n_pos, k))]
    idcg  = sum(ideal_gains) if ideal_gains else 0.0

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_labels, query_label, k, total_relevant):
    """
    Recall@k: how many of the total relevant items we retrieved.
    total_relevant = number of samples in the *index* sharing query_label
    (excluding the query itself when query is in the index).
    """
    if total_relevant == 0:
        return 0.0
    top = retrieved_labels[:k]
    hits = np.sum(np.array(top) == query_label)
    return hits / min(total_relevant, k)


def f1_at_k(retrieved_labels, query_label, k, total_relevant):
    """
    F1@k: harmonic mean of Precision@k and Recall@k.
    """
    p = precision_at_k(retrieved_labels, query_label, k)
    r = recall_at_k(retrieved_labels, query_label, k, total_relevant)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def accuracy_at_k(retrieved_labels, k):
    """
    Accuracy@k: majority-vote label equals ground-truth.
    Returns the predicted label (for aggregation) rather than 0/1,
    because the caller compares to the actual query label.
    """
    top = retrieved_labels[:k]
    if len(top) == 0:
        return None
    counts = Counter(top)
    return counts.most_common(1)[0][0]


def evaluate_retriever(
    index: faiss.Index,
    query_embeddings: np.ndarray,   # (N_test, d)
    query_labels: np.ndarray,       # (N_test,) string labels — family or "benign"
    index_labels: np.ndarray,       # (N_index,) labels for every indexed vector
    query_is_in_index: bool = True, # if True, skip self (first hit when query==index)
    k_values=(1, 3, 5, 10),
) -> dict:
    """
    Returns dict: {k: {metric: mean_value}} for all k in k_values.
    """
    max_k = max(k_values)
    fetch_k = max_k + 1 if query_is_in_index else max_k

    results = {k: {"precision": [], "ndcg": [], "f1": [], "accuracy": []}
               for k in k_values}

    for i in range(len(query_embeddings)):
        q_emb   = query_embeddings[i]
        q_label = query_labels[i]

        _, I = query_faiss(index, q_emb, fetch_k)

        # Drop self-retrieval when query is in the index
        if query_is_in_index:
            I = I[I != i][:max_k]

        ret_labels = index_labels[I]

        # Count how many positives exist in the index (excluding self)
        total_pos = int(np.sum(index_labels == q_label))
        if query_is_in_index:
            total_pos = max(0, total_pos - 1)

        for k in k_values:
            results[k]["precision"].append(precision_at_k(ret_labels, q_label, k))
            results[k]["ndcg"].append(ndcg_at_k(ret_labels, q_label, k))
            results[k]["f1"].append(f1_at_k(ret_labels, q_label, k, total_pos))
            pred = accuracy_at_k(ret_labels, k)
            results[k]["accuracy"].append(float(pred == q_label) if pred is not None else 0.0)

    return {
        k: {m: float(np.mean(v)) for m, v in metrics.items()}
        for k, metrics in results.items()
    }


def print_results(name: str, results: dict):
    header = f"\n{'='*60}\n  {name}\n{'='*60}"
    print(header)
    cols = ["k", "Precision@k", "nDCG@k", "F1@k", "Accuracy@k"]
    print(f"{'k':>4}  {'Precision':>10}  {'nDCG':>10}  {'F1':>8}  {'Accuracy':>10}")
    print("-" * 50)
    for k, m in sorted(results.items()):
        print(f"{k:>4}  {m['precision']:>10.4f}  {m['ndcg']:>10.4f}  "
              f"{m['f1']:>8.4f}  {m['accuracy']:>10.4f}")


# ============================================================
# Train/val/test split  (mirrors your existing split logic)
# ============================================================
def load_splits(global_ids, path="/Users/amangolani/MOE_Paper/moepaper/splits.npz"):
    """
    Load the exact partition persisted by the training script.

    The evaluation script must NEVER recompute its own split: the model was
    trained on train_idx and selected on val_idx, so any independently derived
    'test' set would contain samples the model has already seen. The stored
    global_ids act as a fingerprint — a mismatch means the corpus changed since
    training and the checkpoint is not valid for this data.
    """
    d = np.load(path, allow_pickle=True)
    saved_ids = [str(s) for s in d["global_ids"]]
    if saved_ids != [str(s) for s in global_ids]:
        raise RuntimeError(
            f"Corpus mismatch: splits.npz holds {len(saved_ids)} ids, current run has "
            f"{len(global_ids)}. Retrain to regenerate splits.npz before evaluating.")
    train_idx = d["train_idx"]; val_idx = d["val_idx"]; test_idx = d["test_idx"]
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "test/train overlap"
    assert len(np.intersect1d(val_idx,   test_idx)) == 0, "test/val overlap"
    return train_idx, val_idx, test_idx


# ============================================================
# Extract MoE fused embeddings
# ============================================================
@torch.no_grad()
def extract_moe_embeddings(
    model, X_tab_bin, X_tab_retr, E_mal, E_safe, E_n2v, masks,
    batch_size=256, device="cpu",
):
    model.eval()
    model.to(device)
    H = []
    n = X_tab_bin.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        out = model(
            torch.from_numpy(X_tab_bin[start:end]).to(device),
            torch.from_numpy(X_tab_retr[start:end]).to(device),
            torch.from_numpy(E_mal[start:end]).to(device),
            torch.from_numpy(E_safe[start:end]).to(device),
            torch.from_numpy(E_n2v[start:end]).to(device),
            masks=torch.from_numpy(masks[start:end]).to(device),
        )
        H.append(out["h"].cpu().numpy())
    return np.vstack(H)
def load_safe_dir_to_map(emb_dir: Path, pooled_name="binary_embedding.npy"):
    emb_dir = Path(emb_dir)
    m = {}
    missing = 0
    for bd in emb_dir.iterdir():
        if not bd.is_dir():
            continue
        sid = bd.name
        p = bd / pooled_name
        if not p.exists():
            missing += 1
            continue
        v = np.load(p, allow_pickle=False)
        v = np.asarray(v)
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]
        if v.ndim != 1:
            continue
        m[sid] = v.astype(np.float32)
    return m, missing
def resolve_sha_from_pe_name(pe_name: str, filename_to_sha: dict):
    k = pe_name.strip().lower()
    # try exact folder name, then add common extensions
    for cand in (k, k + ".exe", k + ".dll"):
        sha = filename_to_sha.get(cand)
        if sha is not None:
            return sha
    return None
def load_safe_benign_nested_to_map(emb_dir: Path, filename_to_sha: dict, pooled_name="binary_embedding.npy"):
    emb_dir = Path(emb_dir)
    m = {}
    missing = 0
    unmapped = 0
    bad = 0

    for lvl1 in emb_dir.iterdir():
        if not lvl1.is_dir():
            continue

        pe_name = lvl1.name
        pooled_path = lvl1 / pe_name / pooled_name  # <pe>/<pe>/binary_embedding.npy

        if not pooled_path.exists():
            missing += 1
            continue

        sha = resolve_sha_from_pe_name(pe_name, filename_to_sha)
        if sha is None:
            unmapped += 1
            if unmapped <= 10:
                print("[unmapped benign SAFE]", pe_name)
            continue

        try:
            # benign SAFE may be pickled/object npy
            v = np.load(pooled_path, allow_pickle=True)
            v = np.asarray(v)

            # unwrap object arrays
            if v.dtype == object:
                if v.size == 1:
                    v = np.asarray(v.item())
                else:
                    v = np.asarray(list(v), dtype=np.float32)

            if v.ndim == 2 and v.shape[0] == 1:
                v = v[0]

            v = np.asarray(v, dtype=np.float32).reshape(-1)
            if v.ndim != 1 or v.size == 0:
                bad += 1
                continue

            m[sha] = v

        except Exception as e:
            bad += 1
            if bad <= 10:
                print("[bad benign SAFE npy]", pe_name, "->", e)

    return m, missing, unmapped, bad


def build_filename_to_sha():
    ben_df = pd.read_csv(BEN_TABULAR_CSV)

    if "filename" not in ben_df.columns or "sha256" not in ben_df.columns:
        raise RuntimeError("BEN_TABULAR_CSV must contain columns: filename, sha256")

    m = {}
    for fn, sha in zip(ben_df["filename"], ben_df["sha256"]):
        base = Path(str(fn)).name.strip().lower()      # e.g., resmon.exe
        stem = Path(base).stem.strip().lower()         # e.g., resmon
        sha = str(sha)

        m[base] = sha
        m[stem] = sha   # <-- critical for extensionless folder names

    return m
def remap_malconv_map_to_sha(mal_map, filename_to_sha):
    out = {}
    unmapped = 0
    sha_like = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
    for sid, v in mal_map.items():
        name = str(sid).strip().lower()
        if sha_like.match(name):
            out[name] = v
            continue

        base = Path(name).name
        stem = Path(base).stem

        sha = (filename_to_sha.get(base) or filename_to_sha.get(stem) or
               filename_to_sha.get(stem + ".exe") or filename_to_sha.get(stem + ".dll") or
               filename_to_sha.get(base + ".exe") or filename_to_sha.get(base + ".dll"))

        if sha is None:
            unmapped += 1
            continue
        out[str(sha).strip().lower()] = v
    return out, unmapped

def remap_n2v_to_sha(n2v_map: dict, filename_to_sha: dict):
    out = {}
    sha_like = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
    unmapped = 0
    kept_sha = 0

    for k, v in n2v_map.items():
        key = str(k).strip().lower()

        # already sha
        if sha_like.fullmatch(key):
            out[key] = v
            kept_sha += 1
            continue

        # filename-style key -> try basename/stem lookup
        base = Path(key).name.strip().lower()      # aadauthhelper.dll
        stem = Path(base).stem.strip().lower()     # aadauthhelper

        sha = (filename_to_sha.get(base) or filename_to_sha.get(stem) or
               filename_to_sha.get(stem + ".exe") or filename_to_sha.get(stem + ".dll") or
               filename_to_sha.get(base + ".exe") or filename_to_sha.get(base + ".dll"))

        if sha is None:
            unmapped += 1
            continue

        out[str(sha).strip().lower()] = v

    return out, kept_sha, unmapped
# ============================================================
# Main
# ============================================================
def main():
    # ----------------------------------------------------------
    # 1. Build global_ids and load master_df  (same as training)
    # ----------------------------------------------------------

    malware_ids = load_sha_list_txt(MALWARE_HASHES_TXT)
    benign_ids  = load_sha_list_csv(BENIGN_HASHES_CSV)
    global_ids  = list(dict.fromkeys(malware_ids + benign_ids))
    global_ids  = [s.strip().lower() for s in global_ids]
    print(f"Global IDs: {len(global_ids)} (mal={len(malware_ids)}, ben={len(benign_ids)})")

    master_df = pd.read_csv(MASTER_CSV)
    master_df = (master_df
                 .sort_values("is_malware", ascending=False)
                 .drop_duplicates("sha256", keep="first")
                 .set_index("sha256")
                 .reindex(global_ids))
    master_df["sha256"] = master_df.index

    # ---- Drop ids absent from master_tabular.csv -------------------------------
    # MUST match the identical block in the training script. reindex() fabricates
    # all-NaN placeholder rows for ids not present in the CSV; fillna(0) would then
    # label them benign with all-zero EMBER features. Dropping them here keeps the
    # corpus (and therefore the seed-42 permutation) identical to training.
    mal_set   = set(s.strip().lower() for s in malware_ids)
    keep      = master_df["is_malware"].notna().to_numpy()
    n_dropped = int((~keep).sum())
    if n_dropped:
        dropped_ids = [g for g, k in zip(global_ids, keep) if not k]
        dropped_mal = sum(1 for g in dropped_ids if str(g).lower() in mal_set)
        print(f"Dropped {n_dropped} ids absent from master_df "
              f"({dropped_mal} malware, {n_dropped - dropped_mal} benign)")
    master_df  = master_df[keep]
    global_ids = [g for g, k in zip(global_ids, keep) if k]

    n_mal_eff = int((master_df["is_malware"] == 1).sum())
    print(f"Corpus after cleaning: {len(global_ids)} "
          f"(malware: {n_mal_eff} benign: {len(global_ids) - n_mal_eff})")
    assert len(global_ids) == len(master_df), "global_ids and master_df must stay aligned"

    # Binary label for accuracy@k
    y_bin = master_df["is_malware"].fillna(0).astype(int).to_numpy()

    # Fine-grained family label: use malware family if available, else "benign"
    if "family" in master_df.columns:
        family_labels = master_df["family"].fillna("benign").astype(str).to_numpy()
    else:
        # Fallback: just malware / benign
        family_labels = np.where(y_bin == 1, "malware", "benign")

    # ----------------------------------------------------------
    # 2. Load aligned expert embeddings
    # ----------------------------------------------------------
    # Import helpers from your training file (or inline them above)


    filename_to_sha = build_filename_to_sha()

    safe_mal_map, _      = load_safe_dir_to_map(SAFE_DIR_MAL)
    safe_ben_map, *_     = load_safe_benign_nested_to_map(SAFE_DIR_BEN, filename_to_sha)
    safe_map = merge_maps(True, safe_mal_map, safe_ben_map)
    safe_map = {str(k).strip().lower(): v for k, v in safe_map.items()}
    d_safe   = next(iter(safe_map.values())).shape[0]

    mc_mal_raw = load_pooled_folder_to_map(MALCONV_DIR_MAL)
    mc_ben_raw = load_pooled_folder_to_map(MALCONV_DIR_BEN)
    mc_raw     = merge_maps(True, mc_mal_raw, mc_ben_raw)
    mal_map, _ = remap_malconv_map_to_sha(mc_raw, filename_to_sha)
    mal_map    = {str(k).strip().lower(): v for k, v in mal_map.items()}
    d_mal      = next(iter(mal_map.values())).shape[0]

    n2v_map_raw, d_n2v = merge_node2vec(N2V_MALWARE_NPY, N2V_BENIGN_NPY)
    n2v_map, _, _      = remap_n2v_to_sha(n2v_map_raw, filename_to_sha)
    n2v_map            = {str(k).strip().lower(): v for k, v in n2v_map.items()}

    E_safe, m_safe = align_embeddings(global_ids, safe_map, d_safe)
    E_mal,  m_mal  = align_embeddings(global_ids, mal_map,  d_mal)
    E_n2v,  m_n2v  = align_embeddings(global_ids, n2v_map,  d_n2v)
    print(f"SAFE  coverage: {m_safe.mean():.3f}")
    print(f"MalConv coverage: {m_mal.mean():.3f}")
    print(f"N2V   coverage: {m_n2v.mean():.3f}")

    # ----------------------------------------------------------
    # 3. Gate input  (EMBER features + masks)
    # ----------------------------------------------------------
    names     = [l.strip() for l in open(COLUMN_NAMES_TXT)]
    ember_cols = names[1:1+2568]
    Y_type = (master_df[COLS_TYPE].fillna(0).to_numpy() > 0).astype(np.float32)
    for c in ember_cols:
        if c not in master_df.columns:
            master_df[c] = 0.0
    X_tab = master_df[ember_cols].fillna(0).to_numpy(np.float32)
    masks = np.stack([m_mal, m_safe, m_n2v], axis=1).astype(np.float32)
    X_tab = np.concatenate([X_tab, masks], axis=1)
    X_tab_bin=X_tab
    # X_tab_retr=np.concatenate([X_tab,Y_type], axis=1)
    X_tab_retr=X_tab

    # ----------------------------------------------------------
    # 4. Train / val / test split
    # ----------------------------------------------------------
    N = len(global_ids)
    train_idx, val_idx, test_idx = load_splits(global_ids, "/Users/amangolani/MOE_Paper/moepaper/splits.npz")
    print(f"Split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
          f"  (test malware: {int(y_bin[test_idx].sum())}) [loaded from splits.npz]")

    # ----------------------------------------------------------
    # 4b. Which partition are we querying with?
    #
    #   EVAL_SPLIT = "val"  -> hyperparameter selection (dropout sweep).
    #                          Index is TRAIN ONLY, otherwise every val query
    #                          would retrieve itself from the index.
    #   EVAL_SPLIT = "val" -> final reporting. Index is train+val.
    #
    # Sweep on val, commit to a configuration, then run ONCE on test.
    # ----------------------------------------------------------
    if EVAL_SPLIT == "val":
        query_idx = val_idx
        index_idx = train_idx
    elif EVAL_SPLIT == "test":
        query_idx = test_idx
        index_idx = np.concatenate([train_idx, val_idx])
    else:
        raise ValueError(f"EVAL_SPLIT must be 'val' or 'test', got {EVAL_SPLIT!r}")

    assert len(np.intersect1d(query_idx, index_idx)) == 0, "query set overlaps index"
    print(f"Evaluating on '{EVAL_SPLIT}' split — "
          f"index: {len(index_idx)} samples, queries: {len(query_idx)}")

    # ----------------------------------------------------------
    # 5. Per-expert indices are built inside the coverage-aware
    #    evaluation below (see section 7), because each expert is
    #    restricted to the subset of samples where its modality exists.
    # ----------------------------------------------------------
    index_labels = family_labels[index_idx]

    # ----------------------------------------------------------
    # 6. Load MoE → extract fused embeddings → build MoE index
    # ----------------------------------------------------------
    print(f"\nLoading MoE checkpoint from {MOE_CKPT} ...")
    model = MoEOnEmbeddings(
        tab_dim_bin=X_tab_bin.shape[1],   # 2571
        tab_dim_retr=X_tab_retr.shape[1], # 2582
        d_mal=d_mal, d_safe=d_safe, d_n2v=d_n2v,
        d_common=D_COMMON, num_types=len(COLS_TYPE),
    )
    # state = torch.load(MOE_CKPT, map_location="cpu")
    # model.load_state_dict(state)
    # state = torch.load(MOE_CKPT, map_location="cpu")
    # print("Checkpoint keys:", list(state.keys()))
    # print("Model keys:", list(model.state_dict().keys()))
    # print("Step 1: torch.load")
    # state = torch.load(MOE_CKPT, map_location="cpu")
    # print("Step 2: model.state_dict()")
    # model_sd = model.state_dict()
    # print("Step 3: filtering keys")
    # filtered = {k: v for k, v in state.items() if k in model_sd}
    # print("Step 4: copying tensors")
    # with torch.no_grad():
    #     for k, v in filtered.items():
    #         print(f"  copying {k}")
    #         model_sd[k].copy_(v)
    # print("Step 5: done loading")
   
    # print("Loaded successfully")


    state=torch.load(MOE_CKPT, map_location="cpu", weights_only=True)
    filtered={k:v.clone().float() for k, v in state.items()
              if k in dict(model.named_parameters())}
    
    with torch.no_grad():
        for name,param in model.named_parameters():
            if name in filtered:
                param.data=filtered[name]
    print("loaded successfully")

    # state = torch.load(MOE_CKPT, map_location="cpu")
    # model_sd = model.state_dict()
    # for k in state:
    #     if k in model_sd:
    #         if state[k].shape != model_sd[k].shape:
    #             print(f"MISMATCH {k}: ckpt={state[k].shape} model={model_sd[k].shape}")
            
    # print("tab_dim_bin in eval:", X_tab_bin.shape[1])
    # print("tab_dim_retr in eval:", X_tab_retr.shape[1])
    # print("gate_bin input dim in ckpt:", state["gate_bin.net.0.weight"].shape[1])
    # print("gate_retr input dim in ckpt:", state["gate_retr.net.0.weight"].shape[1])
    # missing, unexpected = model.load_state_dict(state, strict=False)
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)
    # print("  Extracting MoE retrieval embeddings for ALL samples...")
    H_all = extract_moe_embeddings(
        model, X_tab_bin, X_tab_retr, E_mal, E_safe, E_n2v,
        masks=masks, batch_size=256, device=DEVICE,
    )
    # (N, D_COMMON)  — L2 normalised output of retrieval_head

    idx_moe = build_faiss_index(H_all[index_idx])
    print("  MoE FAISS index built.")
    faiss.write_index(idx_moe, "faiss_moe.bin")

    # ----------------------------------------------------------
    # 7. Evaluate all four retrievers
    #
    #    COVERAGE-AWARE BASELINES
    #    ------------------------
    #    Samples without an embedding for a given expert carry an all-zero
    #    vector. Under IndexFlatIP over L2-normalised vectors a zero query has
    #    inner product 0.0 with every entry, so FAISS returns tie-broken
    #    (arbitrary) neighbours and the retrieved labels are meaningless.
    #    The availability masks are only consulted inside the MoE forward pass,
    #    so they do NOT protect these raw single-expert baselines.
    #
    #    Each expert is therefore evaluated on the subset where its modality
    #    exists: both its index and its queries are restricted to covered
    #    samples. The MoE is evaluated on the full query partition, since the
    #    gating mechanism handles missing modalities by construction.
    # ----------------------------------------------------------
    print(f"\nRunning evaluation on {EVAL_SPLIT} split...")

    def eval_covered(name, E, m):
        """Build index and query using only samples where this expert exists."""
        ii = index_idx[m[index_idx].astype(bool)]
        qi = query_idx[m[query_idx].astype(bool)]
        if len(ii) == 0 or len(qi) == 0:
            print(f"  [{name}] no covered samples — skipped")
            return None, 0, 0
        index = build_faiss_index(E[ii])
        res = evaluate_retriever(
            index=index,
            query_embeddings=E[qi],
            query_labels=family_labels[qi],
            index_labels=family_labels[ii],
            query_is_in_index=False,
            k_values=K_VALUES,
        )
        return res, len(qi), len(ii)

    results_safe, n_q_safe, n_i_safe = eval_covered("SAFE",    E_safe, m_safe)
    results_mal,  n_q_mal,  n_i_mal  = eval_covered("MalConv", E_mal,  m_mal)
    results_n2v,  n_q_n2v,  n_i_n2v  = eval_covered("Node2Vec", E_n2v, m_n2v)

    # MoE: full query partition — masking is handled inside the model.
    results_moe = evaluate_retriever(
        index=idx_moe,
        query_embeddings=H_all[query_idx],
        query_labels=family_labels[query_idx],
        index_labels=index_labels,
        query_is_in_index=False,
        k_values=K_VALUES,
    )
    n_q_moe = len(query_idx)

    print("\n  Evaluation subsets (queries / index entries):")
    print(f"    SAFE     : {n_q_safe:5d} / {n_i_safe:5d}")
    print(f"    MalConv  : {n_q_mal:5d} / {n_i_mal:5d}")
    print(f"    Node2Vec : {n_q_n2v:5d} / {n_i_n2v:5d}")
    print(f"    MoE      : {n_q_moe:5d} / {len(index_idx):5d}   (full partition)")
    print("    NOTE: single-expert rows are computed on different subsets and")
    print("          are therefore not directly comparable to one another.")

    # ----------------------------------------------------------
    # 8. Print results
    # ----------------------------------------------------------
    print_results(f"Expert 1 — SAFE (function semantics)  [n={n_q_safe}]", results_safe)
    print_results(f"Expert 2 — MalConv (raw bytes)  [n={n_q_mal}]",        results_mal)
    print_results(f"Expert 3 — Node2Vec (graph)  [n={n_q_n2v}]",           results_n2v)
    print_results(f"MoE — Gated soft fusion  [n={n_q_moe}]",               results_moe)


    # ----------------------------------------------------------
    # 9. Summary table: best metric per k across all retrievers
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Summary — nDCG@k comparison")
    print(f"{'='*60}")
    print(f"{'k':>4}  {'SAFE':>8}  {'MalConv':>8}  {'N2V':>8}  {'MoE':>8}")
    print("-" * 46)
    for k in K_VALUES:
        vals = [
            results_safe[k]["ndcg"],
            results_mal[k]["ndcg"],
            results_n2v[k]["ndcg"],
            results_moe[k]["ndcg"],
        ]
        best = max(vals)
        def fmt(v): return f"*{v:.4f}" if v == best else f" {v:.4f}"
        print(f"{k:>4}  " + "  ".join(fmt(v) for v in vals))

    print("\n  (* = best for that k)")


if __name__ == "__main__":
    main()
