# moe_train_precomputed.py

import csv
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.utils.data import WeightedRandomSampler
# =========================
# CONFIG 
# =========================
MALWARE_HASHES_TXT = Path("sha_list.txt")          # SOREL malware sha256 list (one per line)
BENIGN_HASHES_CSV  = Path("benign_hashes.csv")     # benign sha256 list (one column or a column named sha256)

MAL_TABULAR_CSV = Path("dataset_emberSorel_merged.csv")     # malware table
BEN_TABULAR_CSV = Path("benign_ember_features.csv")     # benign EMBER table you generated (f0..f2567 + zeros labels)
MASTER_TABULAR_CSV=Path("merged.csv")
SAFE_DIR_MAL = Path("embeddings")                           # embeddings_dir/<sha256>/binary_embedding.npy
SAFE_DIR_BEN=Path("embeddings_benign")

MALCONV_DIR_MAL=Path("malware_embeddings")
MALCONV_DIR_BEN = Path("benignware_embeddings")                    # <sha256>.npy (pooled malconv emb); update if mixed
COLUMN_NAMES_TXT=Path("column_names_ember.txt")
N2V_MALWARE_NPY = Path("embeddings_with_ids.npy")              # object array (N,2): [sha256, emb]
N2V_BENIGN_NPY  = Path("benign_embeddings_with_ids.npy")       # object array (N,2): [sha256, emb]

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
D_COMMON = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# You said these are your type columns (multi-label, non-binary evidence counts)
COLS_TYPE = [
    "adware", "flooder", "ransomware",
    "dropper", "spyware", "packed", "crypto_miner",
    "file_infector", "installer", "worm", "downloader"
]

# Gate input columns:
# For a minimal working run, use EMBER vector columns f0..f2567.
# (You can later replace with a small engineered set like entropy/sections/imports + masks.)
USE_EMBER_F_AS_GATE = True  # True => X_tab = f0..f2567 + masks


# =========================
# Utilities: ID lists
# =========================
def load_sha_list_txt(path: Path):
    ids = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s:
            ids.append(s)
    return ids

# def load_sha_list_csv(path: Path):
#     df = pd.read_csv(path)
#     if "sha256" in df.columns:
#         return df["sha256"].astype(str).tolist()
#     # else: assume first column
#     return df.iloc[:, 0].astype(str).tolist()


# =========================
# Load experts
# =========================
def load_safe_pooled_from_binary(embeddings_dir: Path):
    """
    embeddings_dir/<sha256>/binary_embedding.npy
    """
    embeddings_dir = Path(embeddings_dir)
    ids, embs, missing = [], [], []

    bin_dirs = sorted([p for p in embeddings_dir.iterdir() if p.is_dir()])
    for bd in bin_dirs:
        sid = bd.name
        pooled_path = bd / "binary_embedding.npy"
        if not pooled_path.exists():
            missing.append(sid)
            continue

        v = np.load(pooled_path, allow_pickle=False)
        v = np.asarray(v)
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]
        if v.ndim != 1:
            raise ValueError(f"SAFE pooled embedding must be 1D. Got {v.shape} at {pooled_path}")

        ids.append(sid)
        embs.append(v.astype(np.float32))

    if not embs:
        raise RuntimeError("No SAFE embeddings loaded.")
    return ids, np.stack(embs, axis=0), missing




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

def merge_maps(prefer_first: bool, map_a, map_b):
    # if prefer_first=True, A overrides B; else B overrides A
    out = dict(map_b) if prefer_first else dict(map_a)
    out.update(map_a if prefer_first else map_b)
    return out


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

def resolve_sha_from_pe_name(pe_name: str, filename_to_sha: dict):
    k = pe_name.strip().lower()
    # try exact folder name, then add common extensions
    for cand in (k, k + ".exe", k + ".dll"):
        sha = filename_to_sha.get(cand)
        if sha is not None:
            return sha
    return None

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
def load_malconv_embeddings(folder: Path):
    """
    folder/<sha256>.npy
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No malconv .npy files in {folder}")
    ids, embs = [], []
    for f in files:
        sid = f.stem
        v = np.load(f, allow_pickle=False)
        v = np.asarray(v)
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]
        if v.ndim != 1:
            raise ValueError(f"MalConv pooled embedding must be 1D. Got {v.shape} in {f}")
        ids.append(sid)
        embs.append(v.astype(np.float32))
    return ids, np.stack(embs, axis=0)

def load_pooled_folder_to_map(folder: Path):
    folder = Path(folder)
    m = {}
    for f in folder.glob("*.npy"):
        sid = f.stem
        v = np.load(f, allow_pickle=False)
        v = np.asarray(v)
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]
        if v.ndim != 1:
            continue
        m[sid] = v.astype(np.float32)
    return m

def load_ids_emb(path: Path):
    """
    object array (N,2): col0=id, col1=embedding vector
    """
    X = np.load(path, allow_pickle=True)
    ids = X[:, 0].astype(str)
    emb = np.vstack([np.asarray(v, dtype=np.float32) for v in X[:, 1]])
    return ids, emb

def merge_node2vec(mal_path: Path, ben_path: Path):
    ids_m, E_m = load_ids_emb(mal_path)
    ids_b, E_b = load_ids_emb(ben_path)
    assert E_m.shape[1] == E_b.shape[1], "Node2Vec dim mismatch between malware and benign."

    ids_all = np.concatenate([ids_m, ids_b])
    E_all = np.vstack([E_m, E_b])

    # dedupe: keep first (malware wins because concatenated first)
    n2v_map = {}
    for sid, vec in zip(ids_all, E_all):
        if sid not in n2v_map:
            n2v_map[sid] = vec.astype(np.float32)

    return n2v_map, E_all.shape[1]


# =========================
# Alignment
# =========================
def align_embeddings(global_ids, emb_map, d):
    E = np.zeros((len(global_ids), d), dtype=np.float32)
    mask = np.zeros((len(global_ids),), dtype=np.float32)
    for i, sid in enumerate(global_ids):
        v = emb_map.get(sid)
        if v is not None:
            E[i] = v
            mask[i] = 1.0
    return E, mask


# =========================
# Losses
# =========================
def l2_normalize(x, eps=1e-12):
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)

def load_balance_loss(p, num_experts=3, masks=None):

    target = masks.mean(dim=0)                          # (3,) — per-expert presence rate
    target = target / target.sum().clamp_min(1e-9)      # normalise so sums to 1
    mean_p = p.mean(dim=0)
    return ((mean_p - target) ** 2).sum()

def multilabel_infonce(emb, Y_multi, y_bin,temperature=0.07):
    """
    Multi-label supervised contrastive / InfoNCE:
      emb: (B, D) float tensor
      Y_multi: (B, K) multi-hot / soft [0..1]
    Positives for anchor i: any j where dot(Y_i, Y_j) > 0 (share ≥1 tag).
    Ignores anchors with 0 positives.
    """
    emb = l2_normalize(emb)
    sim = (emb @ emb.T) / temperature  # (B,B)

    B = emb.size(0)
    eye = torch.eye(B, device=emb.device)

    # positives if share at least one label
    # Y_multi can be float; treat >0 as active
    Y_bin = (Y_multi > 0).float()

    benign_mask = (y_bin == 0).float()  # (B,)
    benign_pos  = benign_mask.unsqueeze(1) * benign_mask.unsqueeze(0)  # (B,B)

    # Malware-malware pairs: share at least one family label
    malware_pos = (Y_bin @ Y_bin.T) > 0  # (B,B)
    malware_pos = malware_pos.float()

    # Combine — a pair is positive if both benign, or both malware sharing a family
    pos = ((malware_pos + benign_pos) > 0).float()
    pos = pos * (1.0 - eye)  # remove self

    exp_sim = torch.exp(sim) * (1.0 - eye)
    denom   = exp_sim.sum(dim=1).clamp_min(1e-12)
    num     = (exp_sim * pos).sum(dim=1)

    valid = (pos.sum(dim=1) > 0).float()
    loss  = -torch.log((num / denom).clamp_min(1e-12)) * valid
    return loss.sum() / valid.sum().clamp_min(1.0)
@torch.no_grad()
def infonce_valid_rate(Y_type: torch.Tensor, y_bin: torch.Tensor = None):
    Yb = (Y_type > 0).float()
    pos = (Yb @ Yb.T) > 0

    if y_bin is not None:
        benign_mask = (y_bin == 0).float()
        benign_pos  = (benign_mask.unsqueeze(1) * benign_mask.unsqueeze(0)) > 0
        pos = pos | benign_pos

    eye = torch.eye(pos.size(0), device=pos.device, dtype=torch.bool)
    pos = pos & (~eye)
    valid = pos.sum(dim=1) > 0
    return valid.float().mean().item()
# =========================
# Dataset / Model
# =========================
class PrecomputedExpertsDataset(Dataset):
    def __init__(self, X_tab_bin, X_tab_retr, E_mal, E_safe, E_n2v, y_bin, Y_type, masks):
        self.X_tab_bin = X_tab_bin.astype(np.float32)
        self.X_tab_retr=X_tab_retr.astype(np.float32)
        self.E_mal = E_mal.astype(np.float32)
        self.E_safe = E_safe.astype(np.float32)
        self.E_n2v = E_n2v.astype(np.float32)
        self.y_bin = y_bin.astype(np.int64)
        self.Y_type = Y_type.astype(np.float32)
        self.masks=masks.astype(np.float32)
        n = self.X_tab_bin.shape[0]
        assert self.E_mal.shape[0] == n and self.E_safe.shape[0] == n and self.E_n2v.shape[0] == n
        assert self.y_bin.shape[0] == n and self.Y_type.shape[0] == n

    def __len__(self):
        return self.X_tab_bin.shape[0]

    def __getitem__(self, i):
        return {
            "x_tab_bin":  torch.from_numpy(self.X_tab_bin[i]),
            "x_tab_retr": torch.from_numpy(self.X_tab_retr[i]),
            "e_mal": torch.from_numpy(self.E_mal[i]),
            "e_safe": torch.from_numpy(self.E_safe[i]),
            "e_n2v": torch.from_numpy(self.E_n2v[i]),
            "y_bin": torch.tensor(self.y_bin[i]),
            "Y_type": torch.from_numpy(self.Y_type[i]),
            "masks": torch.from_numpy(self.masks[i]),
        }

class GatingMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, num_experts=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x):
        logits = self.net(x)
        p = F.softmax(logits, dim=-1)
        return p, logits

class MoEOnEmbeddings(nn.Module):
    def __init__(self, tab_dim_bin,tab_dim_retr, d_mal, d_safe, d_n2v, d_common=256, num_types=11):
        super().__init__()
        self.num_experts = 3
        self.gate_bin = GatingMLP(tab_dim_bin, hidden=256, num_experts=3)
        self.gate_retr=GatingMLP(tab_dim_retr, hidden =256, num_experts=3)

        self.proj_mal = nn.Linear(d_mal, d_common)
        self.proj_safe = nn.Linear(d_safe, d_common)
        self.proj_n2v = nn.Linear(d_n2v, d_common)

        self.bin_clf = nn.Linear(d_common, 2)
        self.type_proj = nn.Sequential(
            nn.Linear(d_common, d_common),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.type_head = nn.Linear(d_common, num_types)  # multi-label BCE

        self.retrieval_head = nn.Sequential(
            nn.Linear(d_common, d_common),
            nn.ReLU(),
            nn.Linear(d_common, d_common),
        )

    def forward(self, x_tab_bin,x_tab_retr, e_mal, e_safe, e_n2v, masks=None):
        p_bin, _ = self.gate_bin(x_tab_bin)  # (B,3)
        p_retr,_=self.gate_retr(x_tab_retr)
        v_mal = self.proj_mal(e_mal)
        v_safe = self.proj_safe(e_safe)
        v_n2v = self.proj_n2v(e_n2v)

        if masks is not None:
        # Zero out projected vectors for missing experts (kills bias contribution)
            v_mal  = v_mal  * masks[:, 0:1]
            v_safe = v_safe * masks[:, 1:2]
            v_n2v  = v_n2v  * masks[:, 2:3]

        # Renormalise gate over present experts only
            p_bin_masked = p_bin * masks
            p_bin = p_bin_masked / p_bin_masked.sum(dim=-1, keepdim=True).clamp_min(1e-9)

            p_retr_masked = p_retr * masks
            p_retr = p_retr_masked / p_retr_masked.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        z_bin = p_bin[:, 0:1]*v_mal + p_bin[:, 1:2]*v_safe + p_bin[:, 2:3]*v_n2v
        z_retr=p_retr[:,0:1]*v_mal+p_retr[:,1:2]*v_safe+p_retr[:,2:3]*v_n2v

        bin_logits = self.bin_clf(z_bin)             # (B,2)
        h = l2_normalize(self.retrieval_head(z_retr)) # (B,d)

        return {
            "p_bin":       p_bin,
            "p_retr":      p_retr,
            "z_bin":       z_bin,
            "z_retr":      z_retr,
            "h":           h,
            "bin_logits":  bin_logits,
         }

# def train(model, loader, device=DEVICE,
#           epochs=EPOCHS, lr=LR,
#           w_bin=1.0, w_type=1.0, w_retr=0.5, w_bal=0.05,
#           temperature=0.07,
#           retr_only_on_malware=True):
#     model.to(device)
#     opt = torch.optim.AdamW(model.parameters(), lr=lr)

#     for ep in range(epochs):
#         model.train()
#         total = 0.0

#         for b in loader:
#             x_tab = b["x_tab"].to(device)
#             e_mal = b["e_mal"].to(device)
#             e_safe = b["e_safe"].to(device)
#             e_n2v = b["e_n2v"].to(device)
#             y_bin = b["y_bin"].to(device)
#             Y_type = b["Y_type"].to(device)

#             out = model(x_tab, e_mal, e_safe, e_n2v)

#             loss = 0.0

#             # malware/benign
#             loss_bin = F.cross_entropy(out["bin_logits"], y_bin)
#             loss = loss + w_bin * loss_bin

#             # multi-label type head: use (count>0) as multi-hot
#             loss_type = F.binary_cross_entropy_with_logits(out["type_logits"], Y_type)
#             loss = loss + w_type * loss_type

#             # multi-label InfoNCE retrieval on h, positives share any type label
#             if w_retr > 0:
#                 if retr_only_on_malware:
#                     mask = (y_bin == 1)
#                     if mask.sum() >= 2:
#                         loss_retr = multilabel_infonce(out["h"][mask], Y_type[mask], temperature=temperature)
#                         loss = loss + w_retr * loss_retr
#                 else:
#                     loss_retr = multilabel_infonce(out["h"], Y_type, temperature=temperature)
#                     loss = loss + w_retr * loss_retr

#             loss_bal = load_balance_loss(out["p"], num_experts=3)
#             loss = loss + w_bal * loss_bal

#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             opt.step()

#             total += float(loss.item())

#         print(f"Epoch {ep+1}/{epochs} | loss={total/len(loader):.4f}")
def train_with_val(
    model,
    train_loader,
    val_loader,
    device="cuda",
    epochs=50,
    lr=5e-4,
    w_bin=1.0,
    w_retr_max=1.0,
    w_bal=0.1,
    temperature=0.07,
    retr_only_on_malware=True,
    ckpt_path="best_moe.pt",
    save_by="val_sup",
):
    model.to(device)
    opt = torch.optim.AdamW([
        {"params": model.gate_bin.parameters(),       "lr": lr},
        {"params": model.gate_retr.parameters(),      "lr": lr * 5},
        {"params": model.proj_mal.parameters(),       "lr": lr},
        {"params": model.proj_safe.parameters(),      "lr": lr},
        {"params": model.proj_n2v.parameters(),       "lr": lr},
        {"params": model.bin_clf.parameters(),        "lr": lr},
        {"params": model.retrieval_head.parameters(), "lr": lr},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=8, threshold=1e-4, min_lr=1e-6
    )

    best_val_sup = float("inf")
    
    torch.save(model.state_dict(), ckpt_path)
    for ep in range(epochs):
        model.train()
        total = 0.0
        valid_rates  = []

        w_retr = w_retr_schedule(ep, w_retr_max=w_retr_max, warmup_epochs=15, ramp_epochs=20)

        for b in train_loader:
            x_tab_bin  = b["x_tab_bin"].to(device)
            x_tab_retr = b["x_tab_retr"].to(device)
            e_mal      = b["e_mal"].to(device)
            e_safe     = b["e_safe"].to(device)
            e_n2v      = b["e_n2v"].to(device)
            y_bin      = b["y_bin"].to(device)
            Y_type     = b["Y_type"].to(device)
            b_masks    = b["masks"].to(device)

            out = model(x_tab_bin, x_tab_retr, e_mal, e_safe, e_n2v, masks=b_masks)

            loss_bin = F.cross_entropy(out["bin_logits"], y_bin)
            # balance loss only on gate_retr — gate_bin is allowed to collapse to MalConv
            loss_bal = load_balance_loss(out["p_retr"], num_experts=3, masks=b_masks)
            loss     = w_bin * loss_bin + w_bal * loss_bal

            if w_retr > 0:
                if retr_only_on_malware:
                    mask = (y_bin == 1)
                    
                    if mask.sum() >= 2:
                        valid_rates.append(infonce_valid_rate(Y_type[mask]))
                        loss_retr = multilabel_infonce(out["h"][mask], Y_type[mask], y_bin=y_bin[mask],temperature=temperature)
                        loss = loss + w_retr * loss_retr
                        Y_mal = Y_type[mask]
                        Y_bin_check = (Y_mal > 0).float()
                        pos_matrix = (Y_bin_check @ Y_bin_check.T) > 0
                        pos_matrix.fill_diagonal_(False)
                        print(f"malware in batch: {mask.sum().item()} | "
                             f"positive pairs: {pos_matrix.sum().item() // 2} | "
                            f"family counts: {Y_mal.sum(axis=0).int().tolist()}")
                else:
                    valid_rates.append(infonce_valid_rate(Y_type,y_bin))
                    loss_retr = multilabel_infonce(out["h"], Y_type,y_bin=y_bin, temperature=temperature)
                    loss = loss + w_retr * loss_retr

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item())

        train_loss = total / max(len(train_loader), 1)
        val_loss, val_sup = evaluate(
            model, val_loader, device,
            w_bin=w_bin, w_retr=w_retr, w_bal=w_bal,
            temperature=temperature, retr_only_on_malware=retr_only_on_malware
        )

        scheduler.step(val_loss)

        if val_sup < best_val_sup:
            best_val_sup = val_sup
            torch.save(model.state_dict(), ckpt_path)

        # Gate weight diagnostics — both gates every epoch
        model.eval()
        with torch.no_grad():
            sb  = next(iter(val_loader))
            out_d = model(
                sb["x_tab_bin"].to(device),
                sb["x_tab_retr"].to(device),
                sb["e_mal"].to(device),
                sb["e_safe"].to(device),
                sb["e_n2v"].to(device),
                masks=sb["masks"].to(device)
            )
        pb = out_d["p_bin"].mean(0)
        safe_present = sb["masks"].to(device)[:, 1] > 0
        if safe_present.sum() > 0:
            pr = out_d["p_retr"][safe_present].mean(0)
        else:
            pr = out_d["p_retr"].mean(0)

        vr  = float(np.mean(valid_rates)) if valid_rates else 0.0
        cur_lr = opt.param_groups[0]["lr"]

        print(
            f"Epoch {ep+1}/{epochs} | lr={cur_lr:.2e} | w_retr={w_retr:.3f} | "
            f"infonce_valid={vr:.3f} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_sup={val_sup:.4f} | "
            f"gate_bin  [Mal:{pb[0]:.3f} SAFE:{pb[1]:.3f} N2V:{pb[2]:.3f}] | "
            f"gate_retr [Mal:{pr[0]:.3f} SAFE:{pr[1]:.3f} N2V:{pr[2]:.3f}]"
        )

# def w_retr_schedule(epoch_idx, w_retr_max=0.5, warmup_epochs=2, ramp_epochs=8):
#     """
#     epoch_idx: 0-based
#     - first warmup_epochs: 0
#     - next ramp_epochs: linear 0 -> w_retr_max
#     - after: w_retr_max
#     """
#     if epoch_idx < warmup_epochs:
#         return 0.0
#     t = epoch_idx - warmup_epochs
#     if t >= ramp_epochs:
#         return w_retr_max
#     return w_retr_max * (t / max(ramp_epochs, 1))

def w_retr_schedule(epoch_idx, w_retr_max=1.0, warmup_epochs=15, ramp_epochs=20):
    """
    Slower/safer anneal than before.
    """
    if epoch_idx < warmup_epochs:
        return 0.0
    t = epoch_idx - warmup_epochs
    if t >= ramp_epochs:
        return w_retr_max
    return w_retr_max * (t / max(ramp_epochs, 1))

def load_sha_list_csv(path: Path):
    # Many of these files are written with no header, 2 columns:
    # filename, sha256
    df = pd.read_csv(path, header=None)

    if df.shape[1] == 1:
        # single column of sha256
        return df.iloc[:, 0].astype(str).str.strip().str.lower().tolist()

    # if >=2 columns, assume last column is sha256
    return df.iloc[:, -1].astype(str).str.strip().str.lower().tolist()

# @torch.no_grad()
# def evaluate(model, loader, device, w_bin, w_type, w_retr, w_bal,
#              temperature=0.07, retr_only_on_malware=True, per_class_pos_weight=None):
#     model.eval()
#     total_loss = 0.0
#     total_sup = 0.0  # supervised only (bin + type)
#     all_type_logits = []
#     all_Y_type = []
#     all_y_bin = []
#     if per_class_pos_weight is not None:
#         per_class_pos_weight = per_class_pos_weight.to(device)
#     for b in loader:
#         x_tab_bin = b["x_tab_bin"].to(device)
#         x_tab_retr=b["x_tab_retr"].to(device)
#         e_mal = b["e_mal"].to(device)
#         e_safe = b["e_safe"].to(device)
#         e_n2v = b["e_n2v"].to(device)
#         y_bin = b["y_bin"].to(device)
#         Y_type = b["Y_type"].to(device)
#         b_masks=b["masks"].to(device)
#         out = model(x_tab_bin,x_tab_retr, e_mal, e_safe, e_n2v,masks=b_masks)

#         loss_bin = F.cross_entropy(out["bin_logits"], y_bin)
#         mal_mask_type = (y_bin == 1)
#         if mal_mask_type.sum() > 0:
#             loss_type = F.binary_cross_entropy_with_logits(
#                 out["type_logits"][mal_mask_type],
#                 Y_type[mal_mask_type],
#                 pos_weight=per_class_pos_weight
#             )
#         else:
#             loss_type = torch.tensor(0.0, device=device)

#         sup = w_bin * loss_bin + w_type * loss_type
#         loss = sup

#         if w_retr > 0:
#             if retr_only_on_malware:
#                 mask = (y_bin == 1)
#                 if mask.sum() >= 2:
#                     loss_retr = multilabel_infonce(out["h"][mask], Y_type[mask], temperature=temperature)
#                     loss = loss + w_retr * loss_retr
#             else:
#                 loss_retr = multilabel_infonce(out["h"], Y_type, temperature=temperature)
#                 loss = loss + w_retr * loss_retr

#         loss_bal = load_balance_loss(out["p"], num_experts=3,masks=b_masks)
#         loss = loss + w_bal * loss_bal

#         total_loss += float(loss.item())
#         total_sup += float(sup.item())

#         all_type_logits.append(out["type_logits"].detach().cpu())
#         all_Y_type.append(Y_type.detach().cpu())
#         all_y_bin.append(y_bin.detach().cpu())

#     avg_loss = total_loss / max(len(loader), 1)
#     avg_sup = total_sup / max(len(loader), 1)

#     type_logits = torch.cat(all_type_logits, dim=0)
#     Y_type_all = torch.cat(all_Y_type, dim=0)
#     y_bin_all = torch.cat(all_y_bin, dim=0)

#     f1_all = multilabel_micro_f1(type_logits, Y_type_all, thresh=0.3)

#     mal_mask = (y_bin_all == 1)
#     f1_mal = multilabel_micro_f1(type_logits[mal_mask], Y_type_all[mal_mask], thresh=0.3) if mal_mask.any() else 0.0
    
# # Add this temporarily at the end of evaluate, before return
#     print("type prob stats — mean:", torch.sigmoid(type_logits).mean().item(),
#       "max:", torch.sigmoid(type_logits).max().item(),
#       "% above 0.5:", (torch.sigmoid(type_logits) > 0.5).float().mean().item())
#     # Add at end of evaluate, after computing f1_mal

#     preds = (torch.sigmoid(type_logits[mal_mask]) > 0.3).int()
#     pred_patterns = [tuple(p.tolist()) for p in preds]
#     print("Top 5 prediction patterns:", Counter(pred_patterns).most_common(5))
#     print("Pred positive per class:", preds.float().sum(dim=0).int().tolist())
#     print("True positive per class:", Y_type_all[mal_mask].sum(dim=0).int().tolist())
#     return avg_loss, avg_sup, f1_all, f1_mal

def evaluate(model, loader, device, w_bin, w_retr, w_bal,
             temperature=0.07, retr_only_on_malware=True):
    model.eval()
    total_loss = 0.0
    total_sup  = 0.0

    with torch.no_grad():
        for b in loader:
            x_tab_bin  = b["x_tab_bin"].to(device)
            x_tab_retr = b["x_tab_retr"].to(device)
            e_mal      = b["e_mal"].to(device)
            e_safe     = b["e_safe"].to(device)
            e_n2v      = b["e_n2v"].to(device)
            y_bin      = b["y_bin"].to(device)
            Y_type     = b["Y_type"].to(device)
            b_masks    = b["masks"].to(device)

            out = model(x_tab_bin, x_tab_retr, e_mal, e_safe, e_n2v, masks=b_masks)

            loss_bin = F.cross_entropy(out["bin_logits"], y_bin)
            loss_bal = load_balance_loss(out["p_retr"], num_experts=3, masks=b_masks)

            loss = w_bin * loss_bin + w_bal * loss_bal

            if w_retr > 0:
                if retr_only_on_malware:
                    mask = (y_bin == 1)
                    if mask.sum() >= 2:
                        loss_retr = multilabel_infonce(out["h"][mask], Y_type[mask], y_bin=y_bin[mask], temperature=temperature)
                        loss = loss + w_retr * loss_retr
                else:
                    loss_retr = multilabel_infonce(out["h"], Y_type, y_bin=y_bin,temperature=temperature)
                    loss = loss + w_retr * loss_retr

            total_loss += float(loss.item())
            total_sup  += float(loss_bin.item())

    avg_loss = total_loss / max(len(loader), 1)
    avg_sup  = total_sup  / max(len(loader), 1)
    return avg_loss, avg_sup

    
@torch.no_grad()
def multilabel_micro_f1(logits: torch.Tensor, targets: torch.Tensor, thresh=0.5, eps=1e-12):
    probs = torch.sigmoid(logits)
    preds = (probs >= thresh).float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1.item()

import re

def sha_stats(keys, name):
    keys = [str(k).strip().lower() for k in keys]
    is_sha = [bool(re.fullmatch(r"[0-9a-f]{64}", k)) for k in keys]
    print(f"{name}: total={len(keys)} sha256_like={sum(is_sha)} non_sha={len(keys)-sum(is_sha)}")
    # print a few bad keys
    bad = [k for k, ok in zip(keys, is_sha) if not ok][:10]
    print(f"{name}: example non_sha keys:", bad)


def make_train_val_split(y_bin: np.ndarray, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    y = y_bin.astype(int)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    rng.shuffle(idx0); rng.shuffle(idx1)

    n0v = int(len(idx0) * val_frac)
    n1v = int(len(idx1) * val_frac)

    val_idx = np.concatenate([idx0[:n0v], idx1[:n1v]])
    train_idx = np.concatenate([idx0[n0v:], idx1[n1v:]])

    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return train_idx, val_idx

#mapping the malconv benignware to sha for it to be present inside the global ids list 
from pathlib import Path

def remap_malconv_map_to_sha(mal_map, filename_to_sha):
    out = {}
    unmapped = 0
    for sid, v in mal_map.items():
        name = str(sid).strip().lower()
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

# =========================
# Master DF + training run
# =========================
def main():
    # 1) global_ids = union(malware sha list + benign sha list)
    malware_ids = load_sha_list_txt(MALWARE_HASHES_TXT)
    import re
    benign_ids = load_sha_list_csv(BENIGN_HASHES_CSV)
    print("benign_ids count:", len(benign_ids))
    print("first 5:", benign_ids[:5])
    print("valid sha256 %:", sum(bool(re.fullmatch(r"[0-9a-f]{64}", x)) for x in benign_ids)/len(benign_ids))
    global_ids  = list(dict.fromkeys(malware_ids + benign_ids))
    print("global_ids:", len(global_ids), "(malware:", len(malware_ids), "benign:", len(benign_ids), ")")

    # 2) Load MASTER and reindex to global_ids (this guarantees row order alignment)
    master_df = pd.read_csv(MASTER_TABULAR_CSV)
    assert "sha256" in master_df.columns, "master_df must have sha256"
    assert "is_malware" in master_df.columns, "master_df must have is_malware"

    # dedupe safety (malware wins)
    master_df = master_df.sort_values("is_malware", ascending=False).drop_duplicates("sha256", keep="first")

    master_df = master_df.set_index("sha256").reindex(global_ids)
    master_df["sha256"] = master_df.index

    # 3) Labels
    y_bin = master_df["is_malware"].fillna(0).astype(int).to_numpy()

    # multi-label targets: count>0 => 1 else 0
    for c in COLS_TYPE:
        if c not in master_df.columns:
            master_df[c] = 0
    Y_type = (master_df[COLS_TYPE].fillna(0).to_numpy() > 0).astype(np.float32)
    print("Y_type nonzero rows:", (Y_type.sum(axis=1) > 0).sum(), "out of", len(Y_type))
    print("Y_type nonzero malware only:", (Y_type[y_bin==1].sum(axis=1) > 0).sum(), "out of", (y_bin==1).sum())
    print("per-class counts:", Y_type.sum(axis=0))
    print("class names:", COLS_TYPE)
    # 4) Gate input X_tab from canonical EMBER named columns
    names = [line.strip() for line in open(COLUMN_NAMES_TXT)]
    ember_cols = names[1:1+2568]  # byte_hist_0 ... fallback_14

    # ensure all exist (fill missing with 0)
    for c in ember_cols:
        if c not in master_df.columns:
            master_df[c] = 0.0

    X_tab = master_df[ember_cols].fillna(0).to_numpy(np.float32)
   
    X_tab_retr=np.concatenate([X_tab,Y_type],axis=1)

    # 5) Load + align experts by sha256
    # ids_safe, E_safe_raw, _ = load_safe_pooled_from_binary(SAFE_DIR)
    # safe_map = {sid: E_safe_raw[i] for i, sid in enumerate(ids_safe)}
    # d_safe = E_safe_raw.shape[1]
    # after master_df is loaded and reindexed
    filename_to_sha = build_filename_to_sha()

    safe_mal_map, miss1 = load_safe_dir_to_map(SAFE_DIR_MAL)   # sha256 folders
    safe_ben_map, miss2, unmapped, bad = load_safe_benign_nested_to_map(SAFE_DIR_BEN, filename_to_sha)
    print("SAFE benign missing pooled:", miss2, "unmapped:", unmapped, "bad npy:", bad)
    safe_map = merge_maps(prefer_first=True, map_a=safe_mal_map, map_b=safe_ben_map)
    gid_set = set(global_ids)
    safe_keys = set(safe_map.keys())
    print("SAFE keys:", len(safe_keys))
    print("SAFE ∩ global_ids:", len(safe_keys & gid_set))
    print("SAFE only (not in global_ids):", len(safe_keys - gid_set))
    global_ids = [s.strip().lower() for s in global_ids]
    safe_map = {str(k).strip().lower(): v for k, v in safe_map.items()}
    d_safe = next(iter(safe_map.values())).shape[0]
    print("SAFE total:", len(safe_map), "dim:", d_safe, "missing pooled:", miss1 + miss2)

    malconv_mal_map = load_pooled_folder_to_map(MALCONV_DIR_MAL)
    malconv_ben_map = load_pooled_folder_to_map(MALCONV_DIR_BEN)

    # prefer malware if collision
    mal_map_raw = merge_maps(prefer_first=True, map_a=malconv_mal_map, map_b=malconv_ben_map)
    mal_map_raw = merge_maps(True, malconv_mal_map, malconv_ben_map)
    mal_map, unmapped = remap_malconv_map_to_sha(mal_map_raw, filename_to_sha)
    print("MalConv remap unmapped:", unmapped, "final keys:", len(mal_map))
    d_mal = next(iter(mal_map.values())).shape[0]
    print("MalConv total:", len(mal_map), "dim:", d_mal)

    n2v_map, d_n2v = merge_node2vec(N2V_MALWARE_NPY, N2V_BENIGN_NPY)

    n2v_map, kept_sha, unmapped = remap_n2v_to_sha(n2v_map, filename_to_sha)
    print("N2V remap: kept_sha:", kept_sha, "unmapped:", unmapped, "final keys:", len(n2v_map))
    mal_map = {str(k).strip().lower(): v for k, v in mal_map.items()}
    n2v_map = {str(k).strip().lower(): v for k, v in n2v_map.items()}
    sha_stats(mal_map.keys(), "MalConv keys")
    sha_stats(n2v_map.keys(), "N2V keys")
    print("global_ids sha256_like:", sum(bool(re.fullmatch(r"[0-9a-f]{64}", str(x).strip().lower())) for x in global_ids))
    E_safe, m_safe = align_embeddings(global_ids, safe_map, d_safe)
    E_mal,  m_mal  = align_embeddings(global_ids, mal_map,  d_mal)
    E_n2v,  m_n2v  = align_embeddings(global_ids, n2v_map,  d_n2v)

    print(f"SAFE coverage:    {m_safe.mean():.3f} ({int(m_safe.sum())}/{len(m_safe)})")
    print(f"MalConv coverage: {m_mal.mean():.3f} ({int(m_mal.sum())}/{len(m_mal)})")
    print(f"N2V coverage:     {m_n2v.mean():.3f} ({int(m_n2v.sum())}/{len(m_n2v)})")

    # 6) Append expert-missing masks to gate input (prevents NaN issues)
    masks = np.stack([m_mal, m_safe, m_n2v], axis=1).astype(np.float32)  # (N,3)
    X_tab = np.concatenate([X_tab, masks], axis=1)
    X_tab_bin=X_tab
    X_tab_retr=np.concatenate([X_tab, Y_type],axis=1)
    # 7) Train
    train_idx, val_idx = make_train_val_split(y_bin, val_frac=0.15, seed=42)

    ds = PrecomputedExpertsDataset(X_tab_bin, X_tab_retr, E_mal, E_safe, E_n2v, y_bin, Y_type, masks)
    family_counts  = Y_type[train_idx].sum(axis=0).clip(1)   # (11,) counts per family in train set only
# compute raw inverse-frequency weights for malware
    sample_weights = np.ones(len(train_idx), dtype=np.float32)

    for i, idx in enumerate(train_idx):
        if y_bin[idx] == 1:
            # malware — use inverse family frequency for within-malware balance
            families_present = np.where(Y_type[idx] > 0)[0]
            if len(families_present) > 0:
                sample_weights[i] = float(np.mean(1.0 / family_counts[families_present])) #within malware inverse frequency weighting so that the rarer families appear more often 
            else:
                sample_weights[i] = 0.1
        # benign stays at 1.0

    # now scale ALL malware weights so their total == total benign weight
    mal_mask = (y_bin[train_idx] == 1)
    ben_total = (~mal_mask).sum()          # each benign = 1.0, so total = count
    mal_total = sample_weights[mal_mask].sum()
    sample_weights[mal_mask] *= (ben_total / mal_total)  # scale up malware       # scale malware to match    # downsample benign heavily so malware dominates batches
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx), batch_size=256, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(torch.utils.data.Subset(ds, val_idx), batch_size=128, shuffle=False, num_workers=0)

    model = MoEOnEmbeddings(tab_dim_bin=X_tab_bin.shape[1],tab_dim_retr=X_tab_retr.shape[1], d_mal=d_mal, d_safe=d_safe, d_n2v=d_n2v, d_common=256, num_types=len(COLS_TYPE))
    mal_Y = Y_type[y_bin == 1]
    pos_counts = mal_Y.sum(axis=0).clip(1)
    neg_counts = len(mal_Y) - pos_counts
    per_class_pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32)
    per_class_pos_weight = torch.full((len(COLS_TYPE),), 3.0)
    train_with_val(
        model, train_loader, val_loader,
        device=DEVICE,
        epochs=50,
        lr=5e-4,
        w_retr_max=0.2,      # safer than 0.5
        ckpt_path="best_moe.pt",
        save_by="val_sup",  
        retr_only_on_malware=True,
        # per_class_pos_weight=per_class_pos_weight,  # if your goal is malware-type tagging
    )
    print("saved as best_moe.pt")

if __name__ == "__main__":
    main()