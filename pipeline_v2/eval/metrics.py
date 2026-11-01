"""Metric functions shared by the MoE evaluation and every baseline."""
import faiss
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


# ------------------------------------------------------------------ detection
def tpr_at_fpr(y, s, fpr_target):
    fpr, tpr, thr = roc_curve(y, s)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_for_fpr(y_val, s_val, fpr_target):
    """Smallest threshold whose VALIDATION FPR <= target (applied unchanged to test)."""
    neg = np.sort(s_val[y_val == 0])[::-1]
    k = int(np.floor(fpr_target * len(neg)))
    return float(neg[k]) + 1e-12 if k < len(neg) else float(neg[-1])


def detection(y, s, y_val=None, s_val=None, fprs=(0.001, 0.01), thr=0.5):
    y = np.asarray(y).astype(int)
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    out = {"n": int(len(y)), "n_malware": int(y.sum()), "accuracy": (tp + tn) / len(y),
           "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
           "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
           "pr_auc": float(average_precision_score(y, s)) if y.sum() else float("nan"),
           "majority_acc": float(max(y.mean(), 1 - y.mean()))}
    n_neg = int((y == 0).sum())
    for f in fprs:
        out[f"tpr@fpr={f:g}"] = tpr_at_fpr(y, s, f)
        out[f"reliable@fpr={f:g}"] = bool(n_neg * f >= 10)       # >=10 negatives inside the budget
        if s_val is not None:
            t = threshold_for_fpr(np.asarray(y_val), s_val, f)
            out[f"test_tpr@valthr_fpr={f:g}"] = float((s[y == 1] >= t).mean()) if y.sum() else float("nan")
            out[f"test_fpr@valthr_fpr={f:g}"] = float((s[y == 0] >= t).mean()) if n_neg else float("nan")
    return out


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def fit_temperature(y_val, p_val):
    """Temperature scaling on the validation logits (1-D grid search, no test use)."""
    logit = np.log(np.clip(p_val, 1e-7, 1 - 1e-7) / np.clip(1 - p_val, 1e-7, 1))
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.05), np.log(20), 200)):
        q = 1 / (1 + np.exp(np.clip(-logit / t, -50, 50)))
        nll = -np.mean(y_val * np.log(q + 1e-12) + (1 - y_val) * np.log(1 - q + 1e-12))
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t


def apply_temperature(p, t):
    logit = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
    return 1 / (1 + np.exp(np.clip(-logit / t, -50, 50)))


# ------------------------------------------------------------------ retrieval
def index_of(E):
    E = np.ascontiguousarray(E, dtype=np.float32).copy()
    faiss.normalize_L2(E)
    idx = faiss.IndexFlatIP(E.shape[1])
    idx.add(E)
    return idx


def search(idx, Q, k):
    Q = np.ascontiguousarray(Q, dtype=np.float32).copy()
    faiss.normalize_L2(Q)
    return idx.search(Q, k)


def knn_scores(E_index, y_index, E_query, k=20, tau=0.07):
    """Similarity-weighted kNN malware probability + nearest-neighbour similarity."""
    D, I = search(index_of(E_index), E_query, k)
    w = np.exp((D - D.max(1, keepdims=True)) / tau)
    w /= w.sum(1, keepdims=True)
    return (w * y_index[I]).sum(1), D[:, 0], I


def accuracy_at_1(y_index, y_query, I):
    return float((y_index[I[:, 0]] == y_query).mean())


def tag_overlap_retrieval(E_index, Y_index, E_query, Y_query, ks=(1, 3, 5, 10)):
    """Relevant neighbour = shares >= 1 behaviour tag with the query (your Table 10)."""
    D, I = search(index_of(E_index), E_query, max(ks))
    out = {}
    disc = 1.0 / np.log2(np.arange(2, max(ks) + 2))
    rel = (np.einsum("qkt,qt->qk", Y_index[I], Y_query) > 0).astype(float)
    n_rel = (Y_query @ Y_index.T > 0).sum(1)
    for k in ks:
        p = rel[:, :k].mean(1)
        idcg = np.array([disc[:min(k, n)].sum() for n in n_rel])
        nd = np.where(idcg > 0, (rel[:, :k] * disc[:k]).sum(1) / np.maximum(idcg, 1e-12), 0)
        out[k] = {"precision": float(p.mean()), "ndcg": float(nd.mean())}
    return out


def knn_tag_scores(E_index, Y_index, E_query, k=20, tau=0.3):
    D, I = search(index_of(E_index), E_query, k)
    w = np.exp((D - D.max(1, keepdims=True)) / tau)
    w /= w.sum(1, keepdims=True)
    return np.einsum("qk,qkt->qt", w, Y_index[I])


def multilabel(y_true, scores, thr):
    pred = (scores >= thr).astype(float)
    tp = (y_true * pred).sum(0); fp = ((1 - y_true) * pred).sum(0); fn = (y_true * (1 - pred)).sum(0)
    mp, mr = tp.sum() / max(tp.sum() + fp.sum(), 1e-9), tp.sum() / max(tp.sum() + fn.sum(), 1e-9)
    per = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    aps = [average_precision_score(y_true[:, j], scores[:, j]) for j in range(y_true.shape[1]) if y_true[:, j].any()]
    return {"micro_f1": float(2 * mp * mr / max(mp + mr, 1e-9)), "macro_f1": float(per.mean()),
            "micro_precision": float(mp), "micro_recall": float(mr), "mAP": float(np.mean(aps)) if aps else float("nan"),
            "per_label_f1": per.tolist(), "support": y_true.sum(0).astype(int).tolist(), "threshold": float(thr)}


def select_threshold(y_val, s_val, grid=np.arange(0.05, 0.95, 0.05)):
    best = max(grid, key=lambda t: multilabel(y_val, s_val, t)["micro_f1"])
    return float(best)


# ------------------------------------------------------------------ triage / abstention
def risk_coverage(y, p, novelty, coverages=(1.0, 0.95, 0.9, 0.8)):
    """Abstain on the most novel samples (send them to a sandbox / analyst);
    report accuracy and TPR on the samples that are kept."""
    order = np.argsort(novelty)                      # least novel first
    out = {}
    for c in coverages:
        keep = order[: int(round(c * len(order)))]
        yy, pp = y[keep], (p[keep] >= 0.5).astype(int)
        out[f"cov={c:g}"] = {"n": int(len(keep)), "accuracy": float((yy == pp).mean()),
                             "malware_recall": float(pp[yy == 1].mean()) if (yy == 1).any() else float("nan"),
                             "benign_fpr": float(pp[yy == 0].mean()) if (yy == 0).any() else float("nan")}
    return out
