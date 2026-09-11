"""
Multi-label behaviour prediction by similarity-weighted voting over retrieved
neighbours' SOREL tags.

Add this to evaluate_experts_moe_files.py (or import it). It is evaluation-only:
no retraining, no change to the embedding space.

METHOD
    For each query, retrieve the top-k neighbours from the MoE index, take their
    11-dim multi-hot Y_type vectors, average them weighted by cosine similarity,
    and threshold to obtain a predicted tag set.

    The query's own tags are NEVER an input — only the neighbours' (training)
    tags are used. The query's tags appear solely as ground truth for scoring.

PROTOCOL
    Threshold is swept on the VALIDATION partition and applied ONCE to test,
    the same discipline used for the expert-dropout rate.

SCOPE
    Evaluated on tagged malware only. Benign samples are all-zero by
    construction and would inflate every metric.
"""
import numpy as np


# --------------------------------------------------------------------------
# Core: neighbour tag voting
# --------------------------------------------------------------------------
def knn_tag_scores(index, query_emb, index_idx, Y_type, k=10, tau=0.07):
    """
    Returns (n_queries, 11) soft tag scores in [0, 1].

    index      : FAISS index built from H_all[index_idx]
    query_emb  : (n_queries, 256) embeddings for the query partition
    index_idx  : global row positions of the samples inside `index`
    Y_type     : (n_samples, 11) multi-hot tag matrix, aligned to global_ids
    """
    index_tags = Y_type[index_idx]                      # (n_index, 11)
    scores = np.zeros((len(query_emb), Y_type.shape[1]), dtype=np.float32)

    for i, q in enumerate(query_emb):
        D, I = query_faiss(index, q, k)                 # I indexes INTO the subset
        # similarity-weighted vote (temperature-scaled softmax over cosine sims)
        w = np.exp((D - D.max()) / tau)
        w = w / max(w.sum(), 1e-9)
        scores[i] = (w[:, None] * index_tags[I]).sum(axis=0)
    return scores


# --------------------------------------------------------------------------
# Multi-label metrics
# --------------------------------------------------------------------------
def multilabel_metrics(y_true, y_pred, scores=None, label_names=None):
    """Micro/macro F1, per-label F1, and (optionally) mean average precision."""
    eps = 1e-9
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)

    # micro
    micro_p = tp.sum() / (tp.sum() + fp.sum() + eps)
    micro_r = tp.sum() / (tp.sum() + fn.sum() + eps)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + eps)

    # per-label / macro
    per_p = tp / (tp + fp + eps)
    per_r = tp / (tp + fn + eps)
    per_f1 = 2 * per_p * per_r / (per_p + per_r + eps)
    macro_f1 = per_f1.mean()

    out = {"micro_f1": float(micro_f1), "macro_f1": float(macro_f1),
           "micro_precision": float(micro_p), "micro_recall": float(micro_r),
           "per_label_f1": per_f1, "support": y_true.sum(axis=0).astype(int)}

    if scores is not None:
        aps = []
        for j in range(y_true.shape[1]):
            if y_true[:, j].sum() == 0:
                continue
            order = np.argsort(-scores[:, j])
            rel = y_true[order, j]
            csum = np.cumsum(rel)
            prec_at = csum / (np.arange(len(rel)) + 1)
            aps.append((prec_at * rel).sum() / rel.sum())
        out["mAP"] = float(np.mean(aps)) if aps else float("nan")

    if label_names is not None:
        out["label_names"] = label_names
    return out


# --------------------------------------------------------------------------
# Driver: sweep threshold on val, report once on test
# --------------------------------------------------------------------------
def evaluate_multilabel(H_all, Y_type, index_idx, val_idx, test_idx,
                        y_bin, k=10, tau=0.07, label_names=None,
                        thresholds=np.arange(0.10, 0.65, 0.05)):
    # tagged malware only
    def tagged_malware(idx):
        m = (y_bin[idx] == 1) & (Y_type[idx].sum(axis=1) > 0)
        return idx[m]

    val_q  = tagged_malware(val_idx)
    test_q = tagged_malware(test_idx)
    print(f"  multi-label eval subsets: val {len(val_q)}, test {len(test_q)} "
          f"(tagged malware only)")

    index = build_faiss_index(H_all[index_idx])

    # --- threshold selection on validation --------------------------------
    val_scores = knn_tag_scores(index, H_all[val_q], index_idx, Y_type, k, tau)
    val_true   = Y_type[val_q]

    best_t, best_f1 = None, -1.0
    print("\n  threshold sweep (validation):")
    for t in thresholds:
        m = multilabel_metrics(val_true, (val_scores >= t).astype(np.float32))
        print(f"    t={t:.2f}  micro-F1={m['micro_f1']:.4f}  macro-F1={m['macro_f1']:.4f}")
        if m["micro_f1"] > best_f1:
            best_f1, best_t = m["micro_f1"], float(t)
    print(f"  -> selected threshold {best_t:.2f} (val micro-F1 {best_f1:.4f})")

    # --- single application to test ---------------------------------------
    test_scores = knn_tag_scores(index, H_all[test_q], index_idx, Y_type, k, tau)
    test_true   = Y_type[test_q]
    res = multilabel_metrics(test_true, (test_scores >= best_t).astype(np.float32),
                             scores=test_scores, label_names=label_names)
    res["threshold"] = best_t
    res["n_test"] = len(test_q)

    # --- print -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Multi-label behaviour prediction (k={k}, threshold={best_t:.2f}, "
          f"n={len(test_q)})")
    print("=" * 60)
    print(f"  micro-F1 {res['micro_f1']:.4f} | macro-F1 {res['macro_f1']:.4f} | "
          f"mAP {res.get('mAP', float('nan')):.4f}")
    print(f"  micro-precision {res['micro_precision']:.4f} | "
          f"micro-recall {res['micro_recall']:.4f}")
    print("\n  per-label F1 (support in test):")
    names = label_names or [f"tag{j}" for j in range(test_true.shape[1])]
    for j, nm in enumerate(names):
        print(f"    {nm:<15s} F1={res['per_label_f1'][j]:.4f}  n={res['support'][j]}")
    print("\n  NOTE: rare categories have very small test support; their per-label")
    print("        F1 is unstable and should be read alongside micro-F1.")
    return res


# --------------------------------------------------------------------------
# Call this from main(), after H_all is computed:
#
#     evaluate_multilabel(H_all, Y_type, index_idx, val_idx, test_idx,
#                         y_bin, k=10, label_names=COLS_TYPE)
# --------------------------------------------------------------------------
