"""
Inserts the multi-label evaluation functions into evaluate_experts_moe_files.py
without manual copy-paste (which mangles indentation).

Run once:   python install_multilabel.py

It inserts the three functions immediately above `def main():` so they are
defined before main() runs, and makes a .bak backup first.
Re-running is safe: it detects the marker and skips.
"""
import os, shutil, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
TARGET = "evaluate_experts_moe_files.py"
MARKER = "# === MULTI-LABEL BEHAVIOUR PREDICTION (inserted) ==="

BLOCK = '''
# === MULTI-LABEL BEHAVIOUR PREDICTION (inserted) ===
# Predicts SOREL behaviour tags by similarity-weighted voting over the tags of
# retrieved neighbours. The query's own tags are never an input - only the
# neighbours' (training) tags are used; the query's tags are ground truth only.

def knn_tag_scores(index, query_emb, index_idx, Y_type, k=10, tau=0.07):
    """Returns (n_queries, 11) soft tag scores in [0, 1]."""
    index_tags = Y_type[index_idx]
    scores = np.zeros((len(query_emb), Y_type.shape[1]), dtype=np.float32)
    for i, q in enumerate(query_emb):
        D, I = query_faiss(index, q, k)
        w = np.exp((D - D.max()) / tau)
        w = w / max(w.sum(), 1e-9)
        scores[i] = (w[:, None] * index_tags[I]).sum(axis=0)
    return scores


def multilabel_metrics(y_true, y_pred, scores=None, label_names=None):
    """Micro/macro F1, per-label F1, and optionally mAP."""
    eps = 1e-9
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)

    micro_p = tp.sum() / (tp.sum() + fp.sum() + eps)
    micro_r = tp.sum() / (tp.sum() + fn.sum() + eps)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + eps)

    per_p = tp / (tp + fp + eps)
    per_r = tp / (tp + fn + eps)
    per_f1 = 2 * per_p * per_r / (per_p + per_r + eps)

    out = {"micro_f1": float(micro_f1), "macro_f1": float(per_f1.mean()),
           "micro_precision": float(micro_p), "micro_recall": float(micro_r),
           "per_label_f1": per_f1, "support": y_true.sum(axis=0).astype(int)}

    if scores is not None:
        aps = []
        for j in range(y_true.shape[1]):
            if y_true[:, j].sum() == 0:
                continue
            order = np.argsort(-scores[:, j])
            rel = y_true[order, j]
            prec_at = np.cumsum(rel) / (np.arange(len(rel)) + 1)
            aps.append((prec_at * rel).sum() / rel.sum())
        out["mAP"] = float(np.mean(aps)) if aps else float("nan")
    return out


def evaluate_multilabel(H_all, Y_type, index_idx, val_idx, test_idx,
                        y_bin, k=10, tau=0.07, label_names=None):
    """Sweep the decision threshold on validation, apply once to test."""
    thresholds = np.arange(0.10, 0.65, 0.05)

    def tagged_malware(idx):
        m = (y_bin[idx] == 1) & (Y_type[idx].sum(axis=1) > 0)
        return idx[m]

    val_q = tagged_malware(val_idx)
    test_q = tagged_malware(test_idx)
    print(f"\\n  multi-label subsets: val {len(val_q)}, test {len(test_q)} "
          f"(tagged malware only)")

    index = build_faiss_index(H_all[index_idx])

    val_scores = knn_tag_scores(index, H_all[val_q], index_idx, Y_type, k, tau)
    val_true = Y_type[val_q]

    best_t, best_f1 = None, -1.0
    print("  threshold sweep (validation):")
    for t in thresholds:
        m = multilabel_metrics(val_true, (val_scores >= t).astype(np.float32))
        print(f"    t={t:.2f}  micro-F1={m['micro_f1']:.4f}  "
              f"macro-F1={m['macro_f1']:.4f}")
        if m["micro_f1"] > best_f1:
            best_f1, best_t = m["micro_f1"], float(t)
    print(f"  -> selected threshold {best_t:.2f} (val micro-F1 {best_f1:.4f})")

    test_scores = knn_tag_scores(index, H_all[test_q], index_idx, Y_type, k, tau)
    test_true = Y_type[test_q]
    res = multilabel_metrics(test_true,
                             (test_scores >= best_t).astype(np.float32),
                             scores=test_scores)
    res["threshold"] = best_t
    res["n_test"] = len(test_q)

    print("\\n" + "=" * 60)
    print(f"  Multi-label behaviour prediction "
          f"(k={k}, threshold={best_t:.2f}, n={len(test_q)})")
    print("=" * 60)
    print(f"  micro-F1 {res['micro_f1']:.4f} | macro-F1 {res['macro_f1']:.4f} | "
          f"mAP {res.get('mAP', float('nan')):.4f}")
    print(f"  micro-precision {res['micro_precision']:.4f} | "
          f"micro-recall {res['micro_recall']:.4f}")
    print("\\n  per-label F1 (test support):")
    names = label_names or [f"tag{j}" for j in range(test_true.shape[1])]
    for j, nm in enumerate(names):
        print(f"    {nm:<15s} F1={res['per_label_f1'][j]:.4f}  "
              f"n={res['support'][j]}")
    print("\\n  NOTE: rare categories have small test support; their per-label")
    print("        F1 is unstable and should be read alongside micro-F1.")
    return res

# === END MULTI-LABEL BLOCK ===

'''


def main():
    if not os.path.exists(TARGET):
        sys.exit(f"ABORT: {TARGET} not found in {os.getcwd()}")

    src = open(TARGET).read()

    if MARKER in src:
        print("Already installed — nothing to do.")
        return

    if "def main():" not in src:
        sys.exit("ABORT: could not find 'def main():' — insert manually.")

    shutil.copy(TARGET, TARGET + ".bak")
    print(f"backup -> {TARGET}.bak")

    new = src.replace("def main():", BLOCK + "\ndef main():", 1)
    open(TARGET, "w").write(new)

    import ast
    try:
        ast.parse(open(TARGET).read())
        print("inserted and syntax-checked OK")
    except SyntaxError as e:
        shutil.copy(TARGET + ".bak", TARGET)
        sys.exit(f"ABORT: syntax error after insert ({e}); original restored")

    print("\nNow add this call inside main(), after H_all is computed:\n")
    print("    evaluate_multilabel(H_all, Y_type, index_idx, val_idx, test_idx,")
    print("                        y_bin, k=10, label_names=COLS_TYPE)")


if __name__ == "__main__":
    main()
