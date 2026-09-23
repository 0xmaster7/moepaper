"""
install_final_evals.py
Patches evaluate_experts_moe_files.py with the three remaining experiments.
Run ONCE from the moepaper directory:   python install_final_evals.py

What it changes (backup written to evaluate_experts_moe_files.py.bak2):
  1. MOE_CKPT reads the MOE_CKPT environment variable (if not already).
  2. MoEOnEmbeddings.forward also computes the binary branch
     (p_bin masked + renormalised, z_bin, bin_logits) - identical to training.
  3. Inserts the experiment functions above `def main():`.
  4. Inserts one call after the MoE FAISS index is written (runs on TEST only).
Re-running is safe (marker check). Syntax is verified; on failure the
original file is restored.
"""
import os, re, sys, shutil, ast

os.chdir(os.path.dirname(os.path.abspath(__file__)))
TARGET = "evaluate_experts_moe_files.py"
MARKER = "# === FINAL EVALS (inserted) ==="

FUNCS = r'''
# === FINAL EVALS (inserted) ===
import json

def _softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _roc_auc(y, s):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, s))
    except Exception:
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def extract_gate_outputs(model, X_tab_bin, X_tab_retr, E_mal, E_safe, E_n2v,
                         masks, batch_size=256, device="cpu"):
    """Binary logits and masked/renormalised gate weights for every sample."""
    model.eval(); model.to(device)
    L, PB, PR = [], [], []
    with torch.no_grad():
        for s in range(0, X_tab_bin.shape[0], batch_size):
            e = s + batch_size
            out = model(
                torch.from_numpy(X_tab_bin[s:e]).to(device),
                torch.from_numpy(X_tab_retr[s:e]).to(device),
                torch.from_numpy(E_mal[s:e]).to(device),
                torch.from_numpy(E_safe[s:e]).to(device),
                torch.from_numpy(E_n2v[s:e]).to(device),
                masks=torch.from_numpy(masks[s:e]).to(device),
            )
            L.append(out["bin_logits"].cpu().numpy())
            PB.append(out["p_bin"].cpu().numpy())
            PR.append(out["p_retr"].cpu().numpy())
    return np.vstack(L), np.vstack(PB), np.vstack(PR)


# ---- Experiment 1: binary detection ------------------------------------
def eval_binary_detection(bin_logits, y_bin, test_idx):
    prob = _softmax_np(bin_logits[test_idx])[:, 1]
    y = y_bin[test_idx].astype(int)
    pred = (prob >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    res = {
        "n": int(len(y)), "n_malware": int(y.sum()),
        "accuracy": (tp + tn) / len(y),
        "precision_malware": prec, "recall_malware": rec,
        "f1_malware": 2 * prec * rec / max(prec + rec, 1e-9),
        "roc_auc": _roc_auc(y, prob),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "majority_baseline_acc": float(1 - y.mean()),
    }
    print("\n" + "=" * 60 + "\n  EXPERIMENT 1 - Binary detection (gate_bin + classifier head)\n" + "=" * 60)
    print(f"  n={res['n']} (malware {res['n_malware']})  threshold=0.5")
    print(f"  accuracy {res['accuracy']:.4f}  (all-benign baseline {res['majority_baseline_acc']:.4f})")
    print(f"  malware precision {prec:.4f} | recall {rec:.4f} | F1 {res['f1_malware']:.4f}")
    print(f"  ROC-AUC {res['roc_auc']:.4f}")
    print(f"  confusion  TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    return res


# ---- Experiment 2: behaviour-level (tag-overlap) retrieval ---------------
def _tag_overlap_retrieval(E, index_pos, query_pos, Y_type, ks):
    """Relevant neighbour = shares >= 1 behaviour tag with the query."""
    index = build_faiss_index(E[index_pos])
    index_tags = Y_type[index_pos]
    kmax = max(ks)
    P = {k: [] for k in ks}; N = {k: [] for k in ks}
    for q in query_pos:
        _, I = query_faiss(index, E[q], kmax)
        rel = (index_tags[I] @ Y_type[q] > 0).astype(float)
        n_rel_total = int((index_tags @ Y_type[q] > 0).sum())
        for k in ks:
            r = rel[:k]
            P[k].append(r.mean())
            disc = 1.0 / np.log2(np.arange(2, k + 2))
            idcg = disc[:min(k, n_rel_total)].sum()
            N[k].append((r * disc).sum() / idcg if idcg > 0 else 0.0)
    return {k: {"precision": float(np.mean(P[k])), "ndcg": float(np.mean(N[k]))} for k in ks}


def eval_tag_overlap(H_all, experts, index_idx, test_idx, y_bin, Y_type, ks):
    tagged = test_idx[(y_bin[test_idx] == 1) & (Y_type[test_idx].sum(axis=1) > 0)]
    out = {}
    print("\n" + "=" * 60 + "\n  EXPERIMENT 2 - Behaviour-level retrieval (shared-tag relevance)\n" + "=" * 60)
    print("  Query set: tagged malware in test. Index: train+val (all samples).")
    rows = [("MoE", H_all, None)] + experts
    for name, E, m in rows:
        if m is None:
            ii, qi = index_idx, tagged
        else:
            ii = index_idx[m[index_idx].astype(bool)]
            qi = tagged[m[tagged].astype(bool)]
        r = _tag_overlap_retrieval(E, ii, qi, Y_type, ks)
        out[name] = {"n_queries": int(len(qi)), "metrics": r}
        print(f"\n  {name}  (queries {len(qi)}, index {len(ii)})")
        print("     k   Precision     nDCG")
        for k in ks:
            print(f"   {k:3d}      {r[k]['precision']:.4f}   {r[k]['ndcg']:.4f}")
    qi = tagged[m_safe[tagged].astype(bool)]
    r = _tag_overlap_retrieval(H_all, index_idx, qi, Y_type, ks)
    print("MoE on SAFE-covered queries:", {k: round(r[k]["precision"], 4) for k in ks})
    return out


# ---- Experiment 3: gate weights (Figure panel c + Section 2.5 evidence) ----
def eval_gate_weights(model, X_tab_bin, X_tab_retr, masks, p_bin, p_retr,
                      y_bin, test_idx):
    print("\n" + "=" * 60 + "\n  EXPERIMENT 3 - Gate weights\n" + "=" * 60)
    names = ["MalConv", "SAFE", "Node2Vec"]
    res = {"by_pattern": {}, "example": None}

    print("  Mean renormalised weights on TEST, grouped by availability pattern")
    print("  pattern (Mal,SAFE,N2V)     n    gate_bin [M S N]        gate_retr [M S N]")
    pats = masks[test_idx].astype(int)
    for pat in sorted({tuple(p) for p in pats}, reverse=True):
        sel = test_idx[(pats == np.array(pat)).all(axis=1)]
        mb = p_bin[sel].mean(axis=0); mr = p_retr[sel].mean(axis=0)
        key = "".join(map(str, pat))
        res["by_pattern"][key] = {"n": int(len(sel)), "gate_bin": mb.tolist(), "gate_retr": mr.tolist()}
        print(f"  {str(pat):<24s} {len(sel):5d}   "
              f"[{mb[0]:.3f} {mb[1]:.3f} {mb[2]:.3f}]   [{mr[0]:.3f} {mr[1]:.3f} {mr[2]:.3f}]")

    cand = [i for i in test_idx if masks[i, 0] == 1 and masks[i, 1] == 0
            and masks[i, 2] == 1 and y_bin[i] == 1]
    if cand:
        i = int(cand[0])
        with torch.no_grad():
            rb, _ = model.gate_bin(torch.from_numpy(X_tab_bin[i:i + 1]))
            rr, _ = model.gate_retr(torch.from_numpy(X_tab_retr[i:i + 1]))
        rb = rb.numpy()[0]; rr = rr.numpy()[0]
        res["example"] = {"row": i, "mask": masks[i].tolist(),
                          "gate_bin_raw": rb.tolist(), "gate_bin_renorm": p_bin[i].tolist(),
                          "gate_retr_raw": rr.tolist(), "gate_retr_renorm": p_retr[i].tolist()}
        print(f"\n  Worked example (panel c): test malware row {i}, SAFE missing")
        print(f"  {'':22s} {names[0]:>9s} {names[1]:>9s} {names[2]:>9s}")
        for lbl, v in [("mask", masks[i]), ("gate_bin raw", rb), ("gate_bin renormalised", p_bin[i]),
                       ("gate_retr raw", rr), ("gate_retr renormalised", p_retr[i])]:
            print(f"  {lbl:22s} {v[0]:9.3f} {v[1]:9.3f} {v[2]:9.3f}")
    return res


def run_final_evals(model, X_tab_bin, X_tab_retr, E_mal, E_safe, E_n2v, masks,
                    m_mal, m_safe, m_n2v, y_bin, Y_type, index_idx, test_idx,
                    H_all, device="cpu"):
    loaded = {n for n, _ in model.named_parameters()}
    assert "bin_clf.weight" in loaded, "bin_clf not in model"
    bin_logits, p_bin, p_retr = extract_gate_outputs(
        model, X_tab_bin, X_tab_retr, E_mal, E_safe, E_n2v, masks, device=device)
    results = {
        "checkpoint": str(MOE_CKPT),
        "binary_detection": eval_binary_detection(bin_logits, y_bin, test_idx),
        "tag_overlap_retrieval": eval_tag_overlap(
            H_all, [("SAFE", E_safe, m_safe), ("MalConv", E_mal, m_mal),
                    ("Node2Vec", E_n2v, m_n2v)],
            index_idx, test_idx, y_bin, Y_type, K_VALUES),
        "gate_weights": eval_gate_weights(model, X_tab_bin, X_tab_retr, masks,
                                          p_bin, p_retr, y_bin, test_idx),
    }
    out = f"final_evals_{Path(str(MOE_CKPT)).stem}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\n  results saved -> {out}")
    return results
# === END FINAL EVALS ===

'''

CALL = '''
    # === FINAL EVALS CALL (inserted) ===
    if EVAL_SPLIT == "test":
        run_final_evals(model, X_tab_bin, X_tab_retr, E_mal, E_safe, E_n2v, masks,
                        m_mal, m_safe, m_n2v, y_bin, Y_type, index_idx, test_idx,
                        H_all, device=DEVICE)
'''

RENORM_ANCHOR = ("p_retr = p_retr_masked / p_retr_masked.sum(dim=-1, "
                 "keepdim=True).clamp_min(1e-9)\n")
BIN_MASK = ("            p_bin_masked = p_bin * masks\n"
            "            p_bin = p_bin_masked / p_bin_masked.sum(dim=-1, "
            "keepdim=True).clamp_min(1e-9)\n")
RETURN_RE = re.compile(r'return \{"p_retr": p_retr,\s*\n\s*"z_retr": z_retr,\s*\n\s*"h": h\}')
RETURN_NEW = ('z_bin = p_bin[:,0:1]*v_mal + p_bin[:,1:2]*v_safe + p_bin[:,2:3]*v_n2v\n'
              '        bin_logits = self.bin_clf(z_bin)\n'
              '        return {"p_retr": p_retr, "z_retr": z_retr, "h": h,\n'
              '                "p_bin": p_bin, "z_bin": z_bin, "bin_logits": bin_logits}')
CALL_ANCHOR = 'faiss.write_index(idx_moe, "faiss_moe.bin")\n'


def fail(msg):
    shutil.copy(TARGET + ".bak2", TARGET)
    sys.exit(f"ABORT: {msg} - original restored")


def main():
    if not os.path.exists(TARGET):
        sys.exit(f"ABORT: {TARGET} not found in {os.getcwd()}")
    src = open(TARGET).read()
    if MARKER in src:
        print("Already installed - nothing to do."); return
    shutil.copy(TARGET, TARGET + ".bak2")
    print(f"backup -> {TARGET}.bak2")

    # 1. checkpoint from environment
    m = re.search(r'^MOE_CKPT\s*=.*$', src, re.M)
    if m and "os.environ" not in m.group(0):
        src = src.replace(m.group(0),
            'MOE_CKPT = Path(os.environ.get("MOE_CKPT", "best_moe.pt"))', 1)
        print("  [1] MOE_CKPT now read from environment")
    else:
        print("  [1] MOE_CKPT already env-driven")

    # 2. binary branch in forward
    if src.count(RENORM_ANCHOR) != 1:
        fail("could not locate the p_retr renormalisation line in forward()")
    src = src.replace(RENORM_ANCHOR, RENORM_ANCHOR + BIN_MASK, 1)
    if not RETURN_RE.search(src):
        fail("could not locate forward()'s return dict")
    src = RETURN_RE.sub(RETURN_NEW, src, count=1)
    print("  [2] forward() now returns p_bin, z_bin, bin_logits")

    # 3. functions
    if "def main():" not in src:
        fail("could not find 'def main():'")
    src = src.replace("def main():", FUNCS + "\ndef main():", 1)
    print("  [3] experiment functions inserted")

    # 4. call
    if src.count(CALL_ANCHOR) != 1:
        fail("could not find the faiss_moe.bin write line in main()")
    src = src.replace(CALL_ANCHOR, CALL_ANCHOR + CALL, 1)
    print("  [4] run_final_evals() call inserted (runs when EVAL_SPLIT == 'test')")

    open(TARGET, "w").write(src)
    try:
        ast.parse(src)
    except SyntaxError as e:
        fail(f"syntax error after patch ({e})")
    print("\nInstalled and syntax-checked OK. Next: python run_final_evals.py")


if __name__ == "__main__":
    main()
