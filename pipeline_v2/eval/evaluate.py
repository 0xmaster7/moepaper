"""Evaluate one MoE run or one baseline on the TEST split (selection on VAL only).

    python eval/evaluate.py --run results/runs/moe_full/seed42
    python eval/evaluate.py --preds results/baselines/ember_lgbm/seed42/preds.npz

MoE run (outputs.npz: prob, h, p_bin, p_retr) gets:
  detection_head      classifier head (+ temperature-scaled calibration, ECE)
  detection_knn       FAISS kNN vote on h (index = train+val), Accuracy@1
  detection_fused     alpha*head + (1-alpha)*kNN, alpha picked on VAL
  behaviour_retrieval shared-tag P@k / nDCG@k (your Table 10)
  tags                kNN tag voting, threshold picked on VAL (micro/macro F1, mAP)
  triage              abstain on the most novel samples (NN distance) -> risk/coverage
  gates               mean gate weights by availability pattern
  experts_matched     each single expert vs the MoE on that expert's covered queries
Baseline (preds.npz: score[N], optional tag_scores[N,11], emb[N,d]; NaN = not covered)
  gets detection, tags and (if emb) the same retrieval/kNN metrics.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import load_config                       # noqa: E402
from eval import metrics as M                               # noqa: E402


def load(cfg, split):
    d = Path(cfg["dataset"]["local_dir"])
    D = dict(np.load(d / "dataset.npz", allow_pickle=False))
    S = np.load(d / f"splits_{split}.npz")
    assert str(S["fingerprint"]) == str(D["fingerprint"])
    return D, S["train_idx"], S["val_idx"], S["test_idx"]


def tagged(idx, y, Y):
    return idx[(y[idx] == 1) & (Y[idx].sum(1) > 0)]


def retrieval_block(E, y, Y, tr, va, te, cfg):
    """kNN detection, Acc@1, behaviour retrieval and tag prediction for embedding E."""
    ev = cfg["eval"]
    trva = np.concatenate([tr, va])
    out = {}
    s_val, _, _ = M.knn_scores(E[tr], y[tr], E[va], ev["knn_k"])
    s_te, nn_sim, I = M.knn_scores(E[trva], y[trva], E[te], ev["knn_k"])
    out["detection_knn"] = M.detection(y[te], s_te, y[va], s_val, tuple(ev["fpr_targets"]))
    out["detection_knn"]["accuracy@1"] = M.accuracy_at_1(y[trva], y[te], I)
    q = tagged(te, y, Y)
    out["behaviour_retrieval"] = {"n_queries": int(len(q)),
                                  **M.tag_overlap_retrieval(E[trva], Y[trva], E[q], Y[q], tuple(ev["k_values"]))}
    qv = tagged(va, y, Y)
    sv = M.knn_tag_scores(E[tr], Y[tr], E[qv], ev["knn_k"])
    thr = M.select_threshold(Y[qv], sv)
    st = M.knn_tag_scores(E[trva], Y[trva], E[q], ev["knn_k"])
    out["tags_knn"] = M.multilabel(Y[q], st, thr)
    return out, s_val, s_te, nn_sim


def eval_run(run_dir, cfg):
    O = dict(np.load(Path(run_dir) / "outputs.npz", allow_pickle=False))
    split = str(O["split"])
    D, tr, va, te = load(cfg, split)
    assert str(O["fingerprint"]) == str(D["fingerprint"]), "run was trained on a different dataset"
    y, Y, p, H, masks = D["y"], D["Y_type"], O["prob"], O["h"], O["masks"]
    fprs = tuple(cfg["eval"]["fpr_targets"])
    res = {"run": str(run_dir), "split": split}

    res["detection_head"] = M.detection(y[te], p[te], y[va], p[va], fprs)
    t = M.fit_temperature(y[va], p[va])
    pc = M.apply_temperature(p, t)
    res["calibration"] = {"temperature": t, "ece_raw": M.ece(y[te], p[te]), "ece_calibrated": M.ece(y[te], pc[te])}

    blk, s_val_knn, s_te_knn, nn_sim = retrieval_block(H, y, Y, tr, va, te, cfg)
    res.update(blk)

    # fused head + kNN (alpha chosen on validation PR-AUC)
    alphas = np.linspace(0, 1, 11)
    best_a = max(alphas, key=lambda a: M.detection(y[va], a * pc[va] + (1 - a) * s_val_knn, fprs=())["pr_auc"])
    res["detection_fused"] = M.detection(y[te], best_a * pc[te] + (1 - best_a) * s_te_knn,
                                         y[va], best_a * pc[va] + (1 - best_a) * s_val_knn, fprs)
    res["detection_fused"]["alpha"] = float(best_a)

    # triage: novelty = 1 - cosine similarity to nearest indexed sample
    novelty = 1 - nn_sim
    err = ((p[te] >= 0.5).astype(int) != y[te]).astype(int)
    res["triage"] = {"novelty_auroc_for_errors": float(M.roc_auc_score(err, novelty)) if 0 < err.sum() < len(err) else float("nan"),
                     **M.risk_coverage(y[te], p[te], novelty)}

    if "p_bin" in O:
        pats = {}
        for pat in sorted({tuple(r) for r in masks[te].astype(int)}, reverse=True):
            sel = te[(masks[te].astype(int) == np.array(pat)).all(1)]
            pats["".join(map(str, pat))] = {"n": int(len(sel)), "gate_bin": O["p_bin"][sel].mean(0).tolist(),
                                            "gate_retr": O["p_retr"][sel].mean(0).tolist()}
        res["gates"] = pats

    # single experts on their covered queries vs MoE on the same queries
    g = "ldp" if json.loads((Path(run_dir) / "log.json").read_text())["args"]["graph_expert"] == "ldp" else "n2v"
    matched = {}
    for name, E, m in (("MalConv", D["E_malconv"], D["m_malconv"]), ("SAFE", D["E_safe"], D["m_safe"]),
                       ("Graph", D[f"E_{g}"], D[f"m_{g}"])):
        cov = lambda idx: idx[m[idx] > 0]                                     # noqa: E731
        ctr, cva, cte = cov(tr), cov(va), cov(te)
        if min(len(ctr), len(cva), len(cte)) < 10:
            continue
        e_blk, _, _, _ = retrieval_block(E, y, Y, ctr, cva, cte, cfg)
        trva = np.concatenate([tr, va])
        _, _, I = M.knn_scores(H[trva], y[trva], H[cte], 1)
        q = tagged(cte, y, Y)
        matched[name] = {"n_queries": int(len(cte)),
                         "expert_acc@1": e_blk["detection_knn"]["accuracy@1"],
                         "moe_acc@1": M.accuracy_at_1(y[trva], y[cte], I),
                         "expert_tagP@1": e_blk["behaviour_retrieval"][1]["precision"],
                         "moe_tagP@1": M.tag_overlap_retrieval(H[trva], Y[trva], H[q], Y[q], (1,))[1]["precision"]}
    res["experts_matched"] = matched
    return res


def eval_preds(path, cfg, split):
    P = dict(np.load(path, allow_pickle=False))
    split = str(P["split"]) if "split" in P else split
    D, tr, va, te = load(cfg, split)
    assert "fingerprint" not in P or str(P["fingerprint"]) == str(D["fingerprint"]), "preds from another dataset"
    y, Y = D["y"], D["Y_type"]
    s = P["score"]
    ok = np.isfinite(s)
    coverage = float(ok[te].mean())
    tr, va, te = tr[ok[tr]], va[ok[va]], te[ok[te]]
    fprs = tuple(cfg["eval"]["fpr_targets"])
    res = {"preds": str(path), "split": split, "coverage_test": coverage,
           "detection": M.detection(y[te], s[te], y[va], s[va], fprs)}
    if "tag_scores" in P:
        T = P["tag_scores"]
        qv, qt = tagged(va, y, Y), tagged(te, y, Y)
        thr = M.select_threshold(Y[qv], T[qv])
        res["tags_head"] = M.multilabel(Y[qt], T[qt], thr)
    if "emb" in P:
        blk, _, _, _ = retrieval_block(P["emb"], y, Y, tr, va, te, cfg)
        res.update(blk)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--preds")
    ap.add_argument("--split", default=None)
    args = ap.parse_args()
    cfg = load_config()
    split = args.split or cfg["dataset"]["split_mode"]
    if args.run:
        res, out = eval_run(args.run, cfg), Path(args.run) / "eval.json"
    else:
        res, out = eval_preds(args.preds, cfg, split), Path(args.preds).with_name("eval.json")
    out.write_text(json.dumps(res, indent=1, default=float))
    d = res.get("detection_head", res.get("detection"))
    print(f"{out}: acc {d['accuracy']:.4f} auc {d['roc_auc']:.4f} "
          + " ".join(f"{k} {v:.3f}" for k, v in d.items() if k.startswith("tpr@")))


if __name__ == "__main__":
    main()
