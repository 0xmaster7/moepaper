"""Collect every eval.json into paper-ready markdown tables (mean +/- std over seeds).

    python report/make_tables.py            # -> results/tables.md
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def collect(kind):
    out = defaultdict(list)
    for p in sorted((RES / kind).glob("*/seed*/eval.json")):
        out[p.parent.parent.name].append(json.loads(p.read_text()))
    return out


def get(d, path):
    for k in path:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def ms(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "-"
    return f"{np.mean(v):.4f}" if len(v) == 1 else f"{np.mean(v):.4f} ± {np.std(v):.4f}"


def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    runs, base = collect("runs"), collect("baselines")
    md = ["# Results (test split; selection on validation)\n"]
    DET = [("accuracy",), ("roc_auc",), ("pr_auc",), ("tpr@fpr=0.001",), ("tpr@fpr=0.01",), ("f1",)]

    # ---- detection
    rows = []
    for name, label in (("malconv2_zeroshot", "MalConv2-GCT (released weights, no training)"),
                        ("ember_lgbm", "EMBER + LightGBM"), ("sorel_ffnn", "SOREL-20M FFNN"),
                        ("cfg_gnn", "Call-graph GIN (covered subset)")):
        if name in base:
            ev = base[name]
            rows.append([label] + [ms([get(e, ("detection",) + k) for e in ev]) for k in DET]
                        + [ms([e.get("coverage_test") for e in ev])])
    if "moe_full" in runs:
        for blk, label in (("detection_head", "MoE (classifier head)"), ("detection_knn", "MoE (FAISS kNN)"),
                           ("detection_fused", "MoE (head + kNN)")):
            rows.append([label] + [ms([get(e, (blk,) + k) for e in runs["moe_full"]]) for k in DET] + ["1.000"])
    md += ["## Detection\n", table(rows, ["Model", "Acc", "ROC-AUC", "PR-AUC", "TPR@FPR=1e-3", "TPR@FPR=1e-2",
                                          "F1", "Test coverage"]),
           "\n_TPR@1e-3 is only meaningful with >= 10,000 benign test samples (see `reliable@` flags in eval.json)._\n"]

    # ---- behaviour tags
    rows = []
    for name, blk, label in (("sorel_ffnn", "tags_head", "SOREL FFNN tag head"),
                             ("ember_lgbm", "tags_head", "LightGBM per-tag"),
                             ("cfg_gnn", "tags_head", "GIN tag head"),
                             ("sorel_ffnn", "tags_knn", "kNN on FFNN embedding"),
                             ("cfg_gnn", "tags_knn", "kNN on GIN embedding"),
                             ("malconv2_zeroshot", "tags_knn", "kNN on MalConv2 embedding")):
        if name in base and any(blk in e for e in base[name]):
            rows.append([label] + [ms([get(e, (blk, k)) for e in base[name]]) for k in ("micro_f1", "macro_f1", "mAP")])
    if "moe_full" in runs:
        rows.append(["MoE kNN tag voting"] + [ms([get(e, ("tags_knn", k)) for e in runs["moe_full"]])
                                              for k in ("micro_f1", "macro_f1", "mAP")])
    md += ["\n## Behaviour-tag prediction (tagged test malware)\n", table(rows, ["Model", "micro-F1", "macro-F1", "mAP"])]

    # ---- retrieval
    rows = []
    srcs = [(n, base[n], lbl) for n, lbl in (("malconv2_zeroshot", "MalConv2 embedding"),
                                             ("sorel_ffnn", "FFNN embedding"), ("cfg_gnn", "GIN embedding")) if n in base]
    if "moe_full" in runs:
        srcs.append(("moe_full", runs["moe_full"], "MoE h (gate_retr)"))
    for _, ev, lbl in srcs:
        if not any("behaviour_retrieval" in e for e in ev):
            continue
        rows.append([lbl, ms([get(e, ("detection_knn", "accuracy@1")) for e in ev])]
                    + [ms([get(e, ("behaviour_retrieval", str(k), m)) or get(e, ("behaviour_retrieval", k, m))
                           for e in ev]) for k, m in ((1, "precision"), (10, "precision"), (10, "ndcg"))])
    md += ["\n## Retrieval\n", table(rows, ["Embedding", "Acc@1 (mal/ben)", "Tag P@1", "Tag P@10", "Tag nDCG@10"])]

    # ---- ablations
    rows = []
    for name in sorted(runs):
        if name.startswith("sweep"):
            continue
        ev = runs[name]
        rows.append([name, str(len(ev))]
                    + [ms([get(e, ("detection_head", k)) for e in ev]) for k in ("roc_auc", "tpr@fpr=0.01")]
                    + [ms([get(e, ("detection_knn", "accuracy@1")) for e in ev]),
                       ms([get(e, ("behaviour_retrieval", "1", "precision")) or get(e, ("behaviour_retrieval", 1, "precision")) for e in ev]),
                       ms([get(e, ("tags_knn", "micro_f1")) for e in ev])])
    md += ["\n## Ablations\n", table(rows, ["Variant", "seeds", "Head ROC-AUC", "Head TPR@1e-2", "kNN Acc@1",
                                            "Tag P@1", "Tag micro-F1"])]

    # ---- triage + gates (seed 42 of the full model)
    full = next((e for e in runs.get("moe_full", []) if e["run"].endswith("seed42")), None)
    if full:
        tri = full["triage"]
        rows = [[k, str(v["n"]), f"{v['accuracy']:.4f}", f"{v['malware_recall']:.4f}", f"{v['benign_fpr']:.4f}"]
                for k, v in tri.items() if k.startswith("cov=")]
        md += ["\n## Triage: abstain on the most novel samples (send to sandbox)\n",
               f"Novelty (1 - NN cosine) AUROC for predicting head errors: {tri['novelty_auroc_for_errors']:.4f}\n",
               table(rows, ["Coverage", "n kept", "Accuracy", "Malware recall", "Benign FPR"])]
        if "gates" in full:
            rows = [[p, str(v["n"]), ", ".join(f"{x:.3f}" for x in v["gate_bin"]),
                     ", ".join(f"{x:.3f}" for x in v["gate_retr"])] for p, v in full["gates"].items()]
            md += ["\n## Gate weights by availability (MalConv, SAFE, Graph)\n",
                   table(rows, ["Pattern", "n", "gate_bin", "gate_retr"])]
        if full.get("experts_matched"):
            rows = [[k, str(v["n_queries"]), f"{v['expert_acc@1']:.4f}", f"{v['moe_acc@1']:.4f}",
                     f"{v['expert_tagP@1']:.4f}", f"{v['moe_tagP@1']:.4f}"] for k, v in full["experts_matched"].items()]
            md += ["\n## Single expert vs MoE on the expert's covered test queries\n",
                   table(rows, ["Expert", "n", "Expert Acc@1", "MoE Acc@1", "Expert tag P@1", "MoE tag P@1"])]
    sys.path.insert(0, str(ROOT))
    from common.config import load_config
    cov = Path(load_config()["dataset"]["local_dir"]) / "coverage.md"
    if cov.exists():
        md += ["\n## Expert coverage\n", cov.read_text()]
    RES.mkdir(exist_ok=True)
    (RES / "tables.md").write_text("\n".join(md) + "\n")
    print((RES / "tables.md").read_text())


if __name__ == "__main__":
    sys.exit(main())
