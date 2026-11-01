"""Train + evaluate every MoE variant and baseline over all seeds. Resumable:
anything with an eval.json is skipped.

    python ablations/run_ablations.py --sweep-dropout     # 1) pick expert dropout on VAL (seed 42)
    python ablations/run_ablations.py                     # 2) full grid x seeds + baselines
    python ablations/run_ablations.py --only moe_full abl_concat
    python report/make_tables.py                          # 3) tables

The dropout sweep reads VALIDATION outputs only; put the chosen value in
config.yaml (train.expert_dropout) before step 2.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import load_config                       # noqa: E402
from eval.metrics import detection                          # noqa: E402

PY = sys.executable
VARIANTS = {
    "moe_full":             [],
    "abl_single_gate":      ["--fusion", "single_gate"],
    "abl_concat":           ["--fusion", "concat"],
    "abl_mean":             ["--fusion", "mean"],
    "abl_gate_masks_only":  ["--gate-input", "masks"],
    "abl_gate_ember_only":  ["--gate-input", "ember"],
    "abl_no_expert_dropout": ["--expert-dropout", "0"],
    "abl_graph_node2vec":   ["--graph-expert", "node2vec"],
    "expert_malconv":       ["--fusion", "expert:0"],
    "expert_safe":          ["--fusion", "expert:1"],
    "expert_graph":         ["--fusion", "expert:2"],
}
BASELINES = {
    "ember_lgbm": ["baselines/ember_lgbm.py", "--tags"],
    "sorel_ffnn": ["baselines/sorel_ffnn.py"],
    "cfg_gnn":    ["baselines/cfg_gnn.py"],
}


def sh(args):
    print("+", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def val_prauc(run_dir, cfg):
    d = Path(cfg["dataset"]["local_dir"])
    D = np.load(d / "dataset.npz")
    O = np.load(Path(run_dir) / "outputs.npz")
    va = np.load(d / f"splits_{str(O['split'])}.npz")["val_idx"]
    return detection(D["y"][va], O["prob"][va], fprs=(0.01,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dropout", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed to train_moe (e.g. --epochs 2)")
    args = ap.parse_args()
    cfg = load_config()
    runs = ROOT / "results" / "runs"

    if args.sweep_dropout:
        res = {}
        for d in (0.0, 0.1, 0.2, 0.3, 0.4):
            name = f"sweep_dropout_{d}"
            rd = runs / name / "seed42"
            if not (rd / "outputs.npz").exists():
                sh(["model/train_moe.py", "--name", name, "--seed", "42", "--expert-dropout", str(d), *args.extra])
            m = val_prauc(rd, cfg)
            res[d] = {"val_pr_auc": m["pr_auc"], "val_roc_auc": m["roc_auc"], "val_tpr@fpr=0.01": m["tpr@fpr=0.01"]}
            print(d, res[d])
        best = max(res, key=lambda k: res[k]["val_pr_auc"])
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / "dropout_sweep_val.json").write_text(json.dumps({"results": res, "best": best}, indent=1))
        print(f"\nbest expert dropout on VAL: {best}  -> set train.expert_dropout in config.yaml")
        return

    for name, extra in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        for seed in cfg["train"]["seeds"]:
            rd = runs / name / f"seed{seed}"
            if (rd / "eval.json").exists():
                continue
            if not (rd / "outputs.npz").exists():
                sh(["model/train_moe.py", "--name", name, "--seed", str(seed), *extra, *args.extra])
            sh(["eval/evaluate.py", "--run", str(rd)])

    if not args.skip_baselines:
        for name, cmd in BASELINES.items():
            if args.only and name not in args.only:
                continue
            if name == "cfg_gnn" and not (Path(cfg["dataset"]["local_dir"]) / "graphs.npz").exists():
                print("skip cfg_gnn: run build/build_dataset.py --graphs first")
                continue
            for seed in cfg["train"]["seeds"]:
                p = ROOT / "results" / "baselines" / name / f"seed{seed}" / "preds.npz"
                if p.with_name("eval.json").exists():
                    continue
                if not p.exists():
                    sh([*cmd, "--seed", str(seed)])
                sh(["eval/evaluate.py", "--preds", str(p)])
        p = ROOT / "results" / "baselines" / "malconv2_zeroshot" / "seed0" / "preds.npz"
        if not p.with_name("eval.json").exists():
            sh(["baselines/malconv_zeroshot.py"])
            sh(["eval/evaluate.py", "--preds", str(p)])


if __name__ == "__main__":
    main()
