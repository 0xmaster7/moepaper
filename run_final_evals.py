"""
run_final_evals.py
Runs the three remaining experiments on the three seed checkpoints (dropout 0.3)
and prints paper-ready mean +/- std tables.

Prerequisite: python install_final_evals.py   (once)
Run:          python run_final_evals.py
Evaluation only - nothing is retrained.

Outputs:
  final_evals_log.txt                full console output of all three runs
  final_evals_best_moe_s{42,43,44}.json   per-seed results
  final_evals_summary.md             aggregated tables to paste into the paper
"""
import os, re, sys, json, subprocess, statistics as st

os.chdir(os.path.dirname(os.path.abspath(__file__)))
EVAL = "evaluate_experts_moe_files.py"
SEEDS = [42, 43, 44]
LOG = "final_evals_log.txt"
env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")

src = open(EVAL).read()
if "# === FINAL EVALS (inserted) ===" not in src:
    sys.exit("ABORT: run install_final_evals.py first")
src2, n = re.subn(r'EVAL_SPLIT\s*=\s*["\'](val|test)["\']', 'EVAL_SPLIT = "test"', src, count=1)
if n == 0:
    sys.exit("ABORT: EVAL_SPLIT not found in eval script")
open(EVAL, "w").write(src2)

for s in SEEDS:
    if not os.path.exists(f"best_moe_s{s}.pt"):
        sys.exit(f"ABORT: best_moe_s{s}.pt missing")

with open(LOG, "w") as f:
    f.write("=== Final evaluations: dropout 0.3, TEST split ===\n")
for s in SEEDS:
    ckpt = f"best_moe_s{s}.pt"
    print(f"evaluating {ckpt} ...", flush=True)
    with open(LOG, "a") as f:
        f.write(f"\n\n{'#' * 60}\n### MODEL_SEED = {s}  ({ckpt})\n{'#' * 60}\n\n")
        f.flush()
        r = subprocess.run([sys.executable, EVAL], stdout=f, stderr=subprocess.STDOUT,
                           env=dict(env, MOE_CKPT=ckpt))
        f.flush()
    if r.returncode != 0:
        sys.exit(f"ABORT: evaluation failed for {ckpt} - see {LOG}")

# ---------------------------------------------------------------- aggregate
R = []
for s in SEEDS:
    p = f"final_evals_best_moe_s{s}.json"
    if not os.path.exists(p):
        sys.exit(f"ABORT: {p} not written - see {LOG}")
    R.append(json.load(open(p)))

def ms(xs):
    return f"{st.mean(xs):.4f} ± {st.pstdev(xs):.4f}"

L = ["# Final evaluation summary", "",
     "Dropout 0.3, three model seeds (42/43/44), test partition. Mean ± std.", ""]

b = [r["binary_detection"] for r in R]
L += ["## Table 9. Binary detection (gate_bin + classifier head)", "",
      f"n = {b[0]['n']} ({b[0]['n_malware']} malware). Threshold 0.5. "
      f"All-benign baseline accuracy = {b[0]['majority_baseline_acc']:.4f}.", "",
      "| Metric | Value |", "|---|---|"]
for key, lbl in [("accuracy", "Accuracy"), ("precision_malware", "Precision (malware)"),
                 ("recall_malware", "Recall (malware)"), ("f1_malware", "F1 (malware)"),
                 ("roc_auc", "ROC-AUC")]:
    L.append(f"| {lbl} | {ms([x[key] for x in b])} |")

L += ["", "## Table 10. Behaviour-level retrieval (relevant = shares ≥1 tag)", "",
      "Queries: tagged malware in test. Experts evaluated on covered queries only.", "",
      "| k | Metric | SAFE | MalConv | Node2Vec | MoE |", "|---|---|---|---|---|---|"]
names = ["SAFE", "MalConv", "Node2Vec", "MoE"]
t = [r["tag_overlap_retrieval"] for r in R]
for k in ["1", "3", "5", "10"]:
    for met, lbl in [("precision", "Precision"), ("ndcg", "nDCG")]:
        cells = []
        for nm in names:
            vals = [x[nm]["metrics"][k][met] for x in t]
            cells.append(f"{vals[0]:.4f}" if nm != "MoE" else f"**{ms(vals)}**")
        L.append(f"| {k if met == 'precision' else ''} | {lbl} | " + " | ".join(cells) + " |")
L.append("")
L.append("Query counts: " + ", ".join(f"{nm} {t[0][nm]['n_queries']}" for nm in names))

L += ["", "## Table 10b. Matched comparison — MoE on each expert's own queries", "",
      "Same query set for both columns in each row. Expert = single run; MoE = mean ± std.", "",
      "| Subset | n | k | Expert P | MoE P | Expert nDCG | MoE nDCG |",
      "|---|---|---|---|---|---|---|"]
for nm in ["SAFE", "MalConv", "Node2Vec"]:
    if f"MoE@{nm}" not in t[0]:
        continue
    n = t[0][f"MoE@{nm}"]["n_queries"]
    for k in ["1", "10"]:
        eP = t[0][nm]["metrics"][k]["precision"]; eN = t[0][nm]["metrics"][k]["ndcg"]
        mP = ms([x[f"MoE@{nm}"]["metrics"][k]["precision"] for x in t])
        mN = ms([x[f"MoE@{nm}"]["metrics"][k]["ndcg"] for x in t])
        L.append(f"| {nm} | {n} | {k} | {eP:.4f} | {mP} | {eN:.4f} | {mN} |")

L += ["", "## Table 11. Mean gate weights by availability pattern (seed 42)", "",
      "Pattern = (MalConv, SAFE, Node2Vec) availability.", "",
      "| Pattern | n | gate_bin [Mal, SAFE, N2V] | gate_retr [Mal, SAFE, N2V] |", "|---|---|---|---|"]
for pat, v in R[0]["gate_weights"]["by_pattern"].items():
    fb = ", ".join(f"{x:.3f}" for x in v["gate_bin"])
    fr = ", ".join(f"{x:.3f}" for x in v["gate_retr"])
    L.append(f"| {pat} | {v['n']} | {fb} | {fr} |")

ex = R[0]["gate_weights"]["example"]
if ex:
    L += ["", "## Figure 5(c) data. Worked example, SAFE missing (seed 42)", "",
          "| | MalConv | SAFE | Node2Vec |", "|---|---|---|---|"]
    for key, lbl in [("mask", "mask"), ("gate_bin_raw", "gate_bin raw"),
                     ("gate_bin_renorm", "gate_bin renormalised"),
                     ("gate_retr_raw", "gate_retr raw"),
                     ("gate_retr_renorm", "gate_retr renormalised")]:
        L.append(f"| {lbl} | " + " | ".join(f"{x:.3f}" for x in ex[key]) + " |")

open("final_evals_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("\nsaved -> final_evals_summary.md")
