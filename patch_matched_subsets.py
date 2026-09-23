"""
patch_matched_subsets.py
Adds a fair comparison to Experiment 2: the MoE scored on EXACTLY the same
queries each expert can embed (SAFE's 345, Node2Vec's 279, MalConv's 393).

Run once from moepaper/:   python patch_matched_subsets.py
Then:                      python run_final_evals.py
Backups: *.bak3. Safe to re-run.
"""
import os, sys, shutil, ast

os.chdir(os.path.dirname(os.path.abspath(__file__)))
EVAL, RUN = "evaluate_experts_moe_files.py", "run_final_evals.py"
MARK = "# === MATCHED SUBSETS (inserted) ==="

EVAL_ANCHOR = ("            print(f\"   {k:3d}      {r[k]['precision']:.4f}   {r[k]['ndcg']:.4f}\")\n"
               "    return out\n")
EVAL_NEW = ("            print(f\"   {k:3d}      {r[k]['precision']:.4f}   {r[k]['ndcg']:.4f}\")\n"
            "    " + MARK + "\n"
            "    for name, E, m in experts:\n"
            "        qi = tagged[m[tagged].astype(bool)]\n"
            "        r = _tag_overlap_retrieval(H_all, index_idx, qi, Y_type, ks)\n"
            "        out[f\"MoE@{name}\"] = {\"n_queries\": int(len(qi)), \"metrics\": r}\n"
            "        print(f\"\\n  MoE on {name}-covered queries ({len(qi)})\")\n"
            "        for k in ks:\n"
            "            print(f\"   {k:3d}      {r[k]['precision']:.4f}   {r[k]['ndcg']:.4f}\")\n"
            "    return out\n")

RUN_ANCHOR = 'L.append("Query counts: " + ", ".join(f"{nm} {t[0][nm][\'n_queries\']}" for nm in names))\n'
RUN_NEW = RUN_ANCHOR + '''
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
'''


def patch(path, anchor, new, label):
    src = open(path).read()
    if MARK in src or "Table 10b" in src:
        print(f"  {label}: already patched"); return
    if src.count(anchor) != 1:
        sys.exit(f"ABORT: anchor not found in {path} ({label}) — send me the file")
    shutil.copy(path, path + ".bak3")
    new_src = src.replace(anchor, new, 1)
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        sys.exit(f"ABORT: syntax error patching {path}: {e} (file untouched)")
    open(path, "w").write(new_src)
    print(f"  {label}: patched (backup {path}.bak3)")


patch(EVAL, EVAL_ANCHOR, EVAL_NEW, "eval script")
patch(RUN, RUN_ANCHOR, RUN_NEW, "runner summary")
print("\nDone. Now run:  python run_final_evals.py")
