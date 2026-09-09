import os, re, sys, shutil, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)))
dropouts   = [0.1, 0.2, 0.3, 0.4]
train_file = "moe_train.py"
eval_file  = "evaluate_experts_moe_files.py"
final_log  = "final_dropout_evaluations_set2.txt"

env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")

def patch(path, pattern, replacement, label):
    src = open(path).read()
    new, n = re.subn(pattern, replacement, src,count=1)
    if n == 0:
        sys.exit(f"ABORT: no match for {label} in {path}")
    open(path, "w").write(new)
    print(f"  set {label} in {path}")

# force val-split evaluation for the whole sweep
patch(eval_file, r'EVAL_SPLIT\s*=\s*["\'](val|test)["\']',
      'EVAL_SPLIT = "val"', 'EVAL_SPLIT="val"')

with open(final_log, "w") as f:
    f.write("=== MoE Dropout Ablation Study (selection on VAL) ===\n")

for d in dropouts:
    print(f"\n{'='*50}\nEXPERT_DROPOUT = {d}\n{'='*50}")

    patch(train_file, r'EXPERT_DROPOUT\s*=\s*[0-9.]+',
          f'EXPERT_DROPOUT = {d}', f'EXPERT_DROPOUT={d}')

    # remove stale checkpoint so a crash can't leave the previous model behind
    if os.path.exists("best_moe.pt"):
        os.remove("best_moe.pt")

    train_log = f"moe_train_dropout_{int(d*100)}.txt"
    print(f"  training -> {train_log}")
    with open(train_log, "w") as f:
        r = subprocess.run([sys.executable, train_file], stdout=f,
                           stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0 or not os.path.exists("best_moe.pt"):
        sys.exit(f"ABORT: training failed at dropout {d} — see {train_log}")

    ckpt = f"best_moe_{int(d*100)}.pt"
    shutil.copy("best_moe.pt", ckpt)

    with open(final_log, "a") as f:
        f.write(f"\n\n{'#'*60}\n### EXPERT_DROPOUT = {d}  (VAL split)\n{'#'*60}\n\n")
        f.flush()
        r = subprocess.run([sys.executable, eval_file], stdout=f,
                           stderr=subprocess.STDOUT, env=dict(env, MOE_CKPT=ckpt))
        f.flush()
    if r.returncode != 0:
        sys.exit(f"ABORT: evaluation failed at dropout {d} — see {final_log}")

    print(f"  done: {d}")

print(f"\nSweep complete. Compare val nDCG@10 in {final_log}, pick the winner,")
print("then set EVAL_SPLIT='test' and evaluate ONLY that checkpoint.")