import os, sys, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)))
env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")
log = "multilabel_resultsk20.txt"
with open(log, "w") as f:
    f.write("=== Multi-label behaviour prediction (dropout 0.3, 3 seeds) ===\n")
for s in [42, 43, 44]:
    with open(log, "a") as f:
        f.write(f"\n\n{'#'*60}\n### MODEL_SEED = {s}\n{'#'*60}\n\n")
        f.flush()
        subprocess.run([sys.executable, "evaluate_experts_moe_files.py"],
                       stdout=f, stderr=subprocess.STDOUT,
                       env=dict(env, MOE_CKPT=f"best_moe_s{s}.pt"))
        f.flush()
print(f"done -> {log}")
