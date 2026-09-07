"""
Seed-variance experiment.

EXPERT_DROPOUT is FIXED at 0.2 (selected on the validation partition).
Only MODEL_SEED varies. The data split is generated once and reused, so the
spread across runs measures model variance, not test-set difficulty.

Run this AFTER the dropout sweep. Evaluation here uses the TEST split — this
is the single, final measurement, so do not re-run it to "improve" the number.
"""
import os, re, sys, shutil, subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # paths are relative to this file

seeds       = [42, 43, 44]
DROPOUT     = 0.2                      # fixed: chosen on val
train_file  = "moe_train.py"
eval_file   = "evaluate_experts_moe_files.py"
final_log   = "final_seed_evaluations.txt"

env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")


def patch(path, pattern, replacement, label):
    src = open(path).read()
    new, n = re.subn(pattern, replacement, src, count=1)
    if n == 0:
        sys.exit(f"ABORT: no match for {label} in {path}")
    open(path, "w").write(new)
    print(f"  set {label}")


# --- fixed configuration, applied once -------------------------------------
patch(train_file, r'EXPERT_DROPOUT\s*=\s*[0-9.]+',
      f'EXPERT_DROPOUT = {DROPOUT}', f'EXPERT_DROPOUT={DROPOUT}')
patch(eval_file, r'EVAL_SPLIT\s*=\s*["\'](val|test)["\']',
      'EVAL_SPLIT = "test"', 'EVAL_SPLIT="test"')

# --- the split must already exist and must not change ----------------------
if not os.path.exists("splits.npz"):
    print("splits.npz not found — the first training run will create it.")
else:
    print("splits.npz found — will be reused unchanged across all seeds.")
split_mtime = os.path.getmtime("splits.npz") if os.path.exists("splits.npz") else None

with open(final_log, "w") as f:
    f.write(f"=== MoE Seed Variance (EXPERT_DROPOUT={DROPOUT}, TEST split) ===\n")

for s in seeds:
    print(f"\n{'='*50}\nMODEL_SEED = {s}  (dropout {DROPOUT})\n{'='*50}")

    patch(train_file, r'(?m)^MODEL_SEED\s*=\s*\d+',
          f'MODEL_SEED = {s}', f'MODEL_SEED={s}')
    # guard: SPLIT_SEED must never be touched
    if not re.search(r'^SPLIT_SEED\s*=\s*42', open(train_file).read(), re.M):
        sys.exit("ABORT: SPLIT_SEED is not 42 — the split would move between runs.")

    if os.path.exists("best_moe.pt"):
        os.remove("best_moe.pt")          # a crash must not leave the previous model

    train_log = f"moe_train_seed_{s}.txt"
    print(f"  training -> {train_log}")
    with open(train_log, "w") as f:
        r = subprocess.run([sys.executable, train_file], stdout=f,
                           stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0 or not os.path.exists("best_moe.pt"):
        sys.exit(f"ABORT: training failed at seed {s} — see {train_log}")

    # the split file must be byte-stable across runs
    if split_mtime is None:
        split_mtime = os.path.getmtime("splits.npz")
        print("  splits.npz created by this run; later seeds will reuse it.")
    elif os.path.getmtime("splits.npz") != split_mtime:
        sys.exit("ABORT: splits.npz changed between runs — seeds are not comparable.")

    ckpt = f"best_moe_s{s}.pt"
    shutil.copy("best_moe.pt", ckpt)
    print(f"  checkpoint -> {ckpt}")

    with open(final_log, "a") as f:
        f.write(f"\n\n{'#'*60}\n### MODEL_SEED = {s}   (dropout {DROPOUT}, TEST split)\n{'#'*60}\n\n")
        f.flush()                          # header must reach disk before the child writes
        r = subprocess.run([sys.executable, eval_file], stdout=f,
                           stderr=subprocess.STDOUT, env=env)
        f.flush()
    if r.returncode != 0:
        sys.exit(f"ABORT: evaluation failed at seed {s} — see {final_log}")

    print(f"  done: seed {s}")

# --- collect the headline metric -------------------------------------------
print(f"\n{'='*50}\nAll seeds complete.\n{'='*50}")
try:
    txt = open(final_log).read()
    blocks = txt.split("### MODEL_SEED = ")[1:]
    vals = []
    for b in blocks:
        seed = b.split()[0]
        moe = b.split("MoE — Gated soft fusion")[-1]
        row10 = [l for l in moe.splitlines() if l.strip().startswith("10")]
        if row10:
            ndcg = float(row10[0].split()[2])
            vals.append((seed, ndcg))
            print(f"  seed {seed}: nDCG@10 = {ndcg:.4f}")
    if len(vals) > 1:
        import statistics
        xs = [v for _, v in vals]
        print(f"\n  mean {statistics.mean(xs):.4f}  std {statistics.pstdev(xs):.4f}")
        print(f"  report as: {statistics.mean(xs):.4f} ± {statistics.pstdev(xs):.4f}")
except Exception as e:
    print(f"  (could not auto-parse metrics: {e} — read {final_log} directly)")

print(f"\nFull output: {final_log}")
