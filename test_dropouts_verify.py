import os, sys, shutil, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__)))
eval_file  = "evaluate_experts_moe_files.py"
env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")
for tag in [10,20,30,40]:
    shutil.copy(f"best_moe_{tag}.pt", "best_moe.pt")
    subprocess.run([sys.executable, eval_file], env=env)