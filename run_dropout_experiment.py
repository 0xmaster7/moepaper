import os
import re
import subprocess
import shutil

dropouts = [0.1, 0.2, 0.3, 0.4]
train_file = "moe_train.py"
eval_file = "evaluate_experts_moe.py"
final_log = "final_dropout_evaluations.txt"

# Clear the final log if it exists
with open(final_log, "w") as f:
    f.write("=== MoE Dropout Ablation Study ===\n\n")

for d in dropouts:
    print(f"\n{'='*50}")
    print(f"Starting experiment for EXPERT_DROPOUT = {d}")
    print(f"{'='*50}")
    
    # 1. Modify the training script
    with open(train_file, "r") as f:
        content = f.read()
    
    # Use regex to find and replace the EXPERT_DROPOUT assignment
    new_content = re.sub(r'EXPERT_DROPOUT\s*=\s*[0-9.]+', f'EXPERT_DROPOUT = {d}', content)
    
    with open(train_file, "w") as f:
        f.write(new_content)
        
    print(f"Successfully updated {train_file} to use EXPERT_DROPOUT = {d}")
    
    # 2. Run the training script
    train_log = f"moe_train_dropout_{int(d*100)}.txt"
    print(f"Training model... (This will take ~50 epochs. Logs saved to {train_log})")
    with open(train_log, "w") as f:
        subprocess.run(["python3", train_file], stdout=f, stderr=subprocess.STDOUT)
    print("Training finished!")
    
    # Backup the checkpoint so it isn't overwritten!
    backup_ckpt = f"best_moe_{int(d*100)}.pt"
    if os.path.exists("best_moe.pt"):
        shutil.copy("best_moe.pt", backup_ckpt)
        print(f"Backed up checkpoint to {backup_ckpt}")
    
    # 3. Append a header to the final log
    with open(final_log, "a") as f:
        f.write(f"\n\n{'#'*60}\n")
        f.write(f"### EVALUATION RESULTS FOR EXPERT_DROPOUT = {d}\n")
        f.write(f"{'#'*60}\n\n")
    
    # 4. Run the evaluation script and append to final log
    print(f"Evaluating model... (Appending to {final_log})")
    with open(final_log, "a") as f:
        subprocess.run(["python3", eval_file], stdout=f, stderr=subprocess.STDOUT)
    print(f"Evaluation finished for dropout {d}!")

print("\n\nAll experiments completed successfully! Please check final_dropout_evaluations.txt")
