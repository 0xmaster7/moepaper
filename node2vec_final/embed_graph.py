import os
import sys
import numpy as np
import subprocess
from load_graphs import load_cfg
from generate_node_embeddings import generate_node_embeddings
from generate_functional_embeddings import generate_functional_embedding

def embed_binary(binary_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(binary_path)
    dot_path = os.path.join(output_dir, f"{base_name}.dot")
    npy_path = os.path.join(output_dir, f"{base_name}.npy")
    
    print(f"[*] Analyzing binary with radare2: {binary_path}")
    # Run radare2 to extract the full Call Graph into a .dot file
    try:
        subprocess.run(f"r2 -e scr.color=false -A -q -c 'agCd' \"{binary_path}\" > \"{dot_path}\"", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Failed to run radare2 on {binary_path}: {e}")
        return
        
    print("[*] Loading graph...")
    G = load_cfg(dot_path)
    if G is None:
        print("[!] Graph could not be loaded or is empty.")
        return
        
    print("[*] Generating Node2Vec embeddings (this may take a minute)...")
    node_embeddings = generate_node_embeddings(G)
    if not node_embeddings:
        print("[!] Failed to generate node embeddings.")
        return
        
    print("[*] Generating Functional embeddings...")
    graph_embedding = generate_functional_embedding(node_embeddings)
    if graph_embedding is None:
        print("[!] Failed to generate functional embedding.")
        return
        
    np.save(npy_path, graph_embedding)
    print(f"[✓] Saved final embedding to {npy_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python embed_graph.py <binary_path> <output_dir>")
        sys.exit(1)
        
    binary_path = sys.argv[1]
    output_dir = sys.argv[2]
    embed_binary(binary_path, output_dir)
