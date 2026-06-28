import numpy as np
import faiss
import json

index = faiss.read_index("faiss_index.bin")

emb = np.load("all_embeddings.npy").astype("float32")

with open("metadata.json") as f:
    meta = json.load(f)

query = emb[0].reshape(1,-1)

D, I = index.search(query, 5)

print("Top 5 neighbors:")

for i in range(5):
    idx = str(I[0][i])
    m = meta[idx]

    print(
        i+1,
        "family:", m["family"],
        "distance:", D[0][i],
        "sha256:", m["sha256"][:12]
    )