import numpy as np
import pandas as pd

# TRY to import faiss if available
try:
    import faiss
    USE_FAISS = True
except ImportError:
    print("WARNING: faiss is not installed. Will only prepare numpy arrays, no FAISS index will be built.")
    USE_FAISS = False

# 1. Load your CSV
csv_path = "dataset_emberSorel_reduced_features.csv"
df = pd.read_csv(csv_path)

print("Loaded CSV with shape:", df.shape)
print("Columns:", list(df.columns))

# 2. Try to detect id/label columns (adapt these names if yours are different)
possible_id_cols = ["sha256", "hash", "id"]
possible_label_cols = ["family", "label", "y", "class"]

id_col = next((c for c in possible_id_cols if c in df.columns), None)
label_col = next((c for c in possible_label_cols if c in df.columns), None)

if id_col is None:
    print("WARNING: No sha256/hash/id column found automatically.")
    # If you know the exact name, set it here manually, e.g.:
    # id_col = "sha256"

if label_col is None:
    print("WARNING: No family/label column found automatically.")
    # If you know the exact name, set it here manually, e.g.:
    # label_col = "family"

# 3. Decide which columns are features/embeddings
non_feature_cols = []
if id_col is not None:
    non_feature_cols.append(id_col)
if label_col is not None:
    non_feature_cols.append(label_col)

feature_cols = [c for c in df.columns if c not in non_feature_cols]

print("Using these columns as features/embeddings:")
print(feature_cols[:10], "..." if len(feature_cols) > 10 else "")

# 4. Extract arrays
if id_col is not None:
    sha256_all = df[id_col].to_numpy()
else:
    sha256_all = np.arange(len(df))  # dummy ids if none

if label_col is not None:
    family_all = df[label_col].to_numpy()
else:
    family_all = np.array(["unknown"] * len(df))

embeddings = np.ascontiguousarray(df[feature_cols].to_numpy().astype("float32"))
print("Embeddings shape:", embeddings.shape)

# Optionally L2 normalize here (for FAISS or your own L2)
if USE_FAISS:
    faiss.normalize_L2(embeddings)

# 5. Save numpy versions for later use
np.save("all_embeddings.npy", embeddings)
np.save("all_sha256.npy", sha256_all)
np.save("all_family.npy", family_all)

print("Saved all_embeddings.npy, all_sha256.npy, all_family.npy")

# 6. If FAISS is available, build and save index
if USE_FAISS:
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, "faiss_index.bin")
    
    # Use joblib to save metadata
    import joblib
    joblib.dump(
        {"sha256": sha256_all, "family": family_all},
        "faiss_meta.pkl"
    )
    print("FAISS index and metadata saved: faiss_index.bin, faiss_meta.pkl")
else:
    print("Skipped FAISS index build because faiss is not installed.")
