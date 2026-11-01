"""Stage B: SAFE function embeddings from the token ids stored by Stage A.

Each function -> SAFE (frozen safe.pb, 100-d, L2-normalised) -> mean over the
binary's functions (same pooling as your original pipeline). Runs on any CPU
box (GPU optional) and only reads/writes small parquet files on S3.

    python embed/embed_safe.py                    # all feature chunks not yet embedded
    python embed/embed_safe.py --local-in out/features_local.parquet --local-out out/
    python embed/embed_safe.py ... --mock         # no TensorFlow: random projection (tests only)

safe.pb + word2id.json: run safe_final/download_model.sh once, or copy them
from s3://<bucket>/assets/ (see RUNBOOK).
"""
import argparse
import os
import socket
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import s3io                                     # noqa: E402
from common.config import REPO, load_config                 # noqa: E402

MAX_INS = 150


class SafeModel:
    def __init__(self, model_path, mock=False):
        self.mock = mock
        if mock:
            rng = np.random.default_rng(0)
            self.W = rng.normal(size=(1 << 16, 100)).astype(np.float32)
            return
        sys.path.insert(0, str(REPO / "safe_final"))
        from neural_network.SAFEEmbedder import SAFEEmbedder
        self.emb = SAFEEmbedder(str(model_path))
        self.emb.loadmodel()
        self.emb.get_tensor()

    def __call__(self, X, L):
        if self.mock:
            m = (np.arange(MAX_INS)[None] < L[:, None]).astype(np.float32)
            v = (self.W[X % len(self.W)] * m[..., None]).sum(1)
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        return self.emb.embedd(X, L)


def embed_table(tbl, model, batch):
    shas = tbl.column("sha256").to_pylist()
    toks = tbl.column("safe_tokens").to_pylist()
    owner, X, L = [], [], []
    for i, funcs in enumerate(toks):
        for f in funcs or []:
            f = f[:MAX_INS]
            if not f:
                continue
            row = np.zeros(MAX_INS, np.int32)
            row[:len(f)] = f
            X.append(row)
            L.append(len(f))
            owner.append(i)
    sums = np.zeros((len(shas), 100), np.float64)
    cnt = np.zeros(len(shas), np.int64)
    if X:
        X, L, owner = np.stack(X), np.asarray(L, np.int32), np.asarray(owner)
        for s in range(0, len(X), batch):
            e = model(X[s:s + batch], L[s:s + batch])
            np.add.at(sums, owner[s:s + batch], e)
            np.add.at(cnt, owner[s:s + batch], 1)
    out = [(sums[i] / cnt[i]).astype(np.float32).tolist() if cnt[i] else None for i in range(len(shas))]
    return pa.table({"sha256": shas, "safe": pa.array(out, pa.list_(pa.float32())),
                     "safe_n_funcs": pa.array(cnt.astype(np.int32))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO / "safe_final" / "data" / "safe.pb"))
    ap.add_argument("--local-in")
    ap.add_argument("--local-out")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    model = SafeModel(args.model, mock=args.mock)
    batch = cfg["embed"]["safe_batch"]
    if args.local_in:
        tbl = pq.read_table(args.local_in, columns=["sha256", "safe_tokens"])
        Path(args.local_out).mkdir(parents=True, exist_ok=True)
        pq.write_table(embed_table(tbl, model, batch), Path(args.local_out) / "safe_local.parquet")
        return
    bucket, region = cfg["s3"]["bucket"], cfg["s3"]["region"]
    wid = f"{socket.gethostname()}-{os.getpid()}"
    for key in sorted(s3io.list_keys(bucket, cfg["s3"]["features_prefix"], region)):
        name = key.rsplit("/", 1)[-1]
        out_key = cfg["s3"]["safe_prefix"] + name
        lease = cfg["s3"]["leases_prefix"] + "B/" + name + ".json"
        if s3io.exists(bucket, out_key, region) or not s3io.acquire_lease(bucket, lease, wid, region=region):
            continue
        try:
            tbl = s3io.read_parquet(bucket, key, region, columns=["sha256", "safe_tokens"])
            s3io.write_parquet(embed_table(tbl, model, batch), bucket, out_key, region)
            print("embedded", name, flush=True)
        finally:
            s3io.release_lease(bucket, lease, region)


if __name__ == "__main__":
    main()
