"""Join manifest + Stage A features + Stage B SAFE embeddings into one local dataset.

    python build/build_dataset.py                    # from S3
    python build/build_dataset.py --graphs           # also pack call graphs for the GNN baseline
    python build/build_dataset.py --local-manifest m.parquet --local-features out/ --local-safe out/

Writes to config.dataset.local_dir:
    dataset.npz          arrays (EMBER, expert embeddings, masks, labels, tags)
    meta.parquet         one row per sample (sha256, source, group, part, diagnostics)
    splits_temporal.npz  malware by SOREL time partition, benign by group
    splits_random.npz    class-stratified random 70/15/15 (comparable to the old paper)
    graphs.npz           (with --graphs) CSR-packed attributed call graphs
    coverage.md          expert coverage by class and split - put this table in the paper
"""
import argparse
import glob
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import s3io                                     # noqa: E402
from common.config import COLS_TYPE, load_config            # noqa: E402

FEAT_COLS = ["sha256", "label", "ok_pe", "ok_ember", "ok_r2", "r2_level", "arch", "bits", "n_funcs",
             "n_nodes", "n_edges", "unk_rate", "err", "secs", "ember", "ldp", "n2v", "malconv", "malconv_logit"]


def read_many(paths_or_keys, cfg, columns=None, local=False):
    tbls = []
    for p in paths_or_keys:
        tbls.append(pq.read_table(p, columns=columns) if local else
                    s3io.read_parquet(cfg["s3"]["bucket"], p, cfg["s3"]["region"], columns=columns))
    return pa.concat_tables(tbls) if tbls else None


def stack(col, n_rows, dim=None):
    """list<float> column -> (N, d) matrix + availability mask."""
    vals = col.to_pylist() if hasattr(col, "to_pylist") else list(col)
    if dim is None:
        dim = next((len(v) for v in vals if v is not None), 0)
    M = np.zeros((n_rows, dim), np.float32)
    m = np.zeros(n_rows, np.float32)
    for i, v in enumerate(vals):
        if v is not None and len(v) == dim:
            M[i] = v
            m[i] = 1.0
    return M, m


def random_split(y, seed, fracs=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_te, n_va = int(len(idx) * fracs[2]), int(len(idx) * fracs[1])
        te.append(idx[:n_te]); va.append(idx[n_te:n_te + n_va]); tr.append(idx[n_te + n_va:])
    return [np.sort(np.concatenate(x)) for x in (tr, va, te)]


def fingerprint(shas):
    return hashlib.sha256("\n".join(shas).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", action="store_true")
    ap.add_argument("--local-manifest")
    ap.add_argument("--local-features")
    ap.add_argument("--local-safe")
    args = ap.parse_args()
    cfg = load_config()
    out = Path(cfg["dataset"]["local_dir"])
    out.mkdir(parents=True, exist_ok=True)
    local = bool(args.local_manifest)

    if local:
        man = pd.read_parquet(args.local_manifest)
        feat = read_many(sorted(glob.glob(f"{args.local_features}/features*.parquet")), cfg, FEAT_COLS, True)
        safe = read_many(sorted(glob.glob(f"{args.local_safe}/safe*.parquet")), cfg, None, True)
        gtab = read_many(sorted(glob.glob(f"{args.local_features}/features*.parquet")), cfg,
                         ["sha256", "node_x", "edge_src", "edge_dst"], True) if args.graphs else None
    else:
        b, r = cfg["s3"]["bucket"], cfg["s3"]["region"]
        man = s3io.read_parquet(b, cfg["s3"]["manifests_prefix"] + "all.parquet", r).to_pandas()
        fkeys = sorted(k for k in s3io.list_keys(b, cfg["s3"]["features_prefix"], r) if k.endswith(".parquet"))
        skeys = sorted(k for k in s3io.list_keys(b, cfg["s3"]["safe_prefix"], r) if k.endswith(".parquet"))
        feat = read_many(fkeys, cfg, FEAT_COLS)
        safe = read_many(skeys, cfg)
        gtab = read_many(fkeys, cfg, ["sha256", "node_x", "edge_src", "edge_dst"]) if args.graphs else None

    f = feat.to_pandas()
    f = f.drop_duplicates("sha256")
    f = f.merge(man.drop(columns=["label"]), on="sha256", how="inner")
    if safe is not None:
        f = f.merge(safe.to_pandas().drop_duplicates("sha256"), on="sha256", how="left")
    else:
        f["safe"], f["safe_n_funcs"] = None, 0
    n0 = len(f)
    f = f[f.ok_pe & f.ok_ember].reset_index(drop=True)          # EMBER is required (gate + baselines)
    print(f"samples with features: {n0:,} -> usable (PE + EMBER ok): {len(f):,}")
    print(f.groupby("label").size().rename("count").to_frame().T)

    N = len(f)
    y = f.label.to_numpy().astype(np.int64)
    Y_counts = f[COLS_TYPE].fillna(0).to_numpy(np.float32)
    Y_type = (Y_counts > 0).astype(np.float32)
    X_ember, _ = stack(f.ember, N)
    E_mal, m_mal = stack(f.malconv, N)
    E_safe, m_safe = stack(f.safe, N, 100)
    E_ldp, m_ldp = stack(f.ldp, N)
    E_n2v, m_n2v = stack(f.n2v, N, cfg["extract"]["node2vec"]["dimensions"])
    m_ldp = m_ldp * f.ok_r2.to_numpy(np.float32)
    shas = f.sha256.tolist()

    np.savez_compressed(out / "dataset.npz", sha256=np.array(shas), y=y, Y_type=Y_type, Y_counts=Y_counts,
                        X_ember=X_ember, E_malconv=E_mal, m_malconv=m_mal, E_safe=E_safe, m_safe=m_safe,
                        E_ldp=E_ldp, m_ldp=m_ldp, E_n2v=E_n2v, m_n2v=m_n2v,
                        malconv_logit=f.malconv_logit.fillna(0).to_numpy(np.float32),
                        detections=(f["detections"].fillna(0).to_numpy(np.float32) if "detections" in f
                                    else np.full(N, -1, np.float32)),
                        cols_type=np.array(COLS_TYPE), fingerprint=fingerprint(shas))
    meta_cols = ["sha256", "label", "source", "group", "part", "first_seen", "ok_r2", "r2_level", "arch",
                 "bits", "n_funcs", "n_nodes", "n_edges", "unk_rate", "secs", "safe_n_funcs", "err"]
    meta = f[[c for c in meta_cols if c in f.columns]]
    meta.to_parquet(out / "meta.parquet", index=False)

    # ---- splits (indices into dataset.npz rows)
    part = f.part.to_numpy()
    t = [np.flatnonzero(part == p) for p in ("train", "val", "test")]
    fp = fingerprint(shas)
    np.savez(out / "splits_temporal.npz", train_idx=t[0], val_idx=t[1], test_idx=t[2], fingerprint=fp)
    rnd = random_split(y, cfg["sampling"]["seed"])
    np.savez(out / "splits_random.npz", train_idx=rnd[0], val_idx=rnd[1], test_idx=rnd[2], fingerprint=fp)
    for name, (tr, va, te) in (("temporal", t), ("random", rnd)):
        print(f"{name:9s} train {len(tr):6,} ({y[tr].mean():.2f} mal) | val {len(va):6,} | test {len(te):6,} "
              f"({int(y[te].sum()):,} mal / {int((y[te] == 0).sum()):,} ben)")
    gtr = set(f.group[t[0]][y[t[0]] == 0]); gte = set(f.group[t[2]][y[t[2]] == 0])
    assert not (gtr & gte), "benign group leaked across train/test"

    # ---- coverage table (report in the paper; class-dependent coverage is itself a signal)
    rows = []
    for nm, m in (("MalConv", m_mal), ("SAFE", m_safe), ("Graph (LDP)", m_ldp), ("node2vec", m_n2v)):
        rows.append([nm] + [f"{m[(y == c)].mean():.3f}" for c in (1, 0)] +
                    [f"{m[idx].mean():.3f}" for idx in t])
    cov = pd.DataFrame(rows, columns=["expert", "malware", "benign", "train", "val", "test"])
    (out / "coverage.md").write_text(cov.to_markdown(index=False) if hasattr(cov, "to_markdown") else cov.to_string())
    print(cov.to_string(index=False))
    print(f"mean SAFE UNK rate: {np.nanmean(f.unk_rate):.3f}  (legacy pipeline: 1.000)")

    if args.graphs and gtab is not None:
        g = gtab.to_pandas().drop_duplicates("sha256").set_index("sha256")
        xs, es, nptr, eptr = [], [], [0], [0]
        for s in shas:
            row = g.loc[s] if s in g.index else None
            nx = np.asarray(row.node_x.tolist(), np.float16) if row is not None and row.node_x is not None else np.zeros((0, 14), np.float16)
            if row is not None and row.edge_src is not None:
                e = np.stack([np.asarray(row.edge_src, np.int32), np.asarray(row.edge_dst, np.int32)])
            else:
                e = np.zeros((2, 0), np.int32)
            xs.append(nx.reshape(-1, 14)); es.append(e)
            nptr.append(nptr[-1] + len(xs[-1])); eptr.append(eptr[-1] + e.shape[1])
        np.savez(out / "graphs.npz", node_x=np.concatenate(xs), edges=np.concatenate(es, axis=1),
                 node_ptr=np.asarray(nptr, np.int64), edge_ptr=np.asarray(eptr, np.int64), fingerprint=fp)
        print(f"graphs.npz: {nptr[-1]:,} nodes, {eptr[-1]:,} edges")
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
