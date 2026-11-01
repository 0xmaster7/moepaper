"""Sample N SOREL malware samples from meta.db -> manifest_malware.parquet.

Run on the cloud box (meta.db is 3.5 GB of metadata, no binaries):
    python data/00_sorel_manifest.py --inspect          # print schema, row counts
    python data/00_sorel_manifest.py --include-legacy ../MoE_project_data/sha_list.txt

Sampling
  * Draws from SOREL's own time partitions (train < 2018-11-30 < val <
    2019-01-12 < test, the boundaries in SOREL's config.py), in the
    proportions set in config.sampling.malware_split_fracs.
  * Inside each partition, rare behaviour tags are filled first up to
    min_per_tag (scaled by the partition fraction), then the rest at random.
  * --include-legacy forces your original 3,000 hashes in, each placed in
    the partition its own timestamp says.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import s3io                                     # noqa: E402
from common.config import COLS_TYPE, load_config            # noqa: E402

REQUIRED = ["sha256", "is_malware", "rl_fs_t"] + COLS_TYPE


def fetch_meta_db(cfg, dest):
    dest = Path(dest)
    if not dest.exists():
        print(f"downloading meta.db -> {dest} (3.5 GB, metadata only)")
        s3io.client(cfg["s3"]["region"], unsigned=True).download_file(
            cfg["s3"]["sorel_bucket"], cfg["s3"]["sorel_meta_db_key"], str(dest))
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-db", default="/mnt/work/meta.db")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--include-legacy", default=None, help="sha_list.txt of the original malware")
    ap.add_argument("--out", default="manifest_malware.parquet")
    args = ap.parse_args()
    cfg = load_config()
    sc = cfg["sampling"]
    Path(args.meta_db).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(fetch_meta_db(cfg, args.meta_db))

    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    table = "meta" if "meta" in tables else tables[0]
    cols = [r[1] for r in con.execute(f"pragma table_info({table})")]
    if args.inspect:
        print("tables:", tables, "| using:", table)
        print("columns:", cols)
        print(con.execute(f"select is_malware, count(*) from {table} group by is_malware").fetchall())
        return
    missing = [c for c in REQUIRED if c not in cols]
    assert not missing, f"meta.db lacks {missing}; run --inspect and adjust REQUIRED"
    extra = [c for c in ("rl_ls_const_positives",) if c in cols]

    q = f"select {', '.join(REQUIRED + extra)} from {table} where is_malware = 1"
    df = pd.read_sql_query(q, con)
    print(f"SOREL malware rows: {len(df):,}")
    t1, t2 = sc["sorel_train_val_ts"], sc["sorel_val_test_ts"]
    df["part"] = np.where(df.rl_fs_t < t1, "train", np.where(df.rl_fs_t < t2, "val", "test"))
    print(df.part.value_counts().to_dict())

    rng = np.random.default_rng(sc["seed"])
    forced = set()
    if args.include_legacy:
        forced = {s.strip().lower() for s in Path(args.include_legacy).read_text().split() if s.strip()}
        print(f"legacy hashes requested: {len(forced)} | present in meta.db: {df.sha256.isin(forced).sum()}")

    picked = []
    for part, frac in sc["malware_split_fracs"].items():
        pool = df[df.part == part].reset_index(drop=True)
        budget = int(round(sc["n_malware"] * frac))
        sel_mask = pool.sha256.isin(forced).to_numpy().copy()
        tag_bin = pool[COLS_TYPE].to_numpy() > 0
        quota = int(round(sc["min_per_tag"] * frac))
        for j in np.argsort(tag_bin.sum(0)):                          # rarest tag first
            need = quota - int((tag_bin[:, j] & sel_mask).sum())
            cand = np.flatnonzero(tag_bin[:, j] & ~sel_mask)
            if need > 0 and len(cand):
                sel_mask[rng.choice(cand, size=min(need, len(cand)), replace=False)] = True
        rest = np.flatnonzero(~sel_mask)
        fill = max(0, budget - int(sel_mask.sum()))
        if fill and len(rest):
            sel_mask[rng.choice(rest, size=min(fill, len(rest)), replace=False)] = True
        sel = pool[sel_mask]
        print(f"{part}: picked {len(sel):,} (budget {budget:,})")
        picked.append(sel)

    m = pd.concat(picked, ignore_index=True)
    out = pd.DataFrame({
        "sha256": m.sha256.str.lower(),
        "label": 1,
        "source": "sorel",
        "s3_uri": "s3://" + cfg["s3"]["sorel_bucket"] + "/" + cfg["s3"]["sorel_binaries_prefix"] + m.sha256,
        "compression": "zlib",
        "first_seen": m.rl_fs_t.astype(float),
        "group": m.sha256.str.lower(),
        "part": m.part,
    })
    for c in COLS_TYPE:
        out[c] = m[c].astype(np.float32).to_numpy()
    if extra:
        out["detections"] = m[extra[0]].to_numpy()
    out.to_parquet(args.out, index=False)
    print(f"wrote {args.out}: {len(out):,} rows")
    print("tag prevalence in sample:", (out[COLS_TYPE] > 0).sum().to_dict())


if __name__ == "__main__":
    main()
