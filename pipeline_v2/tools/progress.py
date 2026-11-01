"""How far is the fleet? Counts chunks per stage and summarises finished ones.

    python tools/progress.py            # quick counts
    python tools/progress.py --detail   # also reads finished feature chunks (coverage, sec/sample)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import s3io                                     # noqa: E402
from common.config import load_config                       # noqa: E402


def names(cfg, prefix):
    return {k.rsplit("/", 1)[-1] for k in s3io.list_keys(cfg["s3"]["bucket"], prefix, cfg["s3"]["region"])
            if k.endswith(".parquet")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    s = cfg["s3"]
    chunks = names(cfg, s["manifests_prefix"] + "chunks/")
    feats, safe = names(cfg, s["features_prefix"]), names(cfg, s["safe_prefix"])
    leases = list(s3io.list_keys(s["bucket"], s["leases_prefix"], s["region"]))
    print(f"manifest chunks {len(chunks)} | Stage A done {len(feats & chunks)} | SAFE done {len(safe)} | "
          f"active leases {len(leases)}")
    if args.detail and feats:
        cols = ["label", "ok_pe", "ok_ember", "ok_r2", "unk_rate", "secs", "r2_level"]
        df = pd.concat([s3io.read_parquet(s["bucket"], s["features_prefix"] + n, s["region"], cols).to_pandas()
                        for n in sorted(feats)])
        print(df.groupby("label")[["ok_pe", "ok_ember", "ok_r2", "unk_rate", "secs"]].mean().round(3))
        print("r2 level used:", df.r2_level.value_counts().to_dict())
        left = (len(chunks) - len(feats & chunks)) * cfg["sampling"]["chunk_size"]
        print(f"~{left * df.secs.mean() / 3600:.1f} CPU-hours of Stage A left "
              f"(divide by total vCPUs in the fleet)")


if __name__ == "__main__":
    main()
