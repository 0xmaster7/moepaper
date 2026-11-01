"""Merge malware + benign manifests, assign split partitions, write work chunks.

    python data/02_build_manifest.py --malware manifest_malware.parquet

Benign partitions: whole groups (repo / package / DLL name) are assigned to one
partition, so near-identical files never sit in both train and test.
Malware partitions: SOREL time partitions (set by 00_sorel_manifest.py).

Writes to S3:
    v2/manifests/all.parquet
    v2/manifests/chunks/chunk_00000.parquet ...   (shuffled, mixed classes)
"""
import argparse
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import s3io                                     # noqa: E402
from common.config import COLS_TYPE, load_config            # noqa: E402


def group_part(group, fracs, seed):
    h = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:12], 16) / 16 ** 12
    acc = 0.0
    for part, f in fracs.items():
        acc += f
        if h < acc:
            return part
    return list(fracs)[-1]


def load_benign_index(cfg):
    bucket, region = cfg["s3"]["bucket"], cfg["s3"]["region"]
    frames = []
    for key in s3io.list_keys(bucket, "benign/index/", region):
        if not key.endswith(".csv"):
            continue
        src_file = key.rsplit("/", 1)[-1][:-4]
        source = src_file.split("_")[0]                     # windows_win10_0_... -> windows
        df = pd.read_csv(io.BytesIO(s3io.client(region).get_object(Bucket=bucket, Key=key)["Body"].read()))
        df["source"] = source
        df["s3_uri"] = f"s3://{bucket}/{cfg['s3']['benign_raw_prefix']}{source}/" + df.sha256.str.lower()
        frames.append(df)
        print(f"{key}: {len(df):,}")
    b = pd.concat(frames, ignore_index=True)
    b["sha256"] = b.sha256.str.lower()
    return b.drop_duplicates("sha256")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--malware", default="manifest_malware.parquet")
    args = ap.parse_args()
    cfg = load_config()
    sc = cfg["sampling"]

    mal = pd.read_parquet(args.malware)
    ben = load_benign_index(cfg)
    ben = ben[~ben.sha256.isin(set(mal.sha256))]             # paranoia: no hash in both classes

    # cap files per group, then subsample to the target
    cap = cfg["benign_sources"]["max_per_group"]
    ben = ben.sample(frac=1.0, random_state=sc["seed"]).groupby("group", group_keys=False).head(cap)
    if len(ben) > sc["n_benign_target"]:
        # keep the source mix proportional
        total = len(ben)
        ben = pd.concat([d.sample(n=max(1, int(round(sc["n_benign_target"] * len(d) / total))),
                                  random_state=sc["seed"]) for _, d in ben.groupby("source")])
    ben["part"] = [group_part(g, sc["malware_split_fracs"], sc["seed"]) for g in ben.group]
    bdf = pd.DataFrame({
        "sha256": ben.sha256, "label": 0, "source": ben.source, "s3_uri": ben.s3_uri,
        "compression": "none", "first_seen": np.nan, "group": ben.group, "part": ben.part,
    })
    for c in COLS_TYPE:
        bdf[c] = np.float32(0)
    if "detections" in mal.columns:
        bdf["detections"] = 0

    allm = pd.concat([mal, bdf], ignore_index=True)
    allm = allm.sample(frac=1.0, random_state=sc["seed"]).reset_index(drop=True)
    print("\nrows by class/part:\n", allm.groupby(["label", "part"]).size().unstack())
    print("\nbenign by source/part:\n", allm[allm.label == 0].groupby(["source", "part"]).size().unstack())

    bucket, region, pre = cfg["s3"]["bucket"], cfg["s3"]["region"], cfg["s3"]["manifests_prefix"]
    s3io.write_parquet(pa.Table.from_pandas(allm, preserve_index=False), bucket, pre + "all.parquet", region)
    n = sc["chunk_size"]
    for i in range(0, len(allm), n):
        s3io.write_parquet(pa.Table.from_pandas(allm.iloc[i:i + n], preserve_index=False),
                           bucket, f"{pre}chunks/chunk_{i // n:05d}.parquet", region)
    print(f"\nwrote {len(allm):,} rows in {int(np.ceil(len(allm) / n))} chunks to s3://{bucket}/{pre}")


if __name__ == "__main__":
    main()
