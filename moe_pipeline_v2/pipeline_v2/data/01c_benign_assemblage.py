"""Benign Windows PE files from Assemblage (NeurIPS 2024 D&B): ~91k MSVC-built
binaries compiled from public GitHub repositories, hosted on Hugging Face
(changliu8541/Assemblage_PE).

The binaries archive is *streamed* (tar.xz read sequentially over HTTP) and each
PE is uploaded to your bucket; nothing is written to local disk.

Step 1 (see what the metadata and archive actually contain):
    python data/01c_benign_assemblage.py --inspect
Step 2 (adjust --group-col / --path-col / --config-col if the names differ):
    python data/01c_benign_assemblage.py --max-files 25000

Uploads benign/raw/assemblage/<sha256> and benign/index/assemblage.csv.
group = source repository, capped at benign_sources.max_per_group files.
Release builds are preferred (Debug builds are unrealistic: unoptimised code,
debug CRT) unless --allow-debug.
Each binary keeps its source repo's licence: keep files private, publish hashes.
"""
import argparse
import csv
import io
import sys
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd
from huggingface_hub import HfFileSystem, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pe_parity, s3io                          # noqa: E402
from common.config import load_config                       # noqa: E402


def find_file(fs, repo, suffix):
    for p in fs.ls(f"datasets/{repo}", detail=False):
        if p.endswith(suffix):
            return p
    for p in fs.glob(f"datasets/{repo}/**/*{suffix}"):
        return p
    raise FileNotFoundError(suffix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--max-files", type=int, default=25000)
    ap.add_argument("--group-col", default="repo")
    ap.add_argument("--path-col", default="path")
    ap.add_argument("--config-col", default="build_mode")
    ap.add_argument("--allow-debug", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    repo = cfg["benign_sources"]["assemblage_hf_repo"]
    cap = cfg["benign_sources"]["max_per_group"]
    max_bytes = cfg["benign_sources"]["max_file_mb"] * 2 ** 20
    fs = HfFileSystem()

    csv_path = find_file(fs, repo, "binaries.csv")
    local_csv = hf_hub_download(repo, PurePosixPath(csv_path).name, repo_type="dataset",
                                subfolder=str(PurePosixPath(csv_path).parent).split(repo, 1)[-1].strip("/") or None)
    meta = pd.read_csv(local_csv)
    tar_path = find_file(fs, repo, "binaries.tar.xz")

    if args.inspect:
        print("binaries.csv columns:", list(meta.columns))
        print(meta.head(5).to_string())
        with fs.open(tar_path, "rb", block_size=2 ** 22) as fh, tarfile.open(fileobj=fh, mode="r|xz") as tf:
            for i, m in enumerate(tf):
                print("tar member:", m.name, m.size)
                if i >= 15:
                    break
        return

    for col in (args.group_col, args.path_col):
        assert col in meta.columns, f"{col!r} not in binaries.csv; run --inspect"
    if args.config_col in meta.columns and not args.allow_debug:
        before = len(meta)
        meta = meta[~meta[args.config_col].astype(str).str.lower().str.contains("debug")]
        print(f"dropped {before - len(meta):,} debug builds")
    # index metadata by file basename (tar member names are matched on basename)
    meta["_base"] = meta[args.path_col].astype(str).map(lambda s: PurePosixPath(s.replace("\\", "/")).name.lower())
    by_base = {}
    for base, grp in zip(meta["_base"], meta[args.group_col].astype(str)):
        by_base.setdefault(base, grp)

    bucket, region = cfg["s3"]["bucket"], cfg["s3"]["region"]
    per_group, seen, rows = Counter(), set(), []
    with fs.open(tar_path, "rb", block_size=2 ** 22) as fh, tarfile.open(fileobj=fh, mode="r|xz") as tf:
        for m in tf:
            if len(rows) >= args.max_files:
                break
            if not m.isfile() or m.size > max_bytes or m.size < 1024:
                continue
            base = PurePosixPath(m.name).name.lower()
            grp = by_base.get(base)
            if grp is None or per_group[grp] >= cap:
                continue
            data = tf.extractfile(m).read()
            if not pe_parity.is_pe(data):
                continue
            sha = pe_parity.sha256(data)
            if sha in seen:
                continue
            seen.add(sha)
            per_group[grp] += 1
            s3io.put_bytes(bucket, f"{cfg['s3']['benign_raw_prefix']}assemblage/{sha}", data, region)
            rows.append([sha, f"asm:{grp}", m.name.replace(",", "_"), len(data), ""])
            if len(rows) % 1000 == 0:
                print(f"{len(rows):,} uploaded from {len(per_group):,} repos")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["sha256", "group", "origin", "size", "os"])
    w.writerows(rows)
    s3io.put_bytes(bucket, "benign/index/assemblage.csv", buf.getvalue().encode(), region)
    print(f"done: {len(rows):,} files from {len(per_group):,} repositories")


if __name__ == "__main__":
    main()
