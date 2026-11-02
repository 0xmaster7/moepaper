"""Benign Windows PE files from Assemblage (NeurIPS 2024 D&B): ~91k MSVC-built
binaries compiled from public GitHub repositories, hosted on Hugging Face
(changliu8541/Assemblage_PE).

The Hugging Face copy has only binaries.tar.xz, which is actually **zstd**
compressed (magic 28 b5 2f fd), plus a large SQLite archive. There is no CSV,
so the group (source repository) and build type are taken from each file's
path inside the archive. The archive is streamed; nothing is written to disk.

Step 1 - look at the paths inside the archive:
    python data/01c_benign_assemblage.py --inspect
Step 2 - choose how many leading path parts identify a repository, then run:
    python data/01c_benign_assemblage.py --group-depth 1 --max-files 25000

Uploads benign/raw/assemblage/<sha256> and benign/index/assemblage.csv.
group = first --group-depth path components (the source repository), capped at
benign_sources.max_per_group files. Paths containing "debug" are skipped unless
--allow-debug (Debug builds are unrealistic: unoptimised code, debug CRT).
--keep-prob < 1 spreads the sample over the whole archive instead of taking the
first repositories only.
Each binary keeps its source repo's licence: keep files private, publish hashes.
"""
import argparse
import csv
import io
import random
import sys
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

import zstandard
from huggingface_hub import HfFileSystem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pe_parity, s3io                          # noqa: E402
from common.config import load_config                       # noqa: E402

ARCHIVE = "binaries.tar.xz"


def open_members(repo):
    """Yield (TarInfo, tarfile) for every member, streaming zstd over HTTP."""
    fs = HfFileSystem()
    with fs.open(f"datasets/{repo}/{ARCHIVE}", "rb", block_size=2 ** 23) as fh:
        dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
        with dctx.stream_reader(fh, read_size=2 ** 22) as zr, tarfile.open(fileobj=zr, mode="r|") as tf:
            for m in tf:
                yield m, tf


def group_of(name, depth):
    parts = [p for p in PurePosixPath(name).parts if p not in (".", "")]
    return "/".join(parts[:depth]) if len(parts) > depth else parts[0] if parts else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--max-files", type=int, default=25000)
    ap.add_argument("--group-depth", type=int, default=1)
    ap.add_argument("--keep-prob", type=float, default=0.4)
    ap.add_argument("--allow-debug", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    repo = cfg["benign_sources"]["assemblage_hf_repo"]
    cap = cfg["benign_sources"]["max_per_group"]
    max_bytes = cfg["benign_sources"]["max_file_mb"] * 2 ** 20

    if args.inspect:
        for i, (m, _) in enumerate(open_members(repo)):
            print(("DIR " if m.isdir() else "    ") + m.name, m.size)
            if i >= 60:
                break
        return

    rng = random.Random(cfg["sampling"]["seed"])
    bucket, region = cfg["s3"]["bucket"], cfg["s3"]["region"]
    per_group, seen, rows = Counter(), set(), []
    n_seen = n_debug = 0
    for m, tf in open_members(repo):
        if len(rows) >= args.max_files:
            break
        if not m.isfile() or m.size > max_bytes or m.size < 1024:
            continue
        n_seen += 1
        if not args.allow_debug and "debug" in m.name.lower():
            n_debug += 1
            continue
        grp = group_of(m.name, args.group_depth)
        if per_group[grp] >= cap or rng.random() > args.keep_prob:
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
            print(f"{len(rows):,} uploaded from {len(per_group):,} groups "
                  f"({n_seen:,} files scanned, {n_debug:,} debug skipped)", flush=True)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["sha256", "group", "origin", "size", "os"])
    w.writerows(rows)
    s3io.put_bytes(bucket, "benign/index/assemblage.csv", buf.getvalue().encode(), region)
    print(f"done: {len(rows):,} files from {len(per_group):,} groups")


if __name__ == "__main__":
    main()
