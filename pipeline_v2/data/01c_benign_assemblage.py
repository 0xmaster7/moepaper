"""Benign Windows PE files from Assemblage (NeurIPS 2024 D&B): ~91k MSVC-built
binaries compiled from public GitHub repositories, hosted on Hugging Face
(changliu8541/Assemblage_PE).

The Hugging Face copy has only binaries.tar.xz, which is actually **zstd**
compressed (magic 28 b5 2f fd), plus a large SQLite archive. There is no CSV.
Inside the archive, files sit in id-bucket folders (binaries/83/83/47/...):
.pdb files carry the build config in their names, but the .exe/.dll files only
carry their output name (MyBot.exe). So:

  group  = lower-case file name. Builds of the same project in different
           configurations (x86/x64, toolsets, opt levels) share the output name
           and therefore land in one split; generic names (Project1.exe) are
           merged, which is conservative. Same rule as the Windows harvest.
  debug  = detected from imports of the debug CRT (ucrtbased.dll,
           vcruntime140d.dll, msvcp140d.dll, msvcr120d.dll, ...). Debug builds are
           skipped (unoptimised code, debug CRT: unlike real software) unless
           --allow-debug.

The archive is streamed over plain HTTP with automatic reconnect: if the
connection drops (HF returns 499/5xx on long transfers), it resumes from the
same byte offset with a Range request. Nothing is written to disk.
The index is rewritten to S3 every 500 uploads, so a crash loses nothing;
--resume reloads it and skips files already uploaded.
    python data/01c_benign_assemblage.py --inspect          # first 60 entries
    python data/01c_benign_assemblage.py --max-files 25000
    python data/01c_benign_assemblage.py --max-files 25000 --resume   # after a crash

Uploads benign/raw/assemblage/<sha256> and benign/index/assemblage.csv.
--keep-prob < 1 spreads the sample over the whole archive instead of taking
only the first folders.
Each binary keeps its source repo's licence: keep files private, publish hashes.
"""
import argparse
import csv
import io
import os
import random
import re
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd
import pefile
import requests
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pe_parity, s3io                          # noqa: E402
from common.config import load_config                       # noqa: E402

ARCHIVE = "binaries.tar.xz"
INDEX_KEY = "benign/index/assemblage.csv"


class ResumableHTTP(io.RawIOBase):
    """Sequential reader over one large HTTP file that reconnects with a Range
    request from the current byte offset whenever the connection drops."""

    def __init__(self, url, retries=30, headers=None):
        self.url, self.retries, self.pos = url, retries, 0
        self.headers = dict(headers or {})
        self.total, self.resp = None, None
        self.reconnects = 0

    def readable(self):
        return True

    def _open(self):
        h = dict(self.headers, **({"Range": f"bytes={self.pos}-"} if self.pos else {}))
        r = requests.get(self.url, headers=h, stream=True, timeout=(30, 300), allow_redirects=True)
        r.raise_for_status()
        if self.pos and r.status_code != 206:
            r.close()
            raise IOError("server ignored the Range header")
        if self.total is None:
            cr = r.headers.get("Content-Range")
            self.total = int(cr.rsplit("/", 1)[1]) if cr and "/" in cr else int(r.headers.get("Content-Length", 0)) or None
        self.resp = r

    def _drop(self):
        try:
            if self.resp is not None:
                self.resp.close()
        except Exception:
            pass
        self.resp = None

    def readinto(self, b):
        for attempt in range(self.retries):
            try:
                if self.resp is None:
                    self._open()
                n = self.resp.raw.readinto(b)
                if n == 0 and self.total is not None and self.pos < self.total:
                    raise IOError(f"stream ended early at {self.pos:,} of {self.total:,} bytes")
                self.pos += n
                return n
            except Exception as e:  # noqa: BLE001
                self._drop()
                self.reconnects += 1
                wait = min(60, 2 ** attempt)
                print(f"  [reconnect {self.reconnects}] at byte {self.pos:,}: {type(e).__name__}: "
                      f"{str(e)[:120]} - retrying in {wait}s", flush=True)
                time.sleep(wait)
        raise IOError(f"gave up after {self.retries} retries at byte {self.pos:,}")

    def close(self):
        self._drop()
        super().close()


def open_members(repo):
    """Yield (TarInfo, tarfile) for every member, streaming zstd over HTTP."""
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{ARCHIVE}"
    tok = os.environ.get("HF_TOKEN")
    raw = ResumableHTTP(url, headers={"Authorization": f"Bearer {tok}"} if tok else None)
    fh = io.BufferedReader(raw, buffer_size=2 ** 23)
    dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
    try:
        with dctx.stream_reader(fh, read_size=2 ** 22) as zr, tarfile.open(fileobj=zr, mode="r|") as tf:
            for m in tf:
                yield m, tf
    finally:
        raw.close()


def write_index(rows, bucket, region):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["sha256", "group", "origin", "size", "os"])
    w.writerows(rows)
    s3io.put_bytes(bucket, INDEX_KEY, buf.getvalue().encode(), region)


DEBUG_CRT = re.compile(r"^(ucrtbased|vcruntime\d+d|msvcp\d+d|msvcr\d+d|concrt\d+d|vcomp\d+d|mfc\d+u?d)\.dll$", re.I)


def is_debug_build(data):
    """True if the PE imports a debug C/C++ runtime (MSVC Debug configuration)."""
    try:
        pe = pefile.PE(data=data, fast_load=True)
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        for imp in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
            if DEBUG_CRT.match(imp.dll.decode(errors="ignore")):
                return True
    except Exception:
        pass
    return False


def group_of(name):
    return PurePosixPath(name).name.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--max-files", type=int, default=25000)
    ap.add_argument("--keep-prob", type=float, default=0.4)
    ap.add_argument("--allow-debug", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reload the S3 index and skip files already uploaded")
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
    per_group, seen, rows, done_names = Counter(), set(), [], set()
    if args.resume and s3io.exists(bucket, INDEX_KEY, region):
        body = s3io.client(region).get_object(Bucket=bucket, Key=INDEX_KEY)["Body"].read()
        old = pd.read_csv(io.BytesIO(body))
        rows = old.astype({"size": int}).fillna("").values.tolist()
        seen = set(old.sha256)
        done_names = set(old.origin)
        per_group = Counter(g.split(":", 1)[1] for g in old.group)
        print(f"resume: {len(rows):,} files already uploaded; skipping them", flush=True)
    n_seen = n_debug = 0
    last_flush = len(rows)
    for m, tf in open_members(repo):
        if len(rows) >= args.max_files:
            break
        if not m.isfile() or m.size > max_bytes or m.size < 1024:
            continue
        if not m.name.lower().endswith((".exe", ".dll")):
            continue                                    # skip .pdb and other files
        n_seen += 1
        if m.name.replace(",", "_") in done_names:
            continue
        grp = group_of(m.name)
        if per_group[grp] >= cap or rng.random() > args.keep_prob:
            continue
        data = tf.extractfile(m).read()
        if not pe_parity.is_pe(data):
            continue
        if not args.allow_debug and is_debug_build(data):
            n_debug += 1
            continue
        sha = pe_parity.sha256(data)
        if sha in seen:
            continue
        seen.add(sha)
        per_group[grp] += 1
        s3io.put_bytes(bucket, f"{cfg['s3']['benign_raw_prefix']}assemblage/{sha}", data, region)
        rows.append([sha, f"asm:{grp}", m.name.replace(",", "_"), len(data), ""])
        if len(rows) - last_flush >= 500:
            write_index(rows, bucket, region)
            last_flush = len(rows)
            print(f"{len(rows):,} uploaded from {len(per_group):,} groups "
                  f"({n_seen:,} exe/dll scanned, {n_debug:,} debug builds skipped) - index saved", flush=True)
    write_index(rows, bucket, region)
    print(f"done: {len(rows):,} files from {len(per_group):,} groups "
          f"({n_seen:,} exe/dll scanned, {n_debug:,} debug builds skipped)")


if __name__ == "__main__":
    main()
