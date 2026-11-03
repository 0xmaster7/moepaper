"""Upload benign files you already have locally (e.g. your original ~6.4k Windows
system files) to S3 and index them. Benign only - never point this at malware.

    python data/01d_upload_local_benign.py --dir /path/to/benign_binaries --source legacy

group = lower-case file name (same rule as the Windows harvest).
Files compressed with zlib or gzip (e.g. the *.z files your old SAFE batch
script used) are decompressed first; a trailing .z/.gz is dropped from the name.
Safe to re-run after a network drop: files already in S3 are not uploaded again.
"""

import argparse
import csv
import gzip
import io
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pe_parity, s3io                          # noqa: E402
from common.config import load_config                       # noqa: E402


def load_pe(p: Path):
    """Raw bytes of a PE file, decompressing zlib/gzip wrappers if needed."""
    b = p.read_bytes()
    if pe_parity.is_pe(b):
        return b
    for dec in (zlib.decompress, gzip.decompress):
        try:
            d = dec(b)
            if pe_parity.is_pe(d):
                return d
        except Exception:  # noqa: BLE001
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--source", default="legacy")
    args = ap.parse_args()
    cfg = load_config()
    bucket, region = cfg["s3"]["bucket"], cfg["s3"]["region"]
    rows, seen, stats = [], set(), Counter()
    prefix = f"{cfg['s3']['benign_raw_prefix']}{args.source}/"
    existing = {k.rsplit("/", 1)[-1] for k in s3io.list_keys(bucket, prefix, region)}
    if existing:
        print(f"{len(existing):,} files already in s3://{bucket}/{prefix} - those are skipped", flush=True)
    for p in Path(args.dir).rglob("*"):
        if not p.is_file():
            continue
        stats["files"] += 1
        b = load_pe(p)
        if b is None:
            stats["not_pe"] += 1
            continue
        sha = pe_parity.sha256(b)
        if sha in seen:
            continue
        seen.add(sha)
        if sha not in existing:
            s3io.put_bytes(bucket, prefix + sha, b, region)
            stats["uploaded"] += 1
            if stats["uploaded"] % 500 == 0:
                print(f"{stats['uploaded']:,} uploaded this run ({stats['files']:,} files scanned)", flush=True)
        name = p.name.lower()
        for suf in (".z", ".gz"):
            if name.endswith(suf):
                name = name[: -len(suf)]
        # strip the "<uuid>_" prefix some of the original files carry
        if len(name) > 37 and name[36] == "_" and name.count("-") >= 4:
            name = name[37:]
        rows.append([sha, f"win:{name}", str(p).replace(",", "_"), len(b), "legacy"])
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["sha256", "group", "origin", "size", "os"])
    w.writerows(rows)
    s3io.put_bytes(bucket, f"benign/index/{args.source}.csv", buf.getvalue().encode(), region)
    print(f"indexed {len(rows)} unique PE files ({stats['uploaded']} uploaded this run, "
          f"{stats['files']} files scanned, {stats['not_pe']} not PE)")


if __name__ == "__main__":
    main()
