"""Upload benign files you already have locally (e.g. your original ~6.4k Windows
system files) to S3 and index them. Benign only - never point this at malware.

    python data/01d_upload_local_benign.py --dir /path/to/benign_binaries --source legacy

group = lower-case file name (same rule as the Windows harvest).
"""
import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pe_parity, s3io                          # noqa: E402
from common.config import load_config                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--source", default="legacy")
    args = ap.parse_args()
    cfg = load_config()
    bucket, region = cfg["s3"]["bucket"], cfg["s3"]["region"]
    rows, seen = [], set()
    for p in Path(args.dir).rglob("*"):
        if not p.is_file():
            continue
        b = p.read_bytes()
        if not pe_parity.is_pe(b):
            continue
        sha = pe_parity.sha256(b)
        if sha in seen:
            continue
        seen.add(sha)
        s3io.put_bytes(bucket, f"{cfg['s3']['benign_raw_prefix']}{args.source}/{sha}", b, region)
        name = p.name.lower()
        # strip the "<uuid>_" prefix some of the original files carry
        if len(name) > 37 and name[36] == "_" and name.count("-") >= 4:
            name = name[37:]
        rows.append([sha, f"win:{name}", str(p).replace(",", "_"), len(b), "legacy"])
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["sha256", "group", "origin", "size", "os"])
    w.writerows(rows)
    s3io.put_bytes(bucket, f"benign/index/{args.source}.csv", buf.getvalue().encode(), region)
    print(f"uploaded {len(rows)} files")


if __name__ == "__main__":
    main()
