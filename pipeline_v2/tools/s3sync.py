"""Mirror a local folder to/from S3 with boto3 (the cloud boxes have no AWS CLI).

    python tools/s3sync.py up   results v2/results/     # upload new or changed files
    python tools/s3sync.py down results v2/results/     # download files missing locally
A file counts as changed when its size differs from the S3 copy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import s3io                                     # noqa: E402
from common.config import load_config                       # noqa: E402


def main():
    direction, local, prefix = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    cfg = load_config()
    bucket, region = cfg["s3"]["bucket"], cfg["s3"]["region"]
    s3 = s3io.client(region)
    remote = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            remote[o["Key"][len(prefix):]] = o["Size"]
    n = 0
    if direction == "up":
        for p in sorted(local.rglob("*")):
            rel = p.relative_to(local).as_posix()
            if p.is_file() and remote.get(rel) != p.stat().st_size:
                s3.upload_file(str(p), bucket, prefix + rel)
                n += 1
    else:
        for rel, size in sorted(remote.items()):
            p = local / rel
            if not p.exists() or p.stat().st_size != size:
                p.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, prefix + rel, str(p))
                n += 1
    print(f"s3sync {direction}: {n} files", flush=True)


if __name__ == "__main__":
    main()
