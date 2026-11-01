"""S3 helpers: stream objects into memory, write parquet shards, chunk leases.

Nothing here writes a binary to persistent local disk. Bytes live in RAM and,
only when radare2 needs a path, briefly in /dev/shm (RAM-backed tmpfs).
"""
import io
import json
import time
import zlib

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

_clients = {}


def client(region="us-east-1", unsigned=False):
    key = (region, unsigned)
    if key not in _clients:
        cfg = Config(region_name=region, retries={"max_attempts": 10, "mode": "adaptive"},
                     max_pool_connections=64)
        if unsigned:
            cfg = cfg.merge(Config(signature_version=UNSIGNED))
        _clients[key] = boto3.client("s3", config=cfg)
    return _clients[key]


def split_uri(uri):
    assert uri.startswith("s3://"), uri
    b, _, k = uri[5:].partition("/")
    return b, k


def get_bytes(uri, region="us-east-1", unsigned=False):
    b, k = split_uri(uri)
    return client(region, unsigned).get_object(Bucket=b, Key=k)["Body"].read()


def get_sample_bytes(row, region="us-east-1"):
    """Original (un-modified) bytes for one manifest row (s3:// or file:// for tests)."""
    uri = row["s3_uri"]
    if uri.startswith("file://"):
        with open(uri[7:], "rb") as fh:
            raw = fh.read()
    else:
        raw = get_bytes(uri, region, unsigned=uri.startswith("s3://sorel-20m/"))
    if row.get("compression") == "zlib":
        raw = zlib.decompress(raw)
    return raw


def exists(bucket, key, region="us-east-1"):
    try:
        client(region).head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def list_keys(bucket, prefix, region="us-east-1", unsigned=False):
    pag = client(region, unsigned).get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            yield o["Key"]


def put_bytes(bucket, key, data, region="us-east-1"):
    client(region).put_object(Bucket=bucket, Key=key, Body=data)


def write_parquet(table: pa.Table, bucket, key, region="us-east-1"):
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    put_bytes(bucket, key, buf.getvalue(), region)


def read_parquet(bucket, key, region="us-east-1", columns=None):
    data = client(region).get_object(Bucket=bucket, Key=key)["Body"].read()
    return pq.read_table(io.BytesIO(data), columns=columns)


# ---------------------------------------------------------------- leases
# A lease is a tiny object leases/<stage>/<chunk>.json written with
# If-None-Match so only one worker wins it. Leases older than `ttl_s`
# are treated as dead (spot instance reclaimed) and can be stolen.
def acquire_lease(bucket, key, worker_id, ttl_s=4 * 3600, region="us-east-1"):
    body = json.dumps({"worker": worker_id, "t": time.time()}).encode()
    s3 = client(region)
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch="*")
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code not in ("PreconditionFailed", "ConditionalRequestConflict"):
            raise
    # someone holds it: steal only if stale
    try:
        cur = s3.get_object(Bucket=bucket, Key=key)
        info = json.loads(cur["Body"].read())
        if time.time() - info.get("t", 0) < ttl_s:
            return False
        s3.put_object(Bucket=bucket, Key=key, Body=body, IfMatch=cur["ETag"])
        return True
    except ClientError:
        return False


def release_lease(bucket, key, region="us-east-1"):
    try:
        client(region).delete_object(Bucket=bucket, Key=key)
    except ClientError:
        pass
