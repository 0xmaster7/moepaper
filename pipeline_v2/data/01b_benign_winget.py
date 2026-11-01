"""Benign third-party software from the official winget package manifests.

Every winget installer manifest lists the vendor download URL *and* its
SHA-256, so each file is verified against an independent hash before upload.
Run on the cloud box (not your laptop):

    git clone --depth 1 https://github.com/microsoft/winget-pkgs /mnt/work/winget-pkgs
    python data/01b_benign_winget.py --repo /mnt/work/winget-pkgs --max-packages 8000

Uploads benign/raw/winget/<sha256> and benign/index/winget.csv
(sha256,group,origin,size,os) with group = PackageIdentifier.
Zip "portable" packages are opened in memory and each PE inside is uploaded.
Files are never executed.
"""
import argparse
import csv
import hashlib
import io
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pe_parity, s3io                          # noqa: E402
from common.config import load_config                       # noqa: E402

KEEP_TYPES = {"portable", "exe", "nullsoft", "inno", "burn", "zip"}


def ver_key(v):
    return [int(x) if x.isdigit() else x for x in re.split(r"[.\-+]", str(v))]


def latest_installer_manifests(repo: Path):
    """PackageIdentifier -> path of the newest *.installer.yaml."""
    best = {}
    for p in (repo / "manifests").rglob("*.installer.yaml"):
        ver = p.parent.name
        pid = ".".join(p.relative_to(repo / "manifests").parts[1:-2])  # e.g. Mozilla.Firefox
        cur = best.get(pid)
        try:
            if cur is None or ver_key(ver) > ver_key(cur[0]):
                best[pid] = (ver, p)
        except TypeError:
            best.setdefault(pid, (ver, p))
    return {k: v[1] for k, v in best.items()}


def installers(path: Path):
    y = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
    top_type = y.get("InstallerType")
    top_nested = y.get("NestedInstallerType")
    for ins in y.get("Installers", []) or []:
        t = (ins.get("InstallerType") or top_type or "").lower()
        nested = (ins.get("NestedInstallerType") or top_nested or "").lower()
        if t not in KEEP_TYPES or (t == "zip" and nested not in ("portable", "exe")):
            continue
        url, sha = ins.get("InstallerUrl"), str(ins.get("InstallerSha256", "")).lower()
        if url and len(sha) == 64:
            yield url, sha, t


def handle(pid, url, want_sha, cfg, max_bytes):
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    data = r.raw.read(max_bytes + 1, decode_content=True)
    if len(data) > max_bytes or hashlib.sha256(data).hexdigest() != want_sha:
        return []
    blobs = []
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist()[:200]:
                if info.file_size < max_bytes and info.filename.lower().endswith((".exe", ".dll")):
                    b = z.read(info)
                    if pe_parity.is_pe(b):
                        blobs.append((b, f"{url}!{info.filename}"))
    elif pe_parity.is_pe(data):
        blobs.append((data, url))
    rows = []
    bucket, region = cfg["s3"]["bucket"], cfg["s3"]["region"]
    for b, origin in blobs[: cfg["benign_sources"]["max_per_group"]]:
        sha = pe_parity.sha256(b)
        s3io.put_bytes(bucket, f"{cfg['s3']['benign_raw_prefix']}winget/{sha}", b, region)
        rows.append([sha, f"winget:{pid}", origin.replace(",", "_"), len(b), ""])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--max-packages", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config()
    max_bytes = cfg["benign_sources"]["max_file_mb"] * 2 ** 20
    mani = latest_installer_manifests(Path(args.repo))
    print(f"packages with installer manifests: {len(mani):,}")
    jobs = []
    for pid, path in list(mani.items())[: args.max_packages]:
        seen = set()
        for url, sha, _ in installers(path):
            if sha not in seen:
                seen.add(sha)
                jobs.append((pid, url, sha))
                break                       # one installer (first arch) per package
    rows, done = [], set()
    with ThreadPoolExecutor(args.threads) as ex:
        futs = [ex.submit(handle, pid, url, sha, cfg, max_bytes) for pid, url, sha in jobs]
        for i, f in enumerate(as_completed(futs)):
            try:
                for row in f.result():
                    if row[0] not in done:
                        done.add(row[0])
                        rows.append(row)
            except Exception:                # dead links, TLS errors, rate limits
                pass
            if i % 250 == 0:
                print(f"{i}/{len(futs)} downloads, {len(rows)} PE files")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["sha256", "group", "origin", "size", "os"])
    w.writerows(rows)
    s3io.put_bytes(cfg["s3"]["bucket"], "benign/index/winget.csv", buf.getvalue().encode(), cfg["s3"]["region"])
    print(f"uploaded {len(rows)} benign PE files from winget")


if __name__ == "__main__":
    main()
