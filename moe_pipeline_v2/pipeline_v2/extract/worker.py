"""Stage A worker (CPU spot instances): manifest chunk -> feature parquet.

Per sample, entirely in RAM (+ a short-lived copy in /dev/shm for radare2):
  1. stream the original bytes from S3 (SOREL: zlib, anonymous; benign: your bucket)
  2. model_view  = Machine/Subsystem zeroed for every class  -> EMBER features
  3. analysis_view = Machine restored                        -> radare2 (one pass)
        -> SAFE token ids per function (SAFE's own normalisation)
        -> call graph + ACFG node attributes (GNN baseline input)
        -> LDP graph embedding (+ legacy node2vec if enabled)
  4. model_view bytes -> MalConv2 GCT (pretrained, frozen) -> 256-d penultimate
     embedding + the checkpoint's own malware logit (zero-shot baseline)
  5. append a row; at the end of the chunk write one parquet to S3
SAFE embeddings are computed later from the stored token ids (embed/embed_safe.py).

Resumable and safe on spot: a chunk is skipped if its output exists, claimed
with an S3 lease, and released if the instance gets a 2-minute interruption
notice. Re-launching an instance simply continues where the fleet stopped.

    python extract/worker.py                       # process chunks until none left
    python extract/worker.py --max-chunks 1        # pilot: time one chunk
    python extract/worker.py --local-manifest m.parquet --local-out out/   # no S3 (tests)
"""
import argparse
import json
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
import pyarrow as pa                                        # noqa: E402
import pyarrow.parquet as pq                                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import ember_feats, graph_feats, pe_parity, s3io   # noqa: E402
from common.config import REPO, load_config                     # noqa: E402

R2_SCRIPT = ROOT / "common" / "r2_extract.py"
SHM = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()

# ------------------------------------------------------------ per-process state
_G = {}


def _init(cfg, vocab_path):
    _G["cfg"] = cfg
    with open(vocab_path) as fh:
        _G["i2id"] = json.load(fh)
    _G["unk_x"] = _G["i2id"]["X_UNK"] + 1
    _G["unk_a"] = _G["i2id"]["A_UNK"] + 1
    try:
        ember_feats.get_extractor(cfg["extract"]["ember_extractor"])
    except Exception as e:  # noqa: BLE001
        print("EMBER extractor unavailable:", e, flush=True)
    import torch
    torch.set_num_threads(1)
    _G["malconv"] = load_malconv(cfg)


def load_malconv(cfg, device="cpu"):
    """MalConv2 GCT with the authors' released checkpoint (what your repo already uses)."""
    import torch
    sys.path.insert(0, str(REPO))
    from MalConv2.MalConvGCT_nocat import MalConvGCT
    e = cfg["embed"]
    m = MalConvGCT(channels=e["malconv_channels"], window_size=e["malconv_window"], stride=e["malconv_stride"])
    ck = torch.load(REPO / "MalConv2" / "malconvGCT_nocat.checkpoint", map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"], strict=False)   # unused legacy keys only
    return m.eval().to(device)


def malconv_embed(model, model_view_bytes, max_bytes, layer="penult", device="cpu"):
    import torch
    x = np.frombuffer(model_view_bytes[:max_bytes], dtype=np.uint8).astype(np.int64) + 1   # 0 = padding
    with torch.no_grad():
        logits, penult, post_conv = model(torch.from_numpy(x)[None].to(device))
    emb = (penult if layer == "penult" else post_conv)[0].float().cpu().numpy()
    lg = logits[0].float().cpu().numpy()
    return emb, float(lg[1] - lg[0])


def _token_ids(tokens):
    i2id, out, unk = _G["i2id"], [], 0
    for t in tokens:
        v = i2id.get(t)
        if v is None:
            unk += 1
            out.append(_G["unk_a"] if t.startswith("A_") else _G["unk_x"])
        else:
            out.append(v + 1)                 # +1: SAFE reserves 0 for padding
    return out, unk


def _run_r2(path, level, timeout, max_funcs, max_insn):
    proc = subprocess.Popen([sys.executable, str(R2_SCRIPT), path, level, str(max_funcs), str(max_insn)],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return json.loads(out or b"{}")
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)       # kills r2 too, not just the python wrapper
        proc.communicate()
        return {"ok": False, "error": f"timeout@{level}"}
    except json.JSONDecodeError:
        return {"ok": False, "error": "bad-json"}


def process_row(row):
    cfg = _G["cfg"]
    ex = cfg["extract"]
    rec = {"sha256": row["sha256"], "label": int(row["label"]), "ok_pe": False, "ok_ember": False,
           "ok_r2": False, "r2_level": "", "arch": "", "bits": 0, "n_funcs": 0, "n_nodes": 0,
           "n_edges": 0, "unk_rate": np.nan, "err": "", "secs": 0.0,
           "ember": None, "safe_tokens": None, "node_x": None, "edge_src": None, "edge_dst": None,
           "ldp": None, "n2v": None, "malconv": None, "malconv_logit": None}
    t0 = time.time()
    try:
        raw = s3io.get_sample_bytes(row, cfg["s3"]["region"])
        if not pe_parity.is_pe(raw):
            rec["err"] = "not-pe"
            return rec
        rec["ok_pe"] = True
        mv, av = pe_parity.model_view(raw), pe_parity.analysis_view(raw)
        del raw
        try:
            rec["ember"] = ember_feats.feature_vector(mv, ex["ember_extractor"]).tolist()
            rec["ok_ember"] = True
        except Exception as e:  # noqa: BLE001
            rec["err"] += f"ember:{type(e).__name__};"
        try:
            emb, lg = malconv_embed(_G["malconv"], mv, ex["malconv_max_bytes"], cfg["embed"]["malconv_layer"])
            rec["malconv"], rec["malconv_logit"] = emb.tolist(), lg
        except Exception as e:  # noqa: BLE001
            rec["err"] += f"malconv:{type(e).__name__};"
        del mv

        fd, path = tempfile.mkstemp(dir=SHM)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(av)
            res = {"ok": False}
            for level, tmo in zip(ex["r2_levels"], ex["r2_timeouts_s"]):
                res = _run_r2(path, level, tmo, ex["max_functions"], ex["safe_max_instructions"])
                if res.get("ok") and res.get("functions"):
                    break
        finally:
            os.unlink(path)

        if res.get("ok") and res.get("functions"):
            funcs = res["functions"]
            ids, n_tok, n_unk = [], 0, 0
            for f in funcs:
                t, u = _token_ids(f["tokens"])
                if t:
                    ids.append(t)
                    n_tok += len(t)
                    n_unk += u
            names, X, ei = graph_feats.build_graph(funcs, res["cg_edges"])
            rec.update(ok_r2=True, r2_level=res["level"], arch=str(res.get("arch") or ""),
                       bits=int(res.get("bits") or 0), n_funcs=len(funcs), n_nodes=len(names),
                       n_edges=int(ei.shape[1]), unk_rate=n_unk / max(n_tok, 1),
                       safe_tokens=ids, node_x=X.tolist(), edge_src=ei[0].tolist(), edge_dst=ei[1].tolist(),
                       ldp=graph_feats.ldp_embedding(X, ei, ex["ldp_bins"]).tolist())
            n2c = ex["node2vec"]
            if n2c.get("enabled"):
                try:
                    v = graph_feats.node2vec_embedding(ei, len(names), n2c["dimensions"], n2c["walk_length"],
                                                       n2c["num_walks"], n2c["window"], n2c["max_nodes"])
                    rec["n2v"] = None if v is None else v.tolist()
                except Exception as e:  # noqa: BLE001
                    rec["err"] += f"n2v:{type(e).__name__};"
        else:
            rec["err"] += f"r2:{res.get('error', 'no-functions')};"
    except Exception as e:  # noqa: BLE001
        rec["err"] += f"fatal:{type(e).__name__}:{e}"[:300]
        traceback.print_exc()
    rec["secs"] = round(time.time() - t0, 2)
    return rec


SCHEMA = pa.schema([
    ("sha256", pa.string()), ("label", pa.int8()), ("ok_pe", pa.bool_()), ("ok_ember", pa.bool_()),
    ("ok_r2", pa.bool_()), ("r2_level", pa.string()), ("arch", pa.string()), ("bits", pa.int16()),
    ("n_funcs", pa.int32()), ("n_nodes", pa.int32()), ("n_edges", pa.int32()), ("unk_rate", pa.float32()),
    ("err", pa.string()), ("secs", pa.float32()),
    ("ember", pa.list_(pa.float32())), ("safe_tokens", pa.list_(pa.list_(pa.int32()))),
    ("node_x", pa.list_(pa.list_(pa.float32()))), ("edge_src", pa.list_(pa.int32())),
    ("edge_dst", pa.list_(pa.int32())), ("ldp", pa.list_(pa.float32())), ("n2v", pa.list_(pa.float32())),
    ("malconv", pa.list_(pa.float32())), ("malconv_logit", pa.float32()),
])


def to_table(recs):
    cols = {f.name: [r.get(f.name) for r in recs] for f in SCHEMA}
    return pa.Table.from_pydict(cols, schema=SCHEMA)


# ------------------------------------------------------------ spot interruption watcher
STOP = threading.Event()


def _watch_spot():
    while not STOP.is_set():
        try:
            tok_req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                             headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
            tok = urllib.request.urlopen(tok_req, timeout=2).read().decode()
            req = urllib.request.Request("http://169.254.169.254/latest/meta-data/spot/instance-action",
                                         headers={"X-aws-ec2-metadata-token": tok})
            urllib.request.urlopen(req, timeout=2)          # 200 => interruption scheduled
            print("SPOT INTERRUPTION NOTICE - releasing lease and stopping", flush=True)
            STOP.set()
            return
        except Exception:                                   # 404 / not on EC2
            pass
        time.sleep(5)


def run_chunk(df, pool):
    rows = df.to_dict("records")
    recs = []
    for r in pool.map(process_row, rows, chunksize=1):
        recs.append(r)
        if STOP.is_set():
            return None
    return recs


def summarise(recs, name):
    d = pd.DataFrame([{k: r[k] for k in ("label", "ok_pe", "ok_ember", "ok_r2", "unk_rate", "secs")} for r in recs])
    print(f"[{name}] n={len(d)} pe={d.ok_pe.mean():.3f} ember={d.ok_ember.mean():.3f} "
          f"r2={d.ok_r2.mean():.3f} (mal {d[d.label == 1].ok_r2.mean():.3f} / ben {d[d.label == 0].ok_r2.mean():.3f}) "
          f"unk={d.unk_rate.mean():.3f} sec/sample={d.secs.mean():.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--local-manifest", default=None)
    ap.add_argument("--local-out", default=None)
    ap.add_argument("--vocab", default=str(REPO / "safe_final" / "data" / "i2v" / "word2id.json"))
    args = ap.parse_args()
    cfg = load_config()
    n_workers = cfg["extract"]["workers"] or os.cpu_count()
    wid = f"{socket.gethostname()}-{os.getpid()}"
    threading.Thread(target=_watch_spot, daemon=True).start()

    with ProcessPoolExecutor(n_workers, initializer=_init, initargs=(cfg, args.vocab)) as pool:
        if args.local_manifest:
            recs = run_chunk(pd.read_parquet(args.local_manifest), pool)
            Path(args.local_out).mkdir(parents=True, exist_ok=True)
            pq.write_table(to_table(recs), Path(args.local_out) / "features_local.parquet")
            summarise(recs, "local")
            return

        bucket, region = cfg["s3"]["bucket"], cfg["s3"]["region"]
        chunks = sorted(k for k in s3io.list_keys(bucket, cfg["s3"]["manifests_prefix"] + "chunks/", region)
                        if k.endswith(".parquet"))
        random.Random(wid).shuffle(chunks)                 # spread workers over the chunk list
        done = 0
        for key in chunks:
            if STOP.is_set() or (args.max_chunks and done >= args.max_chunks):
                break
            name = key.rsplit("/", 1)[-1]
            out_key = cfg["s3"]["features_prefix"] + name
            lease = cfg["s3"]["leases_prefix"] + "A/" + name + ".json"
            if s3io.exists(bucket, out_key, region) or not s3io.acquire_lease(bucket, lease, wid, region=region):
                continue
            try:
                t0 = time.time()
                recs = run_chunk(s3io.read_parquet(bucket, key, region).to_pandas(), pool)
                if recs is None:
                    break
                s3io.write_parquet(to_table(recs), bucket, out_key, region)
                summarise(recs, f"{name} {time.time() - t0:.0f}s")
                done += 1
            finally:
                s3io.release_lease(bucket, lease, region)
    STOP.set()
    print(f"worker {wid} finished {done} chunks", flush=True)


if __name__ == "__main__":
    main()
