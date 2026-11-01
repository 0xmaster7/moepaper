"""End-to-end smoke test on synthetic data (no AWS, no radare2, no malware).

    python tests/smoke_test.py            # ~2-4 minutes on a laptop CPU

Checks: PE header parity, SAFE tokenisation vs the real vocabulary, graph
features, then dataset -> MoE variants -> evaluation -> every baseline ->
tables, using a synthetic dataset whose labels are learnable.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import graph_feats, pe_parity                    # noqa: E402
from common.config import REPO                               # noqa: E402
from common.r2_extract import safe_token                     # noqa: E402


def fake_pe(magic=0x10B, machine=0x14C, subsystem=2):
    b = bytearray(1024)
    b[0:2] = b"MZ"
    b[0x3C:0x40] = (0x80).to_bytes(4, "little")
    b[0x80:0x84] = b"PE\0\0"
    b[0x84:0x86] = machine.to_bytes(2, "little")
    b[0x98:0x9A] = magic.to_bytes(2, "little")
    b[0x98 + 0x44:0x98 + 0x46] = subsystem.to_bytes(2, "little")
    return bytes(b)


def test_parity():
    for magic, mach in ((0x10B, 0x14C), (0x20B, 0x8664)):
        pe = fake_pe(magic, mach)
        mv = pe_parity.model_view(pe)
        assert pe_parity.header_fields(mv)["machine"] == 0 and pe_parity.header_fields(mv)["subsystem"] == 0
        sorel_like = mv                                     # SOREL ships files already zeroed
        av = pe_parity.analysis_view(sorel_like)
        assert pe_parity.header_fields(av)["machine"] == mach, "machine not restored for r2"
        assert pe_parity.model_view(pe) == pe_parity.model_view(sorel_like), "benign/malware views differ"
    assert not pe_parity.is_pe(b"not a pe at all" * 10)
    print("parity ok")


def test_safe_tokens():
    vocab = json.loads((REPO / "safe_final" / "data" / "i2v" / "word2id.json").read_text())
    ops = [{"mnemonic": "mov", "opex": {"operands": [{"type": "reg", "value": "ebp"}, {"type": "reg", "value": "esp"}]}},
           {"mnemonic": "xor", "opex": {"operands": [{"type": "reg", "value": "eax"}, {"type": "reg", "value": "eax"}]}},
           {"mnemonic": "mov", "opex": {"operands": [{"type": "reg", "value": "esi"},
                                                     {"type": "mem", "base": "rdi", "scale": 1, "disp": 72}]}}]
    toks = [safe_token(o, "x86") for o in ops]
    hits = [t in vocab for t in toks]
    print("tokens", toks, "in vocab:", hits)
    assert all(hits), "SAFE normalisation no longer matches the vocabulary"
    legacy = "mov ebp, esp"                               # what safe_final/embed_text.py produced
    assert legacy not in vocab and "X_" not in legacy, "legacy path would map to X_UNK"
    print("SAFE tokenisation ok (legacy strings all map to X_UNK; new ones hit the vocabulary)")


def test_graph():
    funcs = [{"name": f"f{i}", "size": 50 + i, "ninstrs": 10 + i, "cats": list(np.random.randint(0, 5, 9))}
             for i in range(30)]
    edges = [[f"f{i}", f"f{(i * 7) % 30}"] for i in range(30)] + [["f0", "sub.KERNEL32.dll_CreateFileW"]]
    names, X, ei = graph_feats.build_graph(funcs, edges)
    v = graph_feats.ldp_embedding(X, ei, 16)
    assert X.shape == (31, len(graph_feats.NODE_ATTR_NAMES)) and np.isfinite(v).all()
    assert X[-1, -1] == 1.0                                 # import node flagged external
    n2v = graph_feats.node2vec_embedding(ei, len(names), dimensions=16, num_walks=2, walk_length=5)
    assert n2v is not None and abs(np.linalg.norm(n2v) - 1) < 1e-4
    print("graph ok, ldp dim", v.shape[0], "| node2vec ok")


def make_synthetic(d: Path, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.35).astype(np.int64)
    Y = ((rng.random((n, 11)) < 0.25) & (y[:, None] == 1)).astype(np.float32)
    sig = y[:, None] * 3.0 + 0.5 * (Y @ rng.normal(size=(11, 1)))
    def emb(dim, cov):
        E = rng.normal(size=(n, dim)).astype(np.float32) + sig.astype(np.float32) * rng.normal(size=(1, dim)).astype(np.float32)
        m = (rng.random(n) < cov).astype(np.float32)
        return E * m[:, None], m
    Em, mm = emb(256, 1.0); Es, ms = emb(100, 0.8); El, ml = emb(94, 0.7); En, mn = emb(128, 0.7)
    X = np.abs(rng.normal(size=(n, 60)) * 100 + y[:, None] * 30).astype(np.float32)
    sha = np.array([f"{i:064x}" for i in range(n)])
    fp = "synthetic"
    np.savez_compressed(d / "dataset.npz", sha256=sha, y=y, Y_type=Y, Y_counts=Y * 3, X_ember=X,
                        E_malconv=Em, m_malconv=mm, E_safe=Es, m_safe=ms, E_ldp=El, m_ldp=ml, E_n2v=En, m_n2v=mn,
                        malconv_logit=(y * 2 - 1 + rng.normal(size=n)).astype(np.float32),
                        detections=(Y.sum(1) * 5).astype(np.float32), cols_type=np.array(["t"] * 11), fingerprint=fp)
    idx = rng.permutation(n)
    tr, va, te = np.sort(idx[:2000]), np.sort(idx[2000:2500]), np.sort(idx[2500:])
    for s in ("temporal", "random"):
        np.savez(d / f"splits_{s}.npz", train_idx=tr, val_idx=va, test_idx=te, fingerprint=fp)
    sizes = rng.integers(0, 40, n)
    nptr = np.concatenate([[0], np.cumsum(sizes)])
    NX = rng.normal(size=(nptr[-1], 14)).astype(np.float16) + np.repeat(y, sizes)[:, None].astype(np.float16)
    es, eptr = [], [0]
    for i in range(n):
        k = sizes[i] * 2 if sizes[i] else 0
        e = rng.integers(0, max(sizes[i], 1), (2, k)).astype(np.int32)
        es.append(e); eptr.append(eptr[-1] + k)
    np.savez(d / "graphs.npz", node_x=NX, edges=np.concatenate(es, 1), node_ptr=nptr, edge_ptr=np.array(eptr), fingerprint=fp)


def run(args, env):
    print("+", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED: {args}")
    return r.stdout


def main():
    test_parity(); test_safe_tokens(); test_graph()
    tmp = Path(tempfile.mkdtemp())
    data = tmp / "data"; data.mkdir()
    make_synthetic(data)
    res = ROOT / "results"
    backup = None
    if res.exists():
        backup = tmp / "results_backup"; shutil.move(str(res), backup)
    env = dict(os.environ, MOE__dataset__local_dir=str(data))
    try:
        env_seeds = dict(env)
        for name, extra in (("moe_full", []), ("abl_concat", ["--fusion", "concat"]),
                            ("expert_safe", ["--fusion", "expert:1"]), ("abl_n2v", ["--graph-expert", "node2vec"])):
            run(["model/train_moe.py", "--name", name, "--epochs", "3", *extra], env_seeds)
            run(["eval/evaluate.py", "--run", str(res / "runs" / name / "seed42")], env_seeds)
        run(["baselines/ember_lgbm.py", "--rounds", "30", "--tags"], env)
        run(["baselines/sorel_ffnn.py", "--epochs", "3"], env)
        run(["baselines/cfg_gnn.py", "--epochs", "2"], env)
        run(["baselines/malconv_zeroshot.py"], env)
        for b, s in (("ember_lgbm", 42), ("sorel_ffnn", 42), ("cfg_gnn", 42), ("malconv2_zeroshot", 0)):
            run(["eval/evaluate.py", "--preds", str(res / "baselines" / b / f"seed{s}" / "preds.npz")], env)
        out = run(["report/make_tables.py"], env)
        assert "EMBER + LightGBM" in out and "MoE (classifier head)" in out
        full = json.loads((res / "runs" / "moe_full" / "seed42" / "eval.json").read_text())
        print("\nsynthetic MoE head ROC-AUC:", round(full["detection_head"]["roc_auc"], 3),
              "| kNN Acc@1:", round(full["detection_knn"]["accuracy@1"], 3))
        assert full["detection_head"]["roc_auc"] > 0.8, "model failed to learn an easy synthetic task"
        if os.environ.get("SMOKE_SHOW_TABLES"):
            print(out)
        print("\nSMOKE TEST PASSED")
    finally:
        shutil.rmtree(res, ignore_errors=True)
        if backup:
            shutil.move(str(backup), res)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
