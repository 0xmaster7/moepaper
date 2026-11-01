import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import load_config                       # noqa: E402
from model.train_moe import ember_transform                 # noqa: E402


def load(split=None):
    cfg = load_config()
    split = split or cfg["dataset"]["split_mode"]
    d = Path(cfg["dataset"]["local_dir"])
    D = dict(np.load(d / "dataset.npz", allow_pickle=False))
    S = np.load(d / f"splits_{split}.npz")
    assert str(S["fingerprint"]) == str(D["fingerprint"])
    return cfg, split, D, S["train_idx"], S["val_idx"], S["test_idx"]


def save_preds(name, seed, split, D, score, tag_scores=None, emb=None, out=str(ROOT / "results" / "baselines")):
    p = Path(out) / name / f"seed{seed}"
    p.mkdir(parents=True, exist_ok=True)
    kw = {"score": score.astype(np.float32), "split": split, "fingerprint": str(D["fingerprint"])}
    if tag_scores is not None:
        kw["tag_scores"] = tag_scores.astype(np.float32)
    if emb is not None:
        kw["emb"] = emb.astype(np.float32)
    np.savez_compressed(p / "preds.npz", **kw)
    print("saved", p / "preds.npz")
    return p / "preds.npz"


__all__ = ["load", "save_preds", "ember_transform"]
