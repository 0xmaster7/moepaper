"""Baseline 0: the released MalConv2-GCT checkpoint used as-is (no training on
your data). Score = sigmoid of its own malware logit on the header-parity bytes.

    python baselines/malconv_zeroshot.py
"""
import numpy as np

from _base import load, save_preds


def main():
    cfg, split, D, tr, va, te = load()
    s = 1 / (1 + np.exp(-D["malconv_logit"].astype(np.float64)))
    s[D["m_malconv"] == 0] = np.nan
    save_preds("malconv2_zeroshot", 0, split, D, s, emb=D["E_malconv"])


if __name__ == "__main__":
    main()
