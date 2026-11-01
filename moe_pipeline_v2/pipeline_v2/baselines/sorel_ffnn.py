"""Baseline 2: SOREL-20M's FFNN (nets.PENetwork) retrained on your split.

Architecture as released by Sophos: [512, 512, 128] Linear -> LayerNorm -> ELU ->
Dropout(0.05); heads: malware (sigmoid), tags (128->64->64->11, sigmoid) and,
when detection counts exist, a count head (Poisson on the vendor-detection count).
Loss = BCE(malware) + BCE(tags) [+ Poisson(count)], as in SOREL's train.py.
The 128-d penultimate layer is exported as an embedding so the FFNN can also be
compared on kNN detection / behaviour retrieval against the MoE.

    python baselines/sorel_ffnn.py --seed 42
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _base import ember_transform, load, save_preds


class PENetwork(nn.Module):
    def __init__(self, d_in, n_tags, layers=(512, 512, 128), counts=True):
        super().__init__()
        mods, prev = [], d_in
        for h in layers:
            mods += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU(), nn.Dropout(0.05)]
            prev = h
        self.base = nn.Sequential(*mods)
        self.malware = nn.Linear(prev, 1)
        self.tags = nn.Sequential(nn.Linear(prev, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, n_tags))
        self.count = nn.Linear(prev, 1) if counts else None

    def forward(self, x):
        z = self.base(x)
        return {"z": z, "malware": self.malware(z).squeeze(-1), "tags": self.tags(z),
                "count": self.count(z).squeeze(-1) if self.count is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg, split, D, tr, va, te = load(args.split)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.from_numpy(ember_transform(D["X_ember"], tr))
    y = torch.from_numpy(D["y"].astype(np.float32))
    Y = torch.from_numpy(D["Y_type"])
    use_counts = bool((D["detections"] >= 0).all())          # vendor-detection counts from meta.db
    cnt = torch.from_numpy(np.maximum(D["detections"], 0).astype(np.float32))
    model = PENetwork(X.shape[1], Y.shape[1], counts=use_counts).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def loss_on(idx, train):
        model.train(train)
        tot, n = 0.0, 0
        perm = np.random.permutation(idx) if train else idx
        for s in range(0, len(perm), 512):
            b = perm[s:s + 512]
            xb, yb, Yb, cb = X[b].to(dev), y[b].to(dev), Y[b].to(dev), cnt[b].to(dev)
            with torch.set_grad_enabled(train):
                o = model(xb)
                loss = (F.binary_cross_entropy_with_logits(o["malware"], yb)
                        + F.binary_cross_entropy_with_logits(o["tags"], Yb))
                if use_counts:
                    loss = loss + 0.1 * F.poisson_nll_loss(o["count"], cb, log_input=True)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b); n += len(b)
        return tot / n

    best, wait = np.inf, 0
    for ep in range(args.epochs):
        tl, vl = loss_on(tr, True), loss_on(va, False)
        print(f"ep {ep + 1} train {tl:.4f} val {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, wait = vl, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(state)
    model.eval()
    S, T, Z = [], [], []
    with torch.no_grad():
        for s in range(0, len(X), 4096):
            o = model(X[s:s + 4096].to(dev))
            S.append(torch.sigmoid(o["malware"]).cpu()); T.append(torch.sigmoid(o["tags"]).cpu()); Z.append(o["z"].cpu())
    save_preds("sorel_ffnn", args.seed, split, D, torch.cat(S).numpy(), torch.cat(T).numpy(), torch.cat(Z).numpy())


if __name__ == "__main__":
    main()
