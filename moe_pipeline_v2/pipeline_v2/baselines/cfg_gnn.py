"""Baseline 3: graph neural network on the attributed call graph.

Same radare2 call graph the graph expert uses, with ACFG-style node attributes
(instruction-category mix, size, degrees, external flag), in the spirit of MAGIC
(Yan et al., DSN 2019). Model: 3-layer GIN, mean+max readout, malware + tag heads.
Pure PyTorch (no torch_geometric needed). Samples without a graph get NaN and
are reported on the covered subset (evaluate.py prints coverage).

    python build/build_dataset.py --graphs     # once
    python baselines/cfg_gnn.py --seed 42
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _base import load, save_preds

MAX_NODES = 4000


class GIN(nn.Module):
    def __init__(self, d_in, d=128, layers=3, n_tags=11):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.eps = nn.Parameter(torch.zeros(layers))
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, d))
                                   for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(0.2))
        self.malware = nn.Linear(d, 1)
        self.tags = nn.Linear(d, n_tags)

    def forward(self, x, ei, seg, n_graphs):
        h = F.relu(self.inp(x))
        src, dst = torch.cat([ei[0], ei[1]]), torch.cat([ei[1], ei[0]])        # undirected
        for k, mlp in enumerate(self.mlps):
            agg = torch.zeros_like(h).index_add_(0, dst, h[src])
            h = F.relu(mlp((1 + self.eps[k]) * h + agg))
        cnt = torch.zeros(n_graphs, device=x.device).index_add_(0, seg, torch.ones_like(seg, dtype=h.dtype))
        mean = torch.zeros(n_graphs, h.shape[1], device=x.device).index_add_(0, seg, h) / cnt.clamp_min(1)[:, None]
        mx = torch.full((n_graphs, h.shape[1]), -1e9, device=x.device).scatter_reduce(
            0, seg[:, None].expand_as(h), h, "amax", include_self=True)
        z = self.head(torch.cat([mean, mx.clamp_min(-1e8)], 1))
        return z, self.malware(z).squeeze(-1), self.tags(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg, split, D, tr, va, te = load(args.split)
    G = np.load(Path(cfg["dataset"]["local_dir"]) / "graphs.npz")
    assert str(G["fingerprint"]) == str(D["fingerprint"])
    NX, E, nptr, eptr = G["node_x"].astype(np.float32), G["edges"], G["node_ptr"], G["edge_ptr"]
    has = (nptr[1:] - nptr[:-1]) > 0
    tr, va, te = tr[has[tr]], va[has[va]], te[has[te]]
    tr_nodes = np.concatenate([np.arange(nptr[i], min(nptr[i + 1], nptr[i] + MAX_NODES)) for i in tr[:5000]])
    mu, sd = NX[tr_nodes].mean(0), NX[tr_nodes].std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    y, Y = D["y"].astype(np.float32), D["Y_type"]

    def batch(ids):
        xs, es, seg, off = [], [], [], 0
        for gi, i in enumerate(ids):
            n = min(nptr[i + 1] - nptr[i], MAX_NODES)
            x = (NX[nptr[i]:nptr[i] + n] - mu) / sd
            e = E[:, eptr[i]:eptr[i + 1]]
            e = e[:, (e[0] < n) & (e[1] < n)]
            xs.append(x); es.append(e + off); seg.append(np.full(n, gi)); off += n
        t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(dev, dt)   # noqa: E731
        return (t(np.concatenate(xs), torch.float32), t(np.concatenate(es, 1), torch.long),
                t(np.concatenate(seg), torch.long), len(ids))

    model = GIN(NX.shape[1], n_tags=Y.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def run(idx, train):
        model.train(train)
        idx = np.random.permutation(idx) if train else idx
        tot = 0.0
        for s in range(0, len(idx), args.batch):
            b = idx[s:s + args.batch]
            with torch.set_grad_enabled(train):
                _, lm, lt = model(*batch(b))
                loss = (F.binary_cross_entropy_with_logits(lm, torch.from_numpy(y[b]).to(dev))
                        + F.binary_cross_entropy_with_logits(lt, torch.from_numpy(Y[b]).to(dev)))
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        return tot / len(idx)

    best, wait, state = np.inf, 0, None
    for ep in range(args.epochs):
        tl, vl = run(tr, True), run(va, False)
        print(f"ep {ep + 1} train {tl:.4f} val {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, wait, state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(state)
    model.eval()
    N = len(y)
    score = np.full(N, np.nan, np.float32)
    tags = np.full((N, Y.shape[1]), np.nan, np.float32)
    emb = np.zeros((N, 128), np.float32)
    ids = np.flatnonzero(has)
    with torch.no_grad():
        for s in range(0, len(ids), 256):
            b = ids[s:s + 256]
            z, lm, lt = model(*batch(b))
            score[b] = torch.sigmoid(lm).cpu().numpy()
            tags[b] = torch.sigmoid(lt).cpu().numpy()
            emb[b] = z.cpu().numpy()
    save_preds("cfg_gnn", args.seed, split, D, score, tags, emb)


if __name__ == "__main__":
    main()
