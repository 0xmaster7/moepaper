"""Train the MoE (or an ablation variant) on dataset.npz and export outputs for
every sample, so evaluation never needs to reload a checkpoint.

    python model/train_moe.py --name moe_full --seed 42
    python model/train_moe.py --name abl_concat --fusion concat --seed 42
    python model/train_moe.py --name abl_masks_only --gate-input masks
    python model/train_moe.py --name abl_no_dropout --expert-dropout 0

Outputs: results/runs/<name>/seed<seed>/{outputs.npz, model.pt, log.json}

Changes vs moe_train.py (behaviour otherwise kept):
  * EMBER gate input is log-scaled and standardised with TRAIN statistics
    (raw EMBER values span ~10 orders of magnitude, which saturates the gate).
  * Graph expert selectable (ldp | node2vec).
  * Split loaded from build_dataset's splits_<mode>.npz, checked by fingerprint.
  * Checkpoint selected on validation binary CE (as before); test never touched.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import load_config                       # noqa: E402
from model.moe import MoE, load_balance_loss, multilabel_infonce   # noqa: E402


def load_data(cfg, split_mode, graph_expert):
    d = Path(cfg["dataset"]["local_dir"])
    D = dict(np.load(d / "dataset.npz", allow_pickle=False))
    S = np.load(d / f"splits_{split_mode}.npz")
    assert str(S["fingerprint"]) == str(D["fingerprint"]), "splits do not match dataset.npz - rebuild"
    g = "ldp" if graph_expert == "ldp" else "n2v"
    experts = [D["E_malconv"], D["E_safe"], D[f"E_{g}"]]
    masks = np.stack([D["m_malconv"], D["m_safe"], D[f"m_{g}"]], 1).astype(np.float32)
    return D, S["train_idx"], S["val_idx"], S["test_idx"], experts, masks


def ember_transform(X, train_idx):
    Z = np.sign(X) * np.log1p(np.abs(X))
    mu, sd = Z[train_idx].mean(0), Z[train_idx].std(0) + 1e-6
    return ((Z - mu) / sd).astype(np.float32)


def standardise(E, m, train_idx):
    on = train_idx[m[train_idx] > 0]
    mu, sd = E[on].mean(0), E[on].std(0) + 1e-6
    return (((E - mu) / sd) * m[:, None]).astype(np.float32)


def sampler_weights(y, Y, idx):
    """Your scheme: inverse tag frequency inside malware, malware total == benign total."""
    fam = Y[idx].sum(0).clip(1)
    w = np.ones(len(idx), np.float32)
    for i, j in enumerate(idx):
        if y[j] == 1:
            t = np.flatnonzero(Y[j] > 0)
            w[i] = np.mean(1.0 / fam[t]) if len(t) else 0.1
    mal = y[idx] == 1
    w[mal] *= (~mal).sum() / max(w[mal].sum(), 1e-9)
    return w


def w_retr_schedule(ep, w_max, warm=15, ramp=20):
    return 0.0 if ep < warm else w_max * min(1.0, (ep - warm) / max(ramp, 1))


def batch_loss(model, b, args, w_retr, train):
    g, e0, e1, e2, m, y, Y = b
    if train and args.expert_dropout > 0:
        keep = (torch.rand_like(m) > args.expert_dropout).float()
        nm = m * keep
        m = torch.where(nm.sum(1, keepdim=True) == 0, m, nm)
    out = model(g, [e0, e1, e2], m)
    ce = F.cross_entropy(out["bin_logits"], y)
    loss = ce
    if out["p_retr"] is not None and args.fusion == "dual_gate":
        loss = loss + args.w_bal * load_balance_loss(out["p_retr"], m)
    if w_retr > 0:
        mal = y == 1
        if mal.sum() >= 2:
            loss = loss + w_retr * multilabel_infonce(out["h"][mal], Y[mal], y[mal], args.temperature)
    return loss, ce


@torch.no_grad()
def export(model, tensors, device, bs=1024):
    model.eval()
    P, H, PB, PR = [], [], [], []
    n = tensors[0].shape[0]
    for s in range(0, n, bs):
        g, e0, e1, e2, m = [t[s:s + bs].to(device) for t in tensors]
        out = model(g, [e0, e1, e2], m)
        P.append(F.softmax(out["bin_logits"], -1)[:, 1].cpu())
        H.append(out["h"].cpu())
        if out["p_bin"] is not None:
            PB.append(out["p_bin"].cpu())
            PR.append(out["p_retr"].cpu())
    res = {"prob": torch.cat(P).numpy(), "h": torch.cat(H).numpy()}
    if PB:
        res["p_bin"], res["p_retr"] = torch.cat(PB).numpy(), torch.cat(PR).numpy()
    return res


def main():
    cfg = load_config()
    tc = cfg["train"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="moe_full")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default=cfg["dataset"]["split_mode"])
    ap.add_argument("--fusion", default="dual_gate")
    ap.add_argument("--gate-input", default="ember_masks", choices=["ember_masks", "masks", "ember"])
    ap.add_argument("--graph-expert", default=tc["graph_expert"], choices=["ldp", "node2vec"])
    ap.add_argument("--expert-dropout", type=float, default=tc["expert_dropout"])
    ap.add_argument("--epochs", type=int, default=tc["epochs"])
    ap.add_argument("--lr", type=float, default=tc["lr"])
    ap.add_argument("--w-retr-max", type=float, default=tc["w_retr_max"])
    ap.add_argument("--w-bal", type=float, default=tc["w_bal"])
    ap.add_argument("--temperature", type=float, default=tc["temperature"])
    ap.add_argument("--out", default=str(ROOT / "results" / "runs"))
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D, tr, va, te, experts, masks = load_data(cfg, args.split, args.graph_expert)
    y, Y = D["y"], D["Y_type"]
    Xg = ember_transform(D["X_ember"], tr)
    if args.gate_input == "masks":
        G = masks
    elif args.gate_input == "ember":
        G = Xg
    else:
        G = np.concatenate([Xg, masks], 1)
    experts = [standardise(E, masks[:, i], tr) for i, E in enumerate(experts)]
    T = [torch.from_numpy(G), *[torch.from_numpy(E) for E in experts], torch.from_numpy(masks)]
    yt, Yt = torch.from_numpy(y), torch.from_numpy(Y)

    def loader(idx, train):
        ds = TensorDataset(*[t[idx] for t in T], yt[idx], Yt[idx])
        if train:
            w = torch.from_numpy(sampler_weights(y, Y, idx))
            return DataLoader(ds, batch_size=tc["batch_size"],
                              sampler=WeightedRandomSampler(w, len(idx), replacement=True))
        return DataLoader(ds, batch_size=1024, shuffle=False)

    tl, vl = loader(tr, True), loader(va, False)
    model = MoE(G.shape[1], [E.shape[1] for E in experts], tc["d_common"], args.fusion).to(device)
    groups = []
    for name, mod in model.named_children():
        groups.append({"params": mod.parameters(), "lr": args.lr * (5 if name == "gate_retr" else 1)})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=8, min_lr=1e-6)

    out_dir = Path(args.out) / args.name / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best, log = float("inf"), []
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        w_retr = w_retr_schedule(ep, args.w_retr_max)
        tot = 0.0
        for b in tl:
            b = [x.to(device) for x in b]
            loss, _ = batch_loss(model, b, args, w_retr, True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        model.eval()
        vl_loss = vl_ce = 0.0
        with torch.no_grad():
            for b in vl:
                b = [x.to(device) for x in b]
                l, ce = batch_loss(model, b, args, w_retr, False)
                vl_loss += l.item(); vl_ce += ce.item()
        vl_loss /= len(vl); vl_ce /= len(vl)
        sched.step(vl_loss)
        if vl_ce < best:
            best = vl_ce
            torch.save(model.state_dict(), out_dir / "model.pt")
        log.append({"epoch": ep + 1, "train_loss": tot / len(tl), "val_loss": vl_loss, "val_ce": vl_ce,
                    "w_retr": w_retr, "secs": round(time.time() - t0, 1)})
        print(f"ep {ep + 1:3d} train {tot / len(tl):.4f} val {vl_loss:.4f} val_ce {vl_ce:.4f} w_retr {w_retr:.2f}",
              flush=True)

    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    res = export(model, T, device)
    np.savez_compressed(out_dir / "outputs.npz", **res, split=args.split, fingerprint=str(D["fingerprint"]),
                        masks=masks)
    (out_dir / "log.json").write_text(json.dumps({"args": vars(args), "best_val_ce": best, "epochs": log}, indent=1))
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
