"""Dual-gate MoE over precomputed expert embeddings (your architecture), plus the
fusion variants needed for the ablation table.

fusion
  dual_gate    gate_bin -> z_bin -> classifier ; gate_retr -> z_retr -> retrieval head  (paper model)
  single_gate  one gate feeds both heads
  concat       masked concat of projected experts (+ masks) -> MLP          (no gating)
  mean         average of the available projected experts                    (no gating)
  expert:<i>   one expert only (0 MalConv, 1 SAFE, 2 graph)                  (single-modality)
gate_input
  ember_masks  standardised EMBER + availability masks (paper)
  masks        availability masks only   (does EMBER help the router?)
  ember        EMBER only                (do the masks help?)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2n(x, eps=1e-12):
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


class GatingMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, num_experts=3, dropout=0.1, tau=1.0, bound=5.0):
        super().__init__()
        self.tau, self.bound = tau, bound
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, num_experts))

    def forward(self, x):
        logits = torch.tanh(self.net(x) / self.bound) * self.bound      # bounded logits (your fix)
        return F.softmax(logits / self.tau, dim=-1)


def renorm(p, masks):
    pm = p * masks
    return pm / pm.sum(-1, keepdim=True).clamp_min(1e-9)


class MoE(nn.Module):
    def __init__(self, gate_dim, expert_dims, d_common=256, fusion="dual_gate"):
        super().__init__()
        self.fusion = fusion
        self.k = len(expert_dims)
        self.proj = nn.ModuleList([nn.Linear(d, d_common) for d in expert_dims])
        if fusion in ("dual_gate", "single_gate"):
            self.gate_bin = GatingMLP(gate_dim, num_experts=self.k)
            if fusion == "dual_gate":
                self.gate_retr = GatingMLP(gate_dim, num_experts=self.k)
        if fusion == "concat":
            self.fuse = nn.Sequential(nn.Linear(self.k * d_common + self.k, d_common), nn.ReLU(),
                                      nn.Linear(d_common, d_common))
        self.bin_clf = nn.Linear(d_common, 2)
        self.retrieval_head = nn.Sequential(nn.Linear(d_common, d_common), nn.ReLU(),
                                            nn.Linear(d_common, d_common))

    def forward(self, g, experts, masks):
        v = [self.proj[i](experts[i]) * masks[:, i:i + 1] for i in range(self.k)]
        V = torch.stack(v, 1)                                            # (B, k, d)
        p_bin = p_retr = None
        if self.fusion in ("dual_gate", "single_gate"):
            p_bin = renorm(self.gate_bin(g), masks)
            p_retr = renorm(self.gate_retr(g), masks) if self.fusion == "dual_gate" else p_bin
            z_bin = (p_bin.unsqueeze(-1) * V).sum(1)
            z_retr = (p_retr.unsqueeze(-1) * V).sum(1)
        elif self.fusion == "concat":
            z_bin = z_retr = self.fuse(torch.cat([V.flatten(1), masks], 1))
        elif self.fusion == "mean":
            z_bin = z_retr = V.sum(1) / masks.sum(1, keepdim=True).clamp_min(1.0)
        elif self.fusion.startswith("expert:"):
            z_bin = z_retr = V[:, int(self.fusion.split(":")[1])]
        else:
            raise ValueError(self.fusion)
        return {"bin_logits": self.bin_clf(z_bin), "h": l2n(self.retrieval_head(z_retr)),
                "p_bin": p_bin, "p_retr": p_retr}


# ------------------------------------------------------------------ losses (unchanged from moe_train.py)
def multilabel_infonce(emb, Y, y_bin, temperature=0.07):
    emb = l2n(emb)
    sim = emb @ emb.T / temperature
    B = emb.size(0)
    eye = torch.eye(B, device=emb.device)
    Yb = (Y > 0).float()
    ben = (y_bin == 0).float()
    pos = (((Yb @ Yb.T) > 0).float() + ben[:, None] * ben[None, :] > 0).float() * (1 - eye)
    exp_sim = torch.exp(sim - sim.max(1, keepdim=True).values.detach()) * (1 - eye)
    num = (exp_sim * pos).sum(1)
    den = exp_sim.sum(1).clamp_min(1e-12)
    valid = (pos.sum(1) > 0).float()
    return (-torch.log((num / den).clamp_min(1e-12)) * valid).sum() / valid.sum().clamp_min(1.0)


def load_balance_loss(p, masks):
    target = masks.mean(0)
    target = target / target.sum().clamp_min(1e-9)
    return ((p.mean(0) - target) ** 2).sum()
