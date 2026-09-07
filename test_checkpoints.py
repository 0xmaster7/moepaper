import torch, hashlib
for s in [42, 43, 44]:
    sd = torch.load(f"best_moe_s{s}.pt", map_location="cpu")
    h = hashlib.md5(b"".join(v.cpu().numpy().tobytes() for v in sd.values())).hexdigest()
    print(s, h)