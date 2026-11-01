"""Load config.yaml with MOE__section__key env overrides."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]          # pipeline_v2/
REPO = ROOT.parent                                  # moepaper/

COLS_TYPE = ["adware", "flooder", "ransomware", "dropper", "spyware", "packed",
             "crypto_miner", "file_infector", "installer", "worm", "downloader"]
EXPERTS = ["malconv", "safe", "graph"]              # order of masks / gate outputs everywhere


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def load_config(path=None):
    path = Path(path or os.environ.get("MOE_CONFIG", ROOT / "config.yaml"))
    cfg = yaml.safe_load(path.read_text())
    for k, v in os.environ.items():
        if not k.startswith("MOE__"):
            continue
        keys = k[len("MOE__"):].split("__")
        node = cfg
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        node[keys[-1]] = _coerce(v)
    ld = Path(cfg["dataset"]["local_dir"])
    cfg["dataset"]["local_dir"] = str(ld if ld.is_absolute() else ROOT / ld)
    return cfg
