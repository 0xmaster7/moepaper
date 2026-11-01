"""EMBER feature extraction on the *model view* (header-parity) bytes.

Both classes go through the same extractor version, so the gate input and the
LightGBM / FFNN baselines cannot pick up extractor-version artefacts. (Mixing
SOREL's precomputed EMBER v2 vectors for malware with self-extracted vectors for
benign would create exactly that artefact.)

thrember = EMBER2024 feature version 3 (pip install from
https://github.com/FutureComputing4AI/EMBER2024). Its dimension matches the
2568-column EMBER block in your current merged.csv.
"""
import numpy as np

_extractor = None


def get_extractor(kind="thrember"):
    global _extractor
    if _extractor is None:
        if kind == "thrember":
            import thrember
            _extractor = thrember.PEFeatureExtractor()
        elif kind == "ember2018":
            import ember
            _extractor = ember.PEFeatureExtractor(feature_version=2)
        else:
            raise ValueError(kind)
    return _extractor


def feature_vector(model_view_bytes, kind="thrember"):
    ex = get_extractor(kind)
    return np.asarray(ex.feature_vector(model_view_bytes), dtype=np.float32)


def dim(kind="thrember"):
    return int(get_extractor(kind).dim)
