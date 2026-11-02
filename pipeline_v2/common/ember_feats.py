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


def _patch_thrember():
    """Compatibility fixes for thrember + signify 0.8.x (applied once).

    1. thrember slices `signed_data.certificates[:-1]`; in signify 0.8 the
       CertificateStore is no longer a list, so every *signed* file raised
       TypeError and lost all EMBER features (35% of benign in the pilot).
       Give CertificateStore list-style indexing again.
    2. Any other failure inside the Authenticode block now marks the file as
       parse_error=1 (thrember's own convention) instead of discarding all
       2568 features of the sample.
    """
    try:
        from signify.x509.context import CertificateStore
        if not hasattr(CertificateStore, "__getitem__"):
            CertificateStore.__getitem__ = lambda self, i: self.data[i]
    except Exception:  # noqa: BLE001
        pass
    import thrember.features as tf
    cls = getattr(tf, "AuthenticodeSignature", None)
    if cls is not None and not getattr(cls.raw_features, "_moe_safe", False):
        orig = cls.raw_features

        def safe_raw_features(self, bytez, pe):
            try:
                return orig(self, bytez, pe)
            except Exception:  # noqa: BLE001
                return {"num_certs": 0, "self_signed": 0, "empty_program_name": 0, "no_countersigner": 0,
                        "parse_error": 1, "chain_max_depth": 0, "latest_signing_time": 0,
                        "signing_time_diff": 0}
        safe_raw_features._moe_safe = True
        cls.raw_features = safe_raw_features


def get_extractor(kind="thrember"):
    global _extractor
    if _extractor is None:
        if kind == "thrember":
            import thrember
            _patch_thrember()
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
