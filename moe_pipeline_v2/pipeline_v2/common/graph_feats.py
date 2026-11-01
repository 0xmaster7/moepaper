"""Call-graph features: attributed graph (for the GNN baseline) + two graph-level
embeddings for the MoE graph expert.

graph expert options
--------------------
node2vec (legacy): node2vec is fitted *separately on every binary*, so each
  binary's vectors live in their own randomly rotated space. Mean-pooling them
  and comparing across binaries is not well defined. It is kept so the legacy
  expert can be reproduced and ablated.
ldp (recommended): Local Degree Profile histograms (Cai & Wang, 2018) on fixed
  global bins + structural statistics + mean ACFG instruction-category mix.
  Deterministic, needs no fitting, directly comparable across binaries.
"""
import numpy as np

from .r2_extract import CAT_NAMES

NODE_ATTR_NAMES = (["log_ninstrs", "log_size"] + [f"frac_{c}" for c in CAT_NAMES]
                   + ["log_indeg", "log_outdeg", "is_external"])


def build_graph(functions, cg_edges):
    """Return node names, node attribute matrix (N, A) and edge index (2, E)."""
    names, idx = [], {}

    def nid(name):
        if name not in idx:
            idx[name] = len(names)
            names.append(name)
        return idx[name]

    fattr = {}
    for f in functions:
        i = nid(f["name"])
        n_seen = max(1, sum(f["cats"]))
        fattr[i] = [np.log1p(f["ninstrs"]), np.log1p(f["size"])] + [c / n_seen for c in f["cats"]]
    src, dst = [], []
    for a, b in cg_edges:
        ia, ib = nid(a), nid(b)
        if ia != ib:
            src.append(ia)
            dst.append(ib)
    n = len(names)
    edge_index = np.unique(np.array([src, dst], dtype=np.int64).reshape(2, -1), axis=1) if src else np.zeros((2, 0), np.int64)
    indeg = np.bincount(edge_index[1], minlength=n) if n else np.zeros(0)
    outdeg = np.bincount(edge_index[0], minlength=n) if n else np.zeros(0)
    width = 2 + len(CAT_NAMES)
    X = np.zeros((n, len(NODE_ATTR_NAMES)), dtype=np.float32)
    for i in range(n):
        base = fattr.get(i)
        X[i, :width] = base if base is not None else 0.0
        X[i, width] = np.log1p(indeg[i])
        X[i, width + 1] = np.log1p(outdeg[i])
        X[i, width + 2] = 0.0 if base is not None else 1.0       # external / import node
    return names, X, edge_index


def ldp_embedding(X, edge_index, bins=16):
    """Local degree profile histograms + structural stats. Fixed size."""
    n = X.shape[0]
    stats_dim = 6 + len(CAT_NAMES)
    if n == 0:
        return np.zeros(5 * bins + stats_dim, dtype=np.float32)
    und = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    deg = np.bincount(und[0], minlength=n).astype(np.float64)
    nb_min = np.full(n, np.inf)
    nb_max = np.zeros(n)
    nb_sum = np.zeros(n)
    nb_sq = np.zeros(n)
    if und.shape[1]:
        d_nb = deg[und[1]]
        np.minimum.at(nb_min, und[0], d_nb)
        np.maximum.at(nb_max, und[0], d_nb)
        np.add.at(nb_sum, und[0], d_nb)
        np.add.at(nb_sq, und[0], d_nb ** 2)
    nb_min[np.isinf(nb_min)] = 0
    safe_deg = np.maximum(deg, 1)
    nb_mean = nb_sum / safe_deg
    nb_std = np.sqrt(np.maximum(nb_sq / safe_deg - nb_mean ** 2, 0))
    edges = np.linspace(0, np.log1p(1e4), bins + 1)
    hists = []
    for v in (deg, nb_min, nb_max, nb_mean, nb_std):
        h, _ = np.histogram(np.log1p(v), bins=edges)
        hists.append(h / n)
    n_e = edge_index.shape[1]
    width = 2 + len(CAT_NAMES)
    internal = X[:, width + 2] == 0
    cat_mix = X[internal, 2:width].mean(axis=0) if internal.any() else np.zeros(len(CAT_NAMES))
    stats = [np.log1p(n), np.log1p(n_e), n_e / max(n * (n - 1), 1), 1 - internal.mean(),
             float(X[internal, 0].mean()) if internal.any() else 0.0, float(deg.max()) / n]
    return np.concatenate(hists + [np.asarray(stats), cat_mix]).astype(np.float32)


def node2vec_embedding(edge_index, n_nodes, dimensions=128, walk_length=30, num_walks=20,
                       window=10, max_nodes=5000, seed=42):
    """Legacy expert (see module docstring). Returns L2-normalised mean or None."""
    if n_nodes == 0 or edge_index.shape[1] == 0:
        return None
    import networkx as nx
    from node2vec import Node2Vec
    G = nx.DiGraph()
    G.add_edges_from(edge_index.T.tolist())
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() > max_nodes:
        keep = sorted(G.degree, key=lambda t: -t[1])[:max_nodes]
        G = G.subgraph([k for k, _ in keep]).copy()
    if G.number_of_nodes() == 0:
        return None
    G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
    model = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks,
                     workers=1, quiet=True, seed=seed).fit(window=window, min_count=1, seed=seed, workers=1)
    v = np.mean([model.wv[n] for n in G.nodes()], axis=0)
    return (v / max(np.linalg.norm(v), 1e-12)).astype(np.float32)
