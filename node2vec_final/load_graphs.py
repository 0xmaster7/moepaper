# load_graphs.py

import networkx as nx

def load_cfg(dot_path):
    try:
        G = nx.nx_pydot.read_dot(dot_path)
        G = nx.DiGraph(G)

        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))

        if len(G.nodes()) == 0:
            return None

        return G

    except Exception as e:
        print(f"Skipping invalid CFG file: {dot_path}")
        print(f"Exact error: {e}")
        return None

