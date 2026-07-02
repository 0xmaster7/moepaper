# generate_functional_embeddings.py

import numpy as np
from sklearn.preprocessing import normalize

def generate_functional_embedding(node_embeddings):

    if node_embeddings is None:
        return None

    node_embeddings = np.array(node_embeddings)

    # Mean pooling
    graph_embedding = np.mean(node_embeddings, axis=0)

    # Normalize
    graph_embedding = normalize(
        graph_embedding.reshape(1, -1)
    )[0]

    return graph_embedding

