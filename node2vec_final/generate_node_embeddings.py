# generate_node_embeddings.py

from node2vec import Node2Vec

def generate_node_embeddings(G, dimensions=128):

    if len(G.nodes()) == 0:
        return None

    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=30,
        num_walks=200,
        workers=4
    )

    model = node2vec.fit(window=10, min_count=1)

    embeddings = [model.wv[str(n)] for n in G.nodes()]

    return embeddings

