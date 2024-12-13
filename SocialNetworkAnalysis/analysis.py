import pickle
import networkx as nx
from community import community_louvain

# Load graph
with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

# Community detection (Louvain method)
partition = community_louvain.best_partition(G)

# Compute centrality measures
pagerank = nx.pagerank(G)
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Save analysis results
results = {
    "partition": partition,
    "pagerank": pagerank,
    "degree_centrality": degree_centrality,
    "betweenness_centrality": betweenness_centrality,
}

with open("analysis_results.pkl", "wb") as f:
    pickle.dump(results, f)
