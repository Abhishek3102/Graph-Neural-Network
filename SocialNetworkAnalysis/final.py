import pickle
import networkx as nx
import streamlit as st

with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

with open("analysis_results.pkl", "rb") as f:
    results = pickle.load(f)

partition = results["partition"]
pagerank = results["pagerank"]
degree_centrality = results["degree_centrality"]
betweenness_centrality = results["betweenness_centrality"]

st.title("Social Network Analysis")

node_id = st.number_input("Enter Student ID:", min_value=0, max_value=len(G.nodes) - 1, step=1)

if st.button("Analyze"):
    # Community detection
    community = partition[node_id]
    st.write(f"**Community**: {community}")
    
    # Centrality measures
    st.write(f"**PageRank**: {pagerank[node_id]:.4f}")
    st.write(f"**Degree Centrality**: {degree_centrality[node_id]:.4f}")
    st.write(f"**Betweenness Centrality**: {betweenness_centrality[node_id]:.4f}")
    
    # Recommendations: Nearest neighbors
    neighbors = list(G.neighbors(node_id))
    st.write(f"**Closest Connections (Neighbors)**: {neighbors}")
    
    st.write("**Community Visualization**")
    subgraph_nodes = [n for n in G.nodes if partition[n] == community]
    subgraph = G.subgraph(subgraph_nodes)
    pos = nx.spring_layout(subgraph)
    st.pyplot(nx.draw(subgraph, pos, with_labels=True, node_color=[partition[n] for n in subgraph.nodes]))
