import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

df = pd.read_csv("data/03_Clustering_Marketing.csv")

feature_columns = [
    "gradyear", "gender", "age", "NumberOffriends", "basketball", "football", "soccer", "softball", "volleyball",
    "swimming", "cheerleading", "baseball", "tennis", "sports", "cute", "sex", "sexy", "hot", "kissed", "dance", "band",
    "marching", "music", "rock", "god", "church", "jesus", "bible", "hair", "dress", "blonde", "mall", "shopping", "clothes",
    "hollister", "abercrombie", "die", "death", "drunk", "drugs"
]

features = df[feature_columns]

# Normalize numerical features (non-categorical)
for col in features.columns:
    if pd.api.types.is_numeric_dtype(features[col]):
        features[col] = (features[col] - features[col].mean()) / features[col].std()

# Convert categorical columns to numerical 
for col in features.columns:
    if pd.api.types.is_categorical_dtype(features[col]) or features[col].dtype == object:
        features[col] = pd.factorize(features[col])[0]

similarity = cosine_similarity(features)

# Create the graph
G = nx.Graph()

# Add nodes with attributes
for idx, row in df.iterrows():
    G.add_node(idx, **row.to_dict())

# Add edges based on similarity
threshold = 0.5  # Define a similarity threshold
for i in range(len(similarity)):
    for j in range(i + 1, len(similarity)):
        if similarity[i, j] > threshold:
            G.add_edge(i, j, weight=similarity[i, j])

# Save graph
with open("graph.pkl", "wb") as f:
    pickle.dump(G, f)
