from flask import Flask, render_template, request, jsonify
import pandas as pd
import networkx as nx
import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import torch.nn.functional as F

app = Flask(__name__)

df = pd.read_csv("movies.csv")
df = df[["id", "title", "genres", "popularity", "vote_average"]]
df['genres'] = df['genres'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else "Unknown")

num_users = len(df)
user_ids = [f"user_{i}" for i in range(num_users)]

G = nx.Graph()
for idx, row in df.iterrows():
    movie_id = row['id']
    G.add_node(movie_id, node_type="movie", title=row['title'], genres=row['genres'], 
               popularity=row['popularity'], vote_average=row['vote_average'])
    
    user_id = user_ids[idx]
    G.add_node(user_id, node_type="user", title=None, genres=None, vote_average=0.0, popularity=0.0)
    
    interaction_strength = (row['vote_average'] / 10.0) * (row['popularity'] / 1000.0)
    G.add_edge(user_id, movie_id, weight=interaction_strength)

for node in G.nodes:
    G.nodes[node].setdefault("node_type", "unknown")
    G.nodes[node].setdefault("title", None)
    G.nodes[node].setdefault("genres", None)
    G.nodes[node].setdefault("popularity", 0.0)
    G.nodes[node].setdefault("vote_average", 0.0)

node_mapping = {node: idx for idx, node in enumerate(G.nodes)}

edges = [(node_mapping[u], node_mapping[v]) for u, v in G.edges]

data = from_networkx(G)
data.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

data.x = torch.rand((data.num_nodes, 16))

transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=True)
train_data, val_data, test_data = transform(data)

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_features, 64)
        self.conv2 = GCNConv(64, 32)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
train_data = train_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    pred = model.decode(z, train_data.edge_label_index)
    loss = criterion(pred, train_data.edge_label.float())
    loss.backward()
    optimizer.step()

def recommend_movies(user_id):
    model.eval()
    with torch.no_grad():
        if user_id not in G.nodes:
            return []
        z = model.encode(train_data.x, train_data.edge_index)
        recommendations = []
        for movie_id in df['id']:
            if (user_id, movie_id) not in G.edges:
                edge = torch.tensor([[list(G.nodes).index(user_id)], [list(G.nodes).index(movie_id)]], device=device)
                score = model.decode(z, edge).item()
                recommendations.append((movie_id, score))
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:5]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form.get('user_id')
    if user_input in G.nodes:
        recommended_movies = recommend_movies(user_input)
        if recommended_movies:
            movie_list = []
            for movie_id, score in recommended_movies:
                movie_info = df[df['id'] == movie_id]
                interaction_weight = G[user_input][movie_id]['weight'] if (user_input, movie_id) in G.edges else 0
                movie_list.append({
                    'title': movie_info['title'].values[0],
                    'genres': movie_info['genres'].values[0],
                    'score': round(score, 2),
                    'interaction': round(interaction_weight, 2),
                    'popularity': movie_info['popularity'].values[0]
                })
            return render_template('index.html', user_input=user_input, movies=movie_list)
        else:
            return render_template('index.html', user_input=user_input, error="No recommendations available.")
    else:
        return render_template('index.html', user_input=user_input, error="Invalid User ID. Please enter a valid User ID.")

if __name__ == "__main__":
    app.run(debug=True)
