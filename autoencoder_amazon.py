import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

import numpy as np

import os

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from scipy import sparse


# Dataset para carregar a matriz esparsa

class SparseTensorDataset(Dataset):

    def __init__(self, data_matrix):

        self.data_matrix = data_matrix


    def __len__(self):

        return self.data_matrix.shape[0]


    def __getitem__(self, idx):

        return torch.tensor(self.data_matrix[idx].toarray().flatten(), dtype=torch.float32)


# Autoencoder

class Autoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim):

        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(

            nn.Linear(input_dim, input_dim // 200),

            nn.ReLU(),

            nn.Linear(input_dim // 200, input_dim // 800),

            nn.ReLU(),

            nn.Linear(input_dim // 800, latent_dim)

        )

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, input_dim // 800),

            nn.ReLU(),

            nn.Linear(input_dim // 800, input_dim // 200),

            nn.ReLU(),

            nn.Linear(input_dim // 200, input_dim)

        )


    def forward(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded, encoded


# --- Pré-processamento ---

AMAZON_META_PATH = 'amazon-meta.txt'

MAX_PRODUCTS_TO_PARSE = 10000 


def parse_amazon_meta(filepath, max_products=None):

    from collections import defaultdict

    products_similarities = defaultdict(set)

    current_asin = None

    product_count = 0


    with open(filepath, 'r', encoding='latin-1') as f:

        for line in f:

            line = line.strip()

            if line.startswith("ASIN:"):

                current_asin = line.split(":")[1].strip()

                if max_products and product_count >= max_products:

                    break

                product_count += 1

            elif line.startswith("similar:") and current_asin:

                similar_asins = line.split(":")[1].strip().split()[1:]

                for sim_asin in similar_asins:

                    products_similarities[current_asin].add(sim_asin)


    return products_similarities


matrix_file_path = f'similarity_matrix_{MAX_PRODUCTS_TO_PARSE}.npz'


if os.path.exists(matrix_file_path):

    loader = np.load(matrix_file_path)

    data_matrix_csr = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

else:

    all_products_similarities = parse_amazon_meta(AMAZON_META_PATH, max_products=MAX_PRODUCTS_TO_PARSE)

    all_asins_in_graph = set(all_products_similarities.keys())

    for similar_set in all_products_similarities.values():

        all_asins_in_graph.update(similar_set)


    asin_to_idx = {asin: i for i, asin in enumerate(sorted(list(all_asins_in_graph)))}

    similarity_matrix_sparse = sparse.lil_matrix((len(asin_to_idx), len(asin_to_idx)), dtype=np.float32)


    for asin, similar_asins_set in all_products_similarities.items():

        asin_idx = asin_to_idx[asin]

        for sim_asin in similar_asins_set:

            if sim_asin in asin_to_idx:

                sim_asin_idx = asin_to_idx[sim_asin]

                similarity_matrix_sparse[asin_idx, sim_asin_idx] = 1.0


    data_matrix_csr = similarity_matrix_sparse.tocsr()

    sparse.save_npz(matrix_file_path, data_matrix_csr)


# --- Clustering ---

embeddings_save_path = 'latent_embeddings.npy'

num_clusters = 5


if os.path.exists(embeddings_save_path):

    latent_embeddings = np.load(embeddings_save_path)

else:

    dataset = SparseTensorDataset(data_matrix_csr)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = data_matrix_csr.shape[1]

    latent_dim = 64

    model = Autoencoder(input_dim, latent_dim)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)


    model_save_path = 'autoencoder_model.pth'

    if os.path.exists(model_save_path):

        model.load_state_dict(torch.load(model_save_path, map_location=device))

        model.eval()

    else:

        for epoch in range(50):

            total_loss = 0

            for data in dataloader:

                inputs = data.to(device)

                optimizer.zero_grad()

                outputs, _ = model(inputs)

                loss = criterion(outputs, inputs)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:

                print(f'Epoch [{epoch+1}/50], Loss: {total_loss / len(dataloader):.6f}')


        torch.save(model.state_dict(), model_save_path)


    latent_embeddings = []

    model.eval()

    with torch.no_grad():

        for data in dataloader:

            inputs = data.to(device)

            _, encoded_output = model(inputs)

            latent_embeddings.append(encoded_output.cpu().numpy())


    latent_embeddings = np.concatenate(latent_embeddings, axis=0)

    np.save(embeddings_save_path, latent_embeddings)


# --- K-Means Clustering ---

clusters_save_path = 'cluster_labels.npy'


if os.path.exists(clusters_save_path):

    clusters = np.load(clusters_save_path)

else:

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')

    clusters = kmeans.fit_predict(latent_embeddings)

    np.save(clusters_save_path, clusters)


# --- PCA para visualização 3D ---

embeddings_3d_save_path = 'embeddings_3d_for_plot.npy'


if os.path.exists(embeddings_3d_save_path):

    embeddings_3d = np.load(embeddings_3d_save_path)

else:

    pca = PCA(n_components=3)

    embeddings_3d = pca.fit_transform(latent_embeddings)

    np.save(embeddings_3d_save_path, embeddings_3d)


# --- Plotagem 3D ---

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(12, 10))

ax = fig.add_subplot(111, projection='3d')


for i in range(num_clusters):

    cluster_points = embeddings_3d[clusters == i]

    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i+1}', s=20, alpha=0.7)


ax.set_title('Clusterização 3D de Similaridades de Produtos', fontsize=16)

ax.set_xlabel('Componente Principal 1', fontsize=12)

ax.set_ylabel('Componente Principal 2', fontsize=12)

ax.set_zlabel('Componente Principal 3', fontsize=12)

ax.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True)

plt.tight_layout()

plt.show()