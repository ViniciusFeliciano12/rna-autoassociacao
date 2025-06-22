import re
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os
from scipy import sparse

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")

def parse_amazon_meta(filepath, max_products=None):
    """
    Parses the amazon-meta.txt file to extract ASINs and their similar products.
    Args:
        filepath (str): Path to the amazon-meta.txt file.
        max_products (int, optional): Maximum number of products to parse. Useful for testing.
    Returns:
        dict: A dictionary where keys are ASINs and values are sets of similar ASINs.
    """
    products_similarities = defaultdict(set)
    current_asin = None
    product_count = 0

    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if line.startswith("ASIN:"):
                    current_asin = line.split(":")[1].strip()
                    if max_products and product_count >= max_products:
                        break
                    product_count += 1
                elif line.startswith("similar:") and current_asin:
                    similar_asins = line.split(":")[1].strip().split()
                    if similar_asins:
                        # Pegar todos os elementos A PARTIR DO SEGUNDO (índice 1) O primeiro elemento é a contagem de similaridades
                        actual_similar_asins = similar_asins[1:]
                        for sim_asin in actual_similar_asins:
                            products_similarities[current_asin].add(sim_asin)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{filepath}' não foi encontrado. Por favor, verifique o caminho.")
        exit()
    return products_similarities

# --- Configurações Iniciais ---
AMAZON_META_PATH = 'amazon-meta.txt'
MAX_PRODUCTS_TO_PARSE = 10000 # Reduza este valor se ainda tiver problemas de memória (SIGKILL)
batch_size = 32 

# --- 1. Pré-processamento do Dataset e Geração da Matriz de Similaridade ---
print(f"Iniciando o processo de clusterização do dataset amazon-meta.txt...")
print(f"1. Pré-processando o dataset '{AMAZON_META_PATH}' para extrair ASINs e similaridades...")

# Verifique se a matriz já existe em disco para evitar reprocessamento
matrix_file_path = f'similarity_matrix_{MAX_PRODUCTS_TO_PARSE}.npz' # Usamos npz para sparse

if os.path.exists(matrix_file_path):
    print(f"Carregando matriz de similaridade esparsa de '{matrix_file_path}'...")
    loader = np.load(matrix_file_path)
    data_matrix_csr = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    num_total_asins = data_matrix_csr.shape[0]
    print(f"Matriz de similaridade esparsa carregada com shape: {data_matrix_csr.shape}")
else:
    all_products_similarities = parse_amazon_meta(AMAZON_META_PATH, max_products=MAX_PRODUCTS_TO_PARSE)

    if not all_products_similarities:
        print("Nenhum dado de similaridade foi extraído. Verifique o arquivo e o MAX_PRODUCTS_TO_PARSE.")
        exit()

    print(f"Total de ASINs com similaridades extraídas (máximo de {MAX_PRODUCTS_TO_PARSE} produtos): {len(all_products_similarities)}")

    all_asins_in_graph = set(all_products_similarities.keys())
    for similar_set in all_products_similarities.values():
        all_asins_in_graph.update(similar_set)

    asin_to_idx = {asin: i for i, asin in enumerate(sorted(list(all_asins_in_graph)))}
    idx_to_asin = {i: asin for asin, i in asin_to_idx.items()}
    num_total_asins = len(asin_to_idx)

    print(f"Número total de ASINs únicos considerados no grafo de similaridade: {num_total_asins}")

    print(f"\n2. Gerando vetores binários de similaridade (matriz de adjacência) usando uma matriz esparsa...")
    similarity_matrix_sparse = sparse.lil_matrix((num_total_asins, num_total_asins), dtype=np.float32)

    for asin, similar_asins_set in all_products_similarities.items():
        if asin in asin_to_idx:
            asin_idx = asin_to_idx[asin]
            for sim_asin in similar_asins_set:
                if sim_asin in asin_to_idx:
                    sim_asin_idx = asin_to_idx[sim_asin]
                    similarity_matrix_sparse[asin_idx, sim_asin_idx] = 1.0

    data_matrix_csr = similarity_matrix_sparse.tocsr()
    print(f"Matriz de similaridade esparsa criada com shape: {data_matrix_csr.shape}")
    print(f"Número de elementos não-zero: {data_matrix_csr.nnz}")
    estimated_memory_gb = (data_matrix_csr.data.nbytes + data_matrix_csr.indptr.nbytes + data_matrix_csr.indices.nbytes) / (1024**3)
    print(f"Uso de memória estimado da matriz esparsa: {estimated_memory_gb:.4f} GB")

    # Salvando a matriz esparsa para uso futuro
    sparse.save_npz(matrix_file_path, data_matrix_csr)
    print(f"Matriz de similaridade esparsa salva como '{matrix_file_path}'")

    del all_products_similarities, asin_to_idx, idx_to_asin, all_asins_in_graph, similarity_matrix_sparse # Liberar memória

class SparseTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix

    def __len__(self):
        return self.data_matrix.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.data_matrix[idx].toarray().flatten(), dtype=torch.float32)

dataset = SparseTensorDataset(data_matrix_csr)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"DataLoader criado com batch_size={dataloader.batch_size}")


# --- 3. Autoencoder (PyTorch) para Redução de Dimensionalidade ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # Camadas intermediárias ajustadas para input_dim grande
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

input_dim = data_matrix_csr.shape[1]
latent_dim = 64
epochs = 50
learning_rate = 0.001

model = Autoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\nModelo Autoencoder movido para: {device}")
print(f"Autoencoder configurado com input_dim={input_dim}, latent_dim={latent_dim}")

# --- Treinar ou Carregar o Modelo do Autoencoder ---
model_save_path = 'autoencoder_model.pth'

if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval() # Importante para inferência
    print(f"Modelo Autoencoder carregado de '{model_save_path}'")
else:
    print(f"\nIniciando treinamento do Autoencoder por {epochs} épocas...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, data in enumerate(dataloader):
            inputs = data.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

    print("Treinamento do Autoencoder concluído.")
    torch.save(model.state_dict(), model_save_path)
    print(f"Modelo Autoencoder salvo como '{model_save_path}'")

# --- Extrair ou Carregar os Embeddings Latentes ---
embeddings_save_path = 'latent_embeddings.npy'

if os.path.exists(embeddings_save_path):
    latent_embeddings = np.load(embeddings_save_path)
    print(f"Embeddings latentes carregados de '{embeddings_save_path}'")
else:
    print("\nExtraindo embeddings latentes do modelo treinado...")
    model.eval()
    latent_embeddings = []
    with torch.no_grad():
        for data in dataloader:
            inputs = data.to(device)
            _, encoded_output = model(inputs)
            latent_embeddings.append(encoded_output.cpu().numpy())

    latent_embeddings = np.concatenate(latent_embeddings, axis=0)
    print(f"Dimensão das representações latentes (embeddings): {latent_embeddings.shape}")
    np.save(embeddings_save_path, latent_embeddings)
    print(f"Embeddings latentes salvos como '{embeddings_save_path}'")


# --- 4. Aplicar Clustering (K-Means) ou Carregar Clusters ---
clusters_save_path = 'cluster_labels.npy'
num_clusters = 10 # Ajuste este valor

if os.path.exists(clusters_save_path):
    clusters = np.load(clusters_save_path)
    print(f"Rótulos dos clusters carregados de '{clusters_save_path}'")
else:
    print(f"\n4. Aplicando K-Means com {num_clusters} clusters nos embeddings latentes...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(latent_embeddings)

    print(f"Clusters encontrados para {len(np.unique(clusters))} grupos: {np.unique(clusters)}")
    np.save(clusters_save_path, clusters)
    print(f"Rótulos dos clusters salvos como '{clusters_save_path}'")


# --- 5. Reduzir para 3D e Plotar ou Carregar Dados 3D ---
embeddings_3d_save_path = 'embeddings_3d_for_plot.npy'

if os.path.exists(embeddings_3d_save_path):
    embeddings_3d = np.load(embeddings_3d_save_path)
    print(f"Dados 3D para plotagem carregados de '{embeddings_3d_save_path}'")
else:
    print("\n5. Reduzindo dimensionalidade para 3D para visualização usando PCA...")
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(latent_embeddings)

    np.save(embeddings_3d_save_path, embeddings_3d)
    print(f"Dados 3D para plotagem salvos como '{embeddings_3d_save_path}'")

# --- Visualização 3D ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for i in range(num_clusters):
    cluster_points = embeddings_3d[clusters == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
               label=f'Cluster {i+1}', s=20, alpha=0.7)

ax.set_title('Clusterização 3D de Similaridades de Produtos (Embeddings do Autoencoder)', fontsize=16)
ax.set_xlabel('Componente Principal 1', fontsize=12)
ax.set_ylabel('Componente Principal 2', fontsize=12)
ax.set_zlabel('Componente Principal 3', fontsize=12)
ax.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nProcesso de clusterização concluído. O gráfico 3D com os clusters foi gerado.")
print("Lembre-se de que a qualidade dos clusters e a visualização podem ser aprimoradas")
print("ajustando MAX_PRODUCTS_TO_PARSE, latent_dim, num_clusters e a arquitetura do Autoencoder.")