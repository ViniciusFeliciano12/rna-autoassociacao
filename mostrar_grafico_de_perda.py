import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Definir o caminho para o arquivo de perdas
losses_file_path = 'autoencoder_training_losses.npy'

# 2. Verificar se o arquivo existe antes de tentar carregá-lo
if os.path.exists(losses_file_path):
    # 3. Carregar o array NumPy que contém as perdas
    training_losses = np.load(losses_file_path)

    print(f"Perdas de treinamento carregadas com sucesso de: {losses_file_path}")
    print("Primeiras 5 perdas:", training_losses[:5])
    print("Últimas 5 perdas:", training_losses[-5:])
    print("Número total de épocas:", len(training_losses))

    # 4. Exibir o gráfico da curva de perda
    plt.figure(figsize=(12, 7)) # Define o tamanho da figura
    
    # Plota a linha da perda ao longo das épocas
    # O eixo X vai de 1 até o número total de épocas
    plt.plot(range(1, len(training_losses) + 1), training_losses,
             marker='o',       # Adiciona marcadores para cada ponto (perda por época)
             markersize=4,     # Tamanho dos marcadores
             linestyle='-',    # Estilo da linha (sólida)
             color='blue',     # Cor da linha
             label='MSE Loss') # Legenda para a linha

    # 5. Adicionar Títulos e Rótulos aos Eixos
    plt.title('Curva de Perda (MSE) do Autoencoder durante o Treinamento', fontsize=16)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)

    # 6. Adicionar Grade e Legenda
    plt.grid(True) # Adiciona uma grade ao gráfico para facilitar a leitura
    plt.legend(fontsize=10) # Exibe a legenda
    
    # 7. Ajustar os ticks do eixo X para mostrar épocas a cada X (opcional)
    # Se você tiver 50 épocas, mostrar ticks a cada 5 ou 10 pode ser bom.
    if len(training_losses) > 10:
        plt.xticks(np.arange(0, len(training_losses) + 1, max(1, len(training_losses) // 10))) # Ticks a cada 10% das épocas
    else:
        plt.xticks(np.arange(0, len(training_losses) + 1, 1)) # Se poucas épocas, mostrar todas

    plt.tight_layout() # Ajusta o layout para evitar sobreposição de elementos
    plt.show() # Exibe o gráfico

else:
    print(f"Erro: O arquivo '{losses_file_path}' não foi encontrado.")
    print("Certifique-se de que o Autoencoder foi treinado e o arquivo de perdas foi salvo.")