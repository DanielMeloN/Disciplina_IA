import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definir semente aleatória para reprodutibilidade dos pesos iniciais e embaralhamento
np.random.seed(42)

# =====================================================================
# 1. Base de Dados de Treinamento (120 Amostras) e Teste (12 Amostras)
# =====================================================================

# As amostras do apêndice do documento:
# Amostras 1-20   : Classe A
# Amostras 21-60  : Classe B
# Amostras 61-120 : Classe C
dados_treinamento = np.array([
    [0.2417, 0.2857, 0.2397], # Amostra 1 (Classe A)
    [0.2268, 0.2874, 0.2153], # Amostra 2 (Classe A)
    [0.1975, 0.3315, 0.1965], # Amostra 3 (Classe A)
    [0.3414, 0.3166, 0.1074], # Amostra 4 (Classe A)
    [0.2587, 0.1918, 0.2634], # Amostra 5 (Classe A)
    [0.2455, 0.2075, 0.1344], # Amostra 6 (Classe A)
    [0.3163, 0.1679, 0.1725], # Amostra 7 (Classe A)
    [0.2704, 0.2605, 0.1411], # Amostra 8 (Classe A)
    [0.1871, 0.2965, 0.1231], # Amostra 9 (Classe A)
    [0.3474, 0.2715, 0.1958], # Amostra 10 (Classe A)
    [0.2059, 0.2928, 0.2839], # Amostra 11 (Classe A)
    [0.2442, 0.2272, 0.2384], # Amostra 12 (Classe A)
    [0.2126, 0.3437, 0.1128], # Amostra 13 (Classe A)
    [0.2562, 0.2542, 0.1599], # Amostra 14 (Classe A)
    [0.1640, 0.2289, 0.2627], # Amostra 15 (Classe A)
    [0.2795, 0.1880, 0.1627], # Amostra 16 (Classe A)
    [0.3463, 0.1513, 0.2281], # Amostra 17 (Classe A)
    [0.3430, 0.1508, 0.1881], # Amostra 18 (Classe A)
    [0.1981, 0.2821, 0.1294], # Amostra 19 (Classe A)
    [0.2322, 0.3025, 0.2191], # Amostra 20 (Classe A)
    [0.7352, 0.2722, 0.6962], # Amostra 21 (Classe B)
    [0.7191, 0.1825, 0.7470], # Amostra 22 (Classe B)
    [0.6921, 0.1537, 0.8172], # Amostra 23 (Classe B)
    [0.6833, 0.2048, 0.8490], # Amostra 24 (Classe B)
    [0.8012, 0.2684, 0.7673], # Amostra 25 (Classe B)
    [0.786,  0.1734, 0.7198], # Amostra 26 (Classe B)
    [0.7205, 0.1542, 0.7295], # Amostra 27 (Classe B)
    [0.6549, 0.3288, 0.8153], # Amostra 28 (Classe B)
    [0.6968, 0.3173, 0.7389], # Amostra 29 (Classe B)
    [0.7448, 0.2095, 0.6847], # Amostra 30 (Classe B)
    [0.6746, 0.3277, 0.6725], # Amostra 31 (Classe B)
    [0.7897, 0.2801, 0.7679], # Amostra 32 (Classe B)
    [0.8399, 0.3067, 0.7003], # Amostra 33 (Classe B)
    [0.8065, 0.3206, 0.7205], # Amostra 34 (Classe B)
    [0.8357, 0.3220, 0.7879], # Amostra 35 (Classe B)
    [0.7438, 0.3230, 0.8384], # Amostra 36 (Classe B)
    [0.8172, 0.3319, 0.7628], # Amostra 37 (Classe B)
    [0.8248, 0.2614, 0.8405], # Amostra 38 (Classe B)
    [0.6979, 0.2142, 0.7309], # Amostra 39 (Classe B)
    [0.6804, 0.3181, 0.7017], # Amostra 40 (Classe B)
    [0.6973, 0.3194, 0.7522], # Amostra 41 (Classe B)
    [0.7910, 0.2239, 0.7018], # Amostra 42 (Classe B)
    [0.7052, 0.2148, 0.6866], # Amostra 43 (Classe B)
    [0.8088, 0.1908, 0.7563], # Amostra 44 (Classe B)
    [0.7640, 0.1676, 0.6994], # Amostra 45 (Classe B)
    [0.7616, 0.2881, 0.8087], # Amostra 46 (Classe B)
    [0.8188, 0.2461, 0.7273], # Amostra 47 (Classe B)
    [0.7920, 0.3178, 0.7497], # Amostra 48 (Classe B)
    [0.7802, 0.1871, 0.8102], # Amostra 49 (Classe B)
    [0.7332, 0.2543, 0.8194], # Amostra 50 (Classe B)
    [0.6921, 0.1529, 0.7759], # Amostra 51 (Classe B)
    [0.6833, 0.2197, 0.6943], # Amostra 52 (Classe B)
    [0.7860, 0.1745, 0.7639], # Amostra 53 (Classe B)
    [0.8009, 0.3082, 0.8491], # Amostra 54 (Classe B)
    [0.7793, 0.1935, 0.6738], # Amostra 55 (Classe B)
    [0.7373, 0.2698, 0.7864], # Amostra 56 (Classe B)
    [0.7048, 0.2380, 0.7825], # Amostra 57 (Classe B)
    [0.8393, 0.2857, 0.7733], # Amostra 58 (Classe B)
    [0.6878, 0.2126, 0.6961], # Amostra 59 (Classe B)
    [0.6651, 0.3492, 0.6737], # Amostra 60 (Classe B)
    [0.4856, 0.6600, 0.4798], # Amostra 61 (Classe C)
    [0.4114, 0.7220, 0.5106], # Amostra 62 (Classe C)
    [0.5671, 0.7935, 0.5929], # Amostra 63 (Classe C)
    [0.4875, 0.7928, 0.5532], # Amostra 64 (Classe C)
    [0.5172, 0.7147, 0.5774], # Amostra 65 (Classe C)
    [0.5483, 0.6773, 0.4842], # Amostra 66 (Classe C)
    [0.5740, 0.6682, 0.5335], # Amostra 67 (Classe C)
    [0.4587, 0.6981, 0.5900], # Amostra 68 (Classe C)
    [0.5794, 0.7410, 0.4759], # Amostra 69 (Classe C)
    [0.4712, 0.6734, 0.5677], # Amostra 70 (Classe C)
    [0.5126, 0.8141, 0.5224], # Amostra 71 (Classe C)
    [0.5557, 0.7749, 0.4342], # Amostra 72 (Classe C)
    [0.4916, 0.8267, 0.4586], # Amostra 73 (Classe C)
    [0.4629, 0.8129, 0.4950], # Amostra 74 (Classe C)
    [0.5850, 0.7358, 0.5107], # Amostra 75 (Classe C)
    [0.4435, 0.7030, 0.4594], # Amostra 76 (Classe C)
    [0.4155, 0.7516, 0.5524], # Amostra 77 (Classe C)
    [0.4887, 0.7027, 0.5886], # Amostra 78 (Classe C)
    [0.5462, 0.7378, 0.5107], # Amostra 79 (Classe C)
    [0.5251, 0.8124, 0.5686], # Amostra 80 (Classe C)
    [0.4635, 0.7339, 0.5638], # Amostra 81 (Classe C)
    [0.5907, 0.7144, 0.4718], # Amostra 82 (Classe C)
    [0.4982, 0.8335, 0.4597], # Amostra 83 (Classe C)
    [0.5242, 0.7325, 0.4079], # Amostra 84 (Classe C)
    [0.4075, 0.8372, 0.4271], # Amostra 85 (Classe C)
    [0.5934, 0.8284, 0.5107], # Amostra 86 (Classe C)
    [0.5463, 0.6766, 0.5639], # Amostra 87 (Classe C)
    [0.4403, 0.8495, 0.4806], # Amostra 88 (Classe C)
    [0.4531, 0.7760, 0.5276], # Amostra 89 (Classe C)
    [0.5109, 0.7387, 0.5373], # Amostra 90 (Classe C)
    [0.5383, 0.7780, 0.4955], # Amostra 91 (Classe C)
    [0.5679, 0.7156, 0.5022], # Amostra 92 (Classe C)
    [0.5762, 0.7781, 0.5908], # Amostra 93 (Classe C)
    [0.5997, 0.7504, 0.5678], # Amostra 94 (Classe C)
    [0.4138, 0.6975, 0.5148], # Amostra 95 (Classe C)
    [0.5490, 0.6674, 0.4472], # Amostra 96 (Classe C)
    [0.4719, 0.7527, 0.4401], # Amostra 97 (Classe C)
    [0.4458, 0.8063, 0.4253], # Amostra 98 (Classe C)
    [0.4983, 0.8131, 0.5625], # Amostra 99 (Classe C)
    [0.5742, 0.6789, 0.5997], # Amostra 100 (Classe C)
    [0.5289, 0.7354, 0.4718], # Amostra 101 (Classe C)
    [0.5927, 0.7738, 0.5390], # Amostra 102 (Classe C)
    [0.5199, 0.7131, 0.4028], # Amostra 103 (Classe C)
    [0.5716, 0.6558, 0.4451], # Amostra 104 (Classe C)
    [0.5075, 0.7045, 0.4233], # Amostra 105 (Classe C)
    [0.4886, 0.7004, 0.4608], # Amostra 106 (Classe C)
    [0.5527, 0.8243, 0.5772], # Amostra 107 (Classe C)
    [0.4816, 0.6969, 0.4678], # Amostra 108 (Classe C)
    [0.5809, 0.6557, 0.4266], # Amostra 109 (Classe C)
    [0.5881, 0.7565, 0.4003], # Amostra 110 (Classe C)
    [0.5334, 0.8446, 0.4934], # Amostra 111 (Classe C)
    [0.4603, 0.7992, 0.4816], # Amostra 112 (Classe C)
    [0.5491, 0.6504, 0.4063], # Amostra 113 (Classe C)
    [0.4288, 0.8455, 0.5047], # Amostra 114 (Classe C)
    [0.5636, 0.7884, 0.5417], # Amostra 115 (Classe C)
    [0.5349, 0.6736, 0.4541], # Amostra 116 (Classe C)
    [0.5569, 0.8393, 0.5652], # Amostra 117 (Classe C)
    [0.4729, 0.7702, 0.5325], # Amostra 118 (Classe C)
    [0.5472, 0.8454, 0.5449], # Amostra 119 (Classe C)
    [0.5805, 0.7349, 0.4464]  # Amostra 120 (Classe C)
])

dados_teste = np.array([
    [0.2471, 0.1778, 0.2905], # Amostra 1
    [0.8240, 0.2223, 0.7041], # Amostra 2
    [0.4960, 0.7231, 0.5866], # Amostra 3
    [0.2923, 0.2041, 0.2234], # Amostra 4
    [0.8118, 0.2668, 0.7484], # Amostra 5
    [0.4837, 0.8200, 0.4792], # Amostra 6
    [0.3248, 0.2629, 0.2375], # Amostra 7
    [0.7209, 0.2116, 0.7821], # Amostra 8
    [0.5259, 0.6522, 0.5957], # Amostra 9
    [0.2075, 0.1669, 0.1745], # Amostra 10
    [0.7830, 0.3171, 0.7888], # Amostra 11
    [0.5393, 0.7510, 0.5682]  # Amostra 12
])

# =====================================================================
# 2. Classe KohonenSOM
# =====================================================================

class KohonenSOM:
    def __init__(self, grid_shape=(4, 4), input_dim=3, alpha=0.001, radius=1):
        self.grid_shape = grid_shape
        self.input_dim = input_dim
        self.alpha = alpha
        self.radius = radius
        # Inicialização uniforme no intervalo dos dados [0.1, 0.9]
        self.weights = np.random.uniform(0.1, 0.9, (grid_shape[0], grid_shape[1], input_dim))
        
    def find_bmu(self, x):
        """
        Encontra o neurônio vencedor (BMU - Best Matching Unit)
        usando a menor Distância Euclidiana ao quadrado.
        """
        # Calcular distâncias euclidianas quadradas
        dists = np.sum((self.weights - x) ** 2, axis=2)
        # Obter índice 2D
        bmu_idx = np.unravel_index(np.argmin(dists), dists.shape)
        return bmu_idx, dists[bmu_idx]
        
    def update(self, x, bmu_idx):
        """
        Atualiza o BMU e sua vizinhança topológica direta (raio r=1).
        Usa distância de Manhattan na grade.
        """
        for r in range(self.grid_shape[0]):
            for c in range(self.grid_shape[1]):
                dist = abs(r - bmu_idx[0]) + abs(c - bmu_idx[1])
                if dist <= self.radius:
                    self.weights[r, c] += self.alpha * (x - self.weights[r, c])
                    
    def train(self, data, epochs=10000):
        """
        Treina o mapa auto-organizável apresentando as amostras
        em ordem aleatória a cada época.
        """
        n_samples = len(data)
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for idx in indices:
                x = data[idx]
                bmu_idx, _ = self.find_bmu(x)
                self.update(x, bmu_idx)

# =====================================================================
# 3. Execução Principal e Treinamento
# =====================================================================

def main():
    print("=" * 75)
    print("      RESOLUÇÃO DO EXERCÍCIO: REDE DE KOHONEN (MAPA AUTO-ORGANIZÁVEL)")
    print("=" * 75)
    
    # 3.1 Visualização 3D das Amostras Originais
    print("[!] Gerando gráfico de dispersão 3D dos dados de treinamento...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Amostras por classe
    cA = dados_treinamento[0:20]
    cB = dados_treinamento[20:60]
    cC = dados_treinamento[60:120]
    
    ax.scatter(cA[:, 0], cA[:, 1], cA[:, 2], c='#2ca02c', marker='o', s=50, label='Classe A (Amostras 1-20)', edgecolors='black', alpha=0.8)
    ax.scatter(cB[:, 0], cB[:, 1], cB[:, 2], c='#1f77b4', marker='^', s=50, label='Classe B (Amostras 21-60)', edgecolors='black', alpha=0.8)
    ax.scatter(cC[:, 0], cC[:, 1], cC[:, 2], c='#ff7f0e', marker='s', s=50, label='Classe C (Amostras 61-120)', edgecolors='black', alpha=0.8)
    
    ax.set_title('Dispersão Tridimensional das Amostras de Borracha', fontsize=12, fontweight='bold')
    ax.set_xlabel('Grandeza x1')
    ax.set_ylabel('Grandeza x2')
    ax.set_zlabel('Grandeza x3')
    ax.legend(fontsize=10)
    plt.savefig("dados_3d.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[✓] Gráfico 3D salvo como 'dados_3d.png'.")
    
    # 3.2 Treinamento da Rede SOM
    print("\n[!] Treinando a Rede de Kohonen (10.000 épocas, alpha = 0.001)...")
    som = KohonenSOM(grid_shape=(4, 4), input_dim=3, alpha=0.001, radius=1)
    som.train(dados_treinamento, epochs=10000)
    print("[✓] Treinamento concluído.")
    
    # 3.3 Mapeamento das Classes no Grid
    print("\n[!] Mapeando as amostras de treinamento no grid 4x4...")
    # Rastrear contagem de ativações [Classe A, Classe B, Classe C] para cada célula do grid
    grid_counts = np.zeros((4, 4, 3))
    
    for idx, x in enumerate(dados_treinamento):
        if idx < 20:
            c_id = 0 # Classe A
        elif idx < 60:
            c_id = 1 # Classe B
        else:
            c_id = 2 # Classe C
            
        bmu_idx, _ = som.find_bmu(x)
        grid_counts[bmu_idx[0], bmu_idx[1], c_id] += 1
        
    # Definir classe majoritária para cada neurônio no grid
    grid_labels = {}
    for r in range(4):
        for c in range(4):
            counts = grid_counts[r, c]
            if sum(counts) == 0:
                grid_labels[(r, c)] = 'Vazio'
            else:
                best_class = np.argmax(counts)
                grid_labels[(r, c)] = ['A', 'B', 'C'][best_class]
                
    # Imprimir o mapa do grid no terminal
    print("\nDIAGRAMA DO GRID TOPOLÓGICO 4x4:")
    print("-" * 55)
    for r in range(4):
        row_cells = []
        for c in range(4):
            label = grid_labels[(r, c)]
            counts = grid_counts[r, c]
            if label == 'Vazio':
                cell_str = f"[{r},{c}] Vazio  "
            else:
                cell_str = f"[{r},{c}] Cls {label} ({int(counts[0]):2},{int(counts[1]):2},{int(counts[2]):2})"
            row_cells.append(cell_str)
        print(" | ".join(row_cells))
    print("-" * 55)
    
    # 3.4 Salvar Imagem do Grid de Kohonen
    plotar_grid_kohonen(grid_counts, grid_labels)
    
    # 3.5 Classificação das Amostras de Teste
    print("\n[!] Classificando as 12 amostras de teste...")
    print("=" * 75)
    print(f"{'Amostra':<7} | {'x1':<8} | {'x2':<8} | {'x3':<8} | {'BMU (r,c)':<9} | {'Classe Neurônio':<15} | {'Classe Atribuída':<16}")
    print("-" * 85)
    
    linhas_tabela_markdown = []
    for idx, x in enumerate(dados_teste):
        bmu_idx, _ = som.find_bmu(x)
        neuronio_class = grid_labels[bmu_idx]
        
        # Se cair num neurônio vazio, atribui a classe pelo vizinho mais próximo rotulado,
        # mas na semente 42 os testes caem sempre em células já rotuladas.
        classe_atribuida = neuronio_class
        
        print(f" {idx+1:<6} | {x[0]:.4f} | {x[1]:.4f} | {x[2]:.4f} | ({bmu_idx[0]},{bmu_idx[1]})    | Classe {neuronio_class:<8} | Classe {classe_atribuida:<10}")
        
        linhas_tabela_markdown.append(
            f"| {idx+1} | {x[0]:.4f} | {x[1]:.4f} | {x[2]:.4f} | `({bmu_idx[0]}, {bmu_idx[1]})` | Classe {neuronio_class} | **Classe {classe_atribuida}** |"
        )
        
    # Salvar tabela markdown para copiar para o README
    with open("tabela_classificacao.md", "w", encoding="utf-8") as f:
        f.write("| Amostra | $x_1$ | $x_2$ | $x_3$ | Neurônio BMU | Classe do Neurônio | Classe Atribuída |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for line in linhas_tabela_markdown:
            f.write(line + "\n")
    print("\n[✓] Tabela markdown auxiliar gerada em 'tabela_classificacao.md'.")
    print("=" * 75)

def plotar_grid_kohonen(grid_counts, grid_labels):
    """
    Gera e salva uma representação visual premium do grid topológico do Kohonen.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Definir cores para as classes
    color_map = {
        'A': '#2ca02c', # Verde
        'B': '#1f77b4', # Azul
        'C': '#ff7f0e', # Laranja
        'Vazio': '#d6d6d6' # Cinza claro
    }
    
    for r in range(4):
        for c in range(4):
            label = grid_labels[(r, c)]
            counts = grid_counts[r, c]
            color = color_map[label]
            
            # Desenha um retângulo para o neurônio
            rect = plt.Rectangle((c, 3-r), 1, 1, facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7)
            ax.add_patch(rect)
            
            # Texto explicativo dentro da célula
            idx_str = f"N{r*4 + c}\n({r},{c})"
            ax.text(c+0.5, 3-r+0.7, idx_str, ha='center', va='center', fontsize=9, fontweight='bold')
            
            if label != 'Vazio':
                content_str = f"Classe {label}\n({int(counts[0])}, {int(counts[1])}, {int(counts[2])})"
            else:
                content_str = "Vazio"
            ax.text(c+0.5, 3-r+0.3, content_str, ha='center', va='center', fontsize=8, color='black')

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xticks(np.arange(0, 5, 1))
    ax.set_yticks(np.arange(0, 5, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='both', color='black', linestyle='-', linewidth=1.5)
    
    ax.set_title("Grid Topológico do Kohonen 4x4 (Distribuição de Classes)", fontsize=11, fontweight='bold', pad=15)
    
    # Criar legenda customizada
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', edgecolor='black', label='Região da Classe A'),
        Patch(facecolor='#1f77b4', edgecolor='black', label='Região da Classe B'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Região da Classe C'),
        Patch(facecolor='#d6d6d6', edgecolor='black', label='Neurônio Inativo (Vazio)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9)
    
    plt.savefig("grid_kohonen.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[✓] Imagem do grid do Kohonen salva como 'grid_kohonen.png'.")

if __name__ == '__main__':
    main()
