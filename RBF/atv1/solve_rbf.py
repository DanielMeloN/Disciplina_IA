import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONJUNTO DE DADOS ---

# Dados de Treinamento (Anexo) [x1, x2, d]
# d = 1: Presença de radiação
# d = -1: Ausência de radiação
dados_treinamento = np.array([
    [0.2563, 0.9503, -1.0], [0.2405, 0.9018, -1.0], [0.1157, 0.3676,  1.0], [0.5147, 0.0167,  1.0],
    [0.4127, 0.3275,  1.0], [0.2809, 0.5830,  1.0], [0.8263, 0.9301, -1.0], [0.9359, 0.8724, -1.0],
    [0.1096, 0.9165, -1.0], [0.5158, 0.8545, -1.0], [0.1334, 0.1362,  1.0], [0.6371, 0.1439,  1.0],
    [0.7052, 0.6277, -1.0], [0.8703, 0.8666, -1.0], [0.2612, 0.6109,  1.0], [0.0244, 0.5279,  1.0],
    [0.9588, 0.3672, -1.0], [0.9332, 0.5499, -1.0], [0.9623, 0.2961, -1.0], [0.7297, 0.5776, -1.0],
    [0.4560, 0.1871,  1.0], [0.1715, 0.7713,  1.0], [0.5571, 0.5485, -1.0], [0.3344, 0.0259,  1.0],
    [0.4803, 0.7635, -1.0], [0.9721, 0.4850, -1.0], [0.8318, 0.7844, -1.0], [0.1373, 0.0292,  1.0],
    [0.3660, 0.8581, -1.0], [0.3626, 0.7302, -1.0], [0.6474, 0.3324,  1.0], [0.3461, 0.2398,  1.0],
    [0.1353, 0.8120,  1.0], [0.3463, 0.1017,  1.0], [0.9086, 0.1947, -1.0], [0.5227, 0.2321,  1.0],
    [0.5153, 0.2041,  1.0], [0.1832, 0.0661,  1.0], [0.5015, 0.9812, -1.0], [0.5024, 0.5274, -1.0]
])

# Dados de Teste (Tabela de Validação) [x1, x2, d]
dados_teste = np.array([
    [0.8705, 0.9329, -1.0],
    [0.0388, 0.2703,  1.0],
    [0.8236, 0.4458, -1.0],
    [0.7075, 0.1502,  1.0],
    [0.9587, 0.8663, -1.0],
    [0.6115, 0.9365, -1.0],
    [0.3534, 0.3646,  1.0],
    [0.3268, 0.2766,  1.0],
    [0.6129, 0.4518, -1.0],
    [0.9948, 0.4962, -1.0]
])

X_train = dados_treinamento[:, :2]
D_train = dados_treinamento[:, 2]

X_test = dados_teste[:, :2]
D_test = dados_teste[:, 2]

# --- 1. TREINAMENTO DA CAMADA ESCONDIDA (K-MEANS) ---
# Filtrar apenas os padrões de treinamento com d = 1 (presença de radiação)
X_train_d1 = X_train[D_train == 1.0]

print(f"Total de padrões de treinamento com d=1: {len(X_train_d1)}")

# Implementação do algoritmo K-means (K=2)
# Inicialização com os dois primeiros padrões da lista com d=1
init_c1 = X_train_d1[0].copy()
init_c2 = X_train_d1[1].copy()

c1 = init_c1.copy()
c2 = init_c2.copy()

print(f"Centros Iniciais: c1 = {c1}, c2 = {c2}")

max_iter = 100
for it in range(max_iter):
    # Calcular as distâncias euclidianas quadráticas para cada centro
    dist_c1 = np.sum((X_train_d1 - c1)**2, axis=1)
    dist_c2 = np.sum((X_train_d1 - c2)**2, axis=1)
    
    # Atribuir cada ponto ao centro mais próximo (1 ou 2)
    labels = np.where(dist_c1 < dist_c2, 1, 2)
    
    # Separar os pontos de cada cluster
    cluster1 = X_train_d1[labels == 1]
    cluster2 = X_train_d1[labels == 2]
    
    # Recalcular os centros
    new_c1 = np.mean(cluster1, axis=0) if len(cluster1) > 0 else c1
    new_c2 = np.mean(cluster2, axis=0) if len(cluster2) > 0 else c2
    
    # Verificar convergência
    if np.allclose(c1, new_c1) and np.allclose(c2, new_c2):
        print(f"K-Means convergiu em {it+1} iterações.")
        break
    c1, c2 = new_c1, new_c2

# Obter pontos finais dos clusters
dist_c1 = np.sum((X_train_d1 - c1)**2, axis=1)
dist_c2 = np.sum((X_train_d1 - c2)**2, axis=1)
labels = np.where(dist_c1 < dist_c2, 1, 2)
cluster1 = X_train_d1[labels == 1]
cluster2 = X_train_d1[labels == 2]

# Calcular as variâncias respectivas (tanto populacional quanto amostral)
# Variância populacional (Denominador N)
var1_pop = np.mean(np.sum((cluster1 - c1)**2, axis=1))
var2_pop = np.mean(np.sum((cluster2 - c2)**2, axis=1))

# Variância amostral (Denominador N-1)
var1_amostra = np.sum(np.sum((cluster1 - c1)**2, axis=1)) / (len(cluster1) - 1)
var2_amostra = np.sum(np.sum((cluster2 - c2)**2, axis=1)) / (len(cluster2) - 1)

print("\n--- RESULTADOS K-MEANS ---")
print(f"Cluster 1: Centro = {c1}, Tamanho = {len(cluster1)}")
print(f"  Variância Populacional (N):   {var1_pop:.6f}")
print(f"  Variância Amostral (N-1):     {var1_amostra:.6f}")
print(f"Cluster 2: Centro = {c2}, Tamanho = {len(cluster2)}")
print(f"  Variância Populacional (N):   {var2_pop:.6f}")
print(f"  Variância Amostral (N-1):     {var2_amostra:.6f}")

# --- 2. GERAR ATIVAÇÕES DAS FUNÇÕES DE BASE RADIAL ---
# Usaremos a Variância Populacional (N) como padrão para o treinamento principal,
# e a definição gaussiana padrão com o fator 2 no denominador da exponencial:
# g(x) = exp( - ||x - c||^2 / (2 * var) )
var1 = var1_pop
var2 = var2_pop

def gaussian_rbf(x, center, variance):
    dist_sq = np.sum((x - center)**2)
    return np.exp(-dist_sq / (2 * variance))

# --- 3. TREINAMENTO DA CAMADA DE SAÍDA (REGRA DELTA) ---
# Adiciona o bias (x0 = -1) na entrada da camada de saída
# A entrada do neurônio de saída será [x0, g1(x), g2(x)]
bias_input = -1.0
H_train = np.array([[bias_input, gaussian_rbf(x, c1, var1), gaussian_rbf(x, c2, var2)] for x in X_train])

# Parâmetros de Treinamento
ETA = 0.01
EPSILON = 1e-7
np.random.seed(42) # Semente fixa para fins acadêmicos e reprodutibilidade
W = np.random.uniform(0, 1, 3) # Pesos iniciais aleatórios [W21,0, W21,1, W21,2]
W_initial = W.copy()

print("\n--- INICIANDO TREINAMENTO DA CAMADA DE SAÍDA ---")
print(f"Pesos iniciais: W21,0 = {W[0]:.6f}, W21,1 = {W[1]:.6f}, W21,2 = {W[2]:.6f}")

eqm_history = []
epoch = 0
max_epochs = 50000

while True:
    # Ajuste dos pesos padrão online (amostra por amostra, igual aos scripts anteriores)
    for i in range(len(X_train)):
        h = H_train[i]
        d = D_train[i]
        y = np.dot(W, h) # Saída linear da rede
        error = d - y
        
        # Atualização dos pesos (regra delta)
        W += ETA * error * h
        
    # Calcular o EQM desta época
    y_pred_train = np.dot(H_train, W)
    eqm_atual = np.mean((D_train - y_pred_train)**2)
    eqm_history.append(eqm_atual)
    epoch += 1
    
    # Condição de Parada
    if epoch > 1:
        if abs(eqm_history[-1] - eqm_history[-2]) < EPSILON:
            print(f"Convergência atingida na época {epoch}.")
            break
            
    if epoch >= max_epochs:
        print(f"Treinamento interrompido pelo número máximo de épocas ({max_epochs}).")
        break

print("\n--- PESOS FINAIS CONVERGIDOS ---")
print(f"W21,0 (Bias): {W[0]:.6f}")
print(f"W21,1 (G1):   {W[1]:.6f}")
print(f"W21,2 (G2):   {W[2]:.6f}")
print(f"EQM Final:    {eqm_history[-1]:.9f}")

# --- 4. VALIDAÇÃO COM O CONJUNTOS DE TESTE ---
# Gerar ativações RBF para o conjunto de teste
H_test = np.array([[bias_input, gaussian_rbf(x, c1, var1), gaussian_rbf(x, c2, var2)] for x in X_test])

# Obter saídas reais da rede
y_test = np.dot(H_test, W)

# Pós-processamento usando a função sinal: y_pos = 1 se y >= 0 else -1
def sinal(val):
    return 1.0 if val >= 0.0 else -1.0

y_test_processed = np.array([sinal(y) for y in y_test])

# Calcular Taxa de Acerto (%)
acertos = np.sum(y_test_processed == D_test)
taxa_acerto = (acertos / len(D_test)) * 100

print("\n--- TABELA DE VALIDAÇÃO (TESTE) ---")
print(f"{'Amostra':<8} | {'x1':<8} | {'x2':<8} | {'d':<5} | {'y (real)':<10} | {'y_pós':<6} | {'Status':<10}")
print("-" * 65)
for i in range(len(D_test)):
    status = "Correto" if y_test_processed[i] == D_test[i] else "Incorreto"
    print(f"{i+1:<8} | {X_test[i,0]:.4f} | {X_test[i,1]:.4f} | {int(D_test[i]):<5} | {y_test[i]:.6f} | {int(y_test_processed[i]):<6} | {status}")
print(f"Taxa de Acerto (%): {taxa_acerto:.2f}%")

# Calcular Acurácia de Treinamento também para o relatório
y_train_raw = np.dot(H_train, W)
y_train_processed = np.array([sinal(y) for y in y_train_raw])
acertos_train = np.sum(y_train_processed == D_train)
taxa_acerto_train = (acertos_train / len(D_train)) * 100
print(f"Taxa de Acerto no Treinamento (%): {taxa_acerto_train:.2f}%")

# --- 5. GERAR GRÁFICOS ---
# Plot 1: Evolução do EQM
plt.figure(figsize=(8, 5))
plt.plot(eqm_history, color='darkblue', linewidth=2, label='Evolução do EQM')
plt.title('Evolução do Erro Quadrático Médio (EQM) no Treinamento da RBF')
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio')
plt.grid(True, linestyle='--', alpha=0.6)
plt.yscale('log') # Escala logarítmica ajuda a ver a precisão de 1e-7
plt.legend()
plt.tight_layout()
plt.savefig('rbf_eqm_epochs.png', dpi=300)
plt.close()
print("\n[!] Gráfico de convergência salvo como 'rbf_eqm_epochs.png'.")

# Plot 2: Distribuição dos Clusters e Amostras de Teste
plt.figure(figsize=(9, 7))

# Plot dos pontos de treino
plt.scatter(X_train[D_train == 1.0][:,0], X_train[D_train == 1.0][:,1], 
            color='limegreen', marker='o', edgecolors='k', s=60, label='Treino (d=1, Radiação)')
plt.scatter(X_train[D_train == -1.0][:,0], X_train[D_train == -1.0][:,1], 
            color='lightcoral', marker='x', s=60, linewidths=1.5, label='Treino (d=-1, Sem Radiação)')

# Plot dos pontos de teste
plt.scatter(X_test[D_test == 1.0][:,0], X_test[D_test == 1.0][:,1], 
            color='darkgreen', marker='^', edgecolors='k', s=100, label='Teste (d=1, Radiação)')
plt.scatter(X_test[D_test == -1.0][:,0], X_test[D_test == -1.0][:,1], 
            color='darkred', marker='v', edgecolors='k', s=100, label='Teste (d=-1, Sem Radiação)')

# Plot dos centros dos clusters encontrados pelo K-Means
plt.scatter(c1[0], c1[1], color='cyan', marker='D', edgecolors='k', s=150, linewidths=2, label='Centro 1 (c1)')
plt.scatter(c2[0], c2[1], color='gold', marker='D', edgecolors='k', s=150, linewidths=2, label='Centro 2 (c2)')

# Desenhar círculos de raio proporcional ao desvio padrão (variância)
# Usando a variância populacional para desenhar
r1 = np.sqrt(var1_pop)
r2 = np.sqrt(var2_pop)

circle1 = plt.Circle((c1[0], c1[1]), r1, color='cyan', fill=False, linestyle='--', linewidth=1.5, label='Região C1 (Desvio Padrão)')
circle2 = plt.Circle((c2[0], c2[1]), r2, color='gold', fill=False, linestyle='--', linewidth=1.5, label='Região C2 (Desvio Padrão)')
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)

plt.title('Distribuição Espacial das Amostras, Centros e Desvios Padrão')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('clustering_distribution.png', dpi=300)
plt.close()
print("[!] Gráfico de distribuição espacial salvo como 'clustering_distribution.png'.")
