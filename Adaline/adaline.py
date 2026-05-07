import random
import matplotlib.pyplot as plt
import numpy as np

# Parâmetros
ETA = 0.0025
EPSILON = 1e-6

# Conjunto de Treinamento [x1, x2, x3, x4, d]
# O bias (x0 = -1) será adicionado no algoritmo
dados_treinamento = [
    [0.4329, -1.3719, 0.7022, -0.8535, 1.0], [0.3024, 0.2286, 0.8630, 2.7909, -1.0],
    [0.1349, -0.6445, 1.0530, 0.5687, -1.0], [0.3374, -1.7163, 0.3670, -0.6283, -1.0],
    [1.1434, -0.0485, 0.6637, 1.2606, 1.0], [1.3749, -0.5071, 0.4464, 1.3009, 1.0],
    [0.7221, -0.7587, 0.7681, -0.5592, 1.0], [0.4403, -0.8072, 0.5154, -0.3129, 1.0],
    [-0.5231, 0.3548, 0.2538, 1.5776, -1.0], [0.3255, -2.0000, 0.7112, -1.1209, 1.0],
    [0.5824, 1.3915, -0.2291, 4.1735, -1.0], [0.1340, 0.6081, 0.4450, 3.2230, -1.0],
    [0.1480, -0.2988, 0.4778, 0.8649, 1.0], [0.7359, 0.1869, -0.0872, 2.3584, 1.0],
    [0.7115, -1.1469, 0.3394, 0.9573, -1.0], [0.8251, -1.2840, 0.8452, 1.2382, -1.0],
    [0.1569, 0.3712, 0.8825, 1.7633, 1.0], [0.0033, 0.6835, 0.5389, 2.8249, -1.0],
    [0.4243, 0.8313, 0.2634, 3.5855, -1.0], [1.0490, 0.1326, 0.9138, 1.9792, 1.0],
    [1.4276, 0.5331, -0.0145, 3.7286, 1.0], [0.5971, 1.4865, 0.2904, 4.6069, -1.0],
    [0.8475, 2.1479, 0.3179, 5.8235, -1.0], [1.3967, -0.4171, 0.6443, 1.3927, 1.0],
    [0.0044, 1.5378, 0.6099, 4.7755, -1.0], [0.2201, -0.5668, 0.0515, 0.7829, 1.0],
    [0.6300, -1.2480, 0.8591, 0.8093, -1.0], [-0.2479, 0.8960, 0.0547, 1.7381, 1.0],
    [-0.3088, -0.0929, 0.8659, 1.5483, -1.0], [-0.5180, 1.4974, 0.5453, 2.3993, 1.0],
    [0.6833, 0.8266, 0.0829, 2.8864, 1.0], [0.4353, -1.4066, 0.4207, -0.4879, 1.0],
    [-0.1069, -3.2329, 0.1856, -2.4572, -1.0], [0.4662, 0.6261, 0.7304, 3.4370, -1.0],
    [0.8298, -1.4089, 0.3119, 1.3235, -1.0]
]

# Amostras de Teste [x1, x2, x3, x4]
dados_teste = [
    [0.9694, 0.6909, 0.4334, 3.4965], [0.5427, 1.3832, 0.6390, 4.0352],
    [0.6081, -0.9196, 0.5925, 0.1016], [-0.1618, 0.4694, 0.2030, 3.0117],
    [0.1870, -0.2578, 0.6124, 1.7749], [0.4891, -0.5276, 0.4378, 0.6439],
    [0.3777, 2.0149, 0.7423, 3.3932], [1.1498, -0.4067, 0.2469, 1.5866],
    [0.9325, 1.0950, 1.0359, 3.3591], [0.5060, 1.3317, 0.9222, 3.7174],
    [0.0497, -2.0656, 0.6124, -0.6585], [0.4004, 3.5369, 0.9766, 5.3532],
    [-0.1874, 1.3343, 0.5374, 3.2189], [0.5060, 1.3317, 0.9222, 3.7174],
    [1.6375, -0.7911, 0.7537, 0.5515]
]

def calcular_saida(w, x):
    # u = sum(wi * xi)
    return sum(wi * xi for wi, xi in zip(w, x))

def sinal(u):
    return 1 if u >= 0 else -1

def calcular_eqm(w, dados):
    eqm = 0
    for linha in dados:
        x = [-1.0] + linha[:4] # bias
        d = linha[4]
        u = calcular_saida(w, x)
        eqm += (d - u) ** 2
    return eqm / len(dados)

def treinar_adaline(id_treino, plotar_grafico=False):
    # Inicializa vetor de pesos com valores aleatórios entre 0 e 1
    w = [random.uniform(0, 1) for _ in range(5)]
    w_inicial = list(w)
    
    epocas = 0
    eqm_atual = 0
    eqm_anterior = 0
    historico_eqm = []
    historico_w = [list(w)] # Salva o histórico de pesos para a trajetória 3D
    MAX_EPOCAS = 2000 # Trava de segurança para evitar loops infinitos
    
    if plotar_grafico:
        plt.ion() # Modo interativo para animação
        fig, ax = plt.subplots()
        ax.set_title(f"T{id_treino} - Evolução do EQM")
        ax.set_xlabel("Épocas")
        ax.set_ylabel("Erro Quadrático Médio (EQM)")
        linha_grafico, = ax.plot([], [], 'b-')
    
    while True:
        eqm_anterior = eqm_atual
        
        # Ajuste dos pesos (Regra Delta)
        for linha in dados_treinamento:
            x = [-1.0] + linha[:4] # Adiciona entrada x0 = -1
            d = linha[4]
            u = calcular_saida(w, x)
            erro = d - u
            
            for i in range(len(w)):
                w[i] = w[i] + ETA * erro * x[i]
                
        # Cálculo do EQM da época atual
        eqm_atual = calcular_eqm(w, dados_treinamento)
        historico_eqm.append(eqm_atual)
        epocas += 1
        
        # Atualização visual ao vivo
        if plotar_grafico and epocas % 5 == 0:
            linha_grafico.set_xdata(range(len(historico_eqm)))
            linha_grafico.set_ydata(historico_eqm)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001) # Pequena pausa para a interface renderizar
            
        historico_w.append(list(w))
            
        # Condição de parada ou limite de segurança
        if abs(eqm_atual - eqm_anterior) < EPSILON or epocas >= MAX_EPOCAS:
            break
            
    if plotar_grafico:
        plt.ioff()
        plt.close(fig) # Fecha a janela animada temporária
        return w_inicial, w, epocas, historico_eqm, eqm_atual, historico_w
        
    return w_inicial, w, epocas, historico_eqm, eqm_atual, historico_w

def formatar_pesos(pesos):
    return "[" + ", ".join(f"{p:7.4f}" for p in pesos) + "]"

def plotar_superficie_3d(w_final, dados, historico_w, label="T1"):
    print(f"\n[!] Gerando Gráfico 3D da Superfície de Erro para {label}...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Variar w1 e w2 (índices 1 e 2 no array de pesos) mantendo o resto fixo
    w1_vals = np.linspace(w_final[1] - 1.5, w_final[1] + 1.5, 30)
    w2_vals = np.linspace(w_final[2] - 1.5, w_final[2] + 1.5, 30)
    
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    EQM_surf = np.zeros(W1.shape)
    
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w_temp = list(w_final)
            w_temp[1] = W1[i, j]
            w_temp[2] = W2[i, j]
            EQM_surf[i, j] = calcular_eqm(w_temp, dados)
            
    surf = ax.plot_surface(W1, W2, EQM_surf, cmap='viridis', edgecolor='none', alpha=0.7)
    
    # Plotar a trajetória do Gradiente Descendente (Como os pesos w1 e w2 mudaram ao longo das épocas)
    w1_hist = [pesos[1] for pesos in historico_w]
    w2_hist = [pesos[2] for pesos in historico_w]
    eqm_hist = []
    
    for pesos_hist in historico_w:
        w_temp = list(w_final)
        w_temp[1] = pesos_hist[1]
        w_temp[2] = pesos_hist[2]
        # Levantar a linha um pouquinho (+0.05) para não "afundar" dentro da superfície 3D (z-fighting)
        eqm_hist.append(calcular_eqm(w_temp, dados) + 0.05)
        
    ax.plot(w1_hist, w2_hist, eqm_hist, color='black', marker='o', markersize=5, linestyle='solid', linewidth=3, label='Trajetória do Treinamento', zorder=10)
    
    # Ponto do Inicial (Pesos Iniciais)
    ax.scatter([w1_hist[0]], [w2_hist[0]], [eqm_hist[0]], color='magenta', s=80, label='Pesos Iniciais (Início)', zorder=5)

    # Ponto do Mínimo Global (Pesos Finais)
    eqm_min = calcular_eqm(w_final, dados)
    ax.scatter([w_final[1]], [w_final[2]], [eqm_min], color='red', s=100, label=f'Mínimo Global (Fim)', zorder=6)
    
    ax.set_title(f'Superfície 3D do EQM - Variação de w1 e w2 ({label})')
    ax.set_xlabel('Peso w1')
    ax.set_ylabel('Peso w2')
    ax.set_zlabel('Erro Quadrático Médio (EQM)')
    plt.legend()
    plt.savefig(f'grafico_superficie_3d_{label.lower()}.png')
    print(f"[!] Gráfico 3D salvo como 'grafico_superficie_3d_{label.lower()}.png'.")

def main():
    resultados_treinos = []
    historicos_plot = []
    
    print("-" * 110)
    print("TREINANDO REDE ADALINE (Isso pode demorar alguns segundos e abrirá gráficos na tela)...")
    print("-" * 110)
    
    for i in range(1, 6):
        plotar = True if i in [1, 2] else False
        w_ini, w_fin, epocas, hist_eqm, eqm_final, hist_w = treinar_adaline(i, plotar_grafico=plotar)
        resultados_treinos.append({
            'id': i,
            'w_ini': w_ini,
            'w_fin': w_fin,
            'epocas': epocas,
            'eqm_final': eqm_final,
            'hist_w': hist_w
        })
        if plotar:
            historicos_plot.append((f"T{i}", hist_eqm))
            
    # Tabela 1: Resultados dos Treinamentos
    print("\nRESULTADOS DOS TREINAMENTOS:")
    print(f"{'Treino':<8} | {'Pesos Iniciais':<45} | {'Pesos Finais':<45} | {'Épocas':<8} | {'EQM Final':<10}")
    print("-" * 125)
    for res in resultados_treinos:
        print(f"T{res['id']:<7} | {formatar_pesos(res['w_ini']):<45} | {formatar_pesos(res['w_fin']):<45} | {res['epocas']:<8} | {res['eqm_final']:.6f}")
        
    # Plot final unificado e permanente para T1 e T2
    plt.figure(figsize=(10, 6))
    treinos_plot = [r for r in resultados_treinos if r['id'] in [1, 2]]
    for res, (label, hist) in zip(treinos_plot, historicos_plot):
        plt.plot(hist, label=f"{label} (Épocas Necessárias: {res['epocas']})")
    plt.title("Evolução do EQM (Treinamentos T1 e T2)")
    plt.xlabel("Épocas")
    plt.ylabel("Erro Quadrático Médio (EQM)")
    plt.legend()
    plt.grid(True)
    plt.savefig('grafico_eqm_t1_t2.png')
    print("\n[!] O gráfico final de EQM para T1 e T2 foi salvo na pasta como 'grafico_eqm_t1_t2.png'.")
        
    # Gerar Gráfico 3D para T1 com Trajetória
    plotar_superficie_3d(resultados_treinos[0]['w_fin'], dados_treinamento, resultados_treinos[0]['hist_w'], label="T1")
        
    # Tabela 2: Classificação das Amostras Teste
    print("\nCLASSIFICAÇÃO DAS 15 AMOSTRAS DE TESTE:")
    print(f"{'Amostra':<8} | {'y (T1)':<8} | {'y (T2)':<8} | {'y (T3)':<8} | {'y (T4)':<8} | {'y (T5)':<8}")
    print("-" * 65)
    for i, amostra in enumerate(dados_teste):
        x_teste = [-1.0] + amostra
        saidas = []
        for res in resultados_treinos:
            u = calcular_saida(res['w_fin'], x_teste)
            saidas.append(sinal(u))
        
        y_strs = [f"{s:2} ({'B' if s==1 else 'A'})" for s in saidas]
        print(f"{i+1:<8} | {y_strs[0]:<8} | {y_strs[1]:<8} | {y_strs[2]:<8} | {y_strs[3]:<8} | {y_strs[4]:<8}")

    # Questão teórica
    print("\n" + "="*110)
    print("ANÁLISE TEÓRICA")
    print("Questão: Embora o número de épocas seja diferente, explique por que os valores dos pesos")
    print("continuam praticamente inalterados.")
    print("-" * 110)
    print("Resposta:")
    print("A rede ADALINE ajusta os pesos com o objetivo de minimizar a função de custo (o Erro Quadrático")
    print("Médio - EQM). Para um único neurônio de ativação linear, essa função de custo desenha um")
    print("hiperparaboloide, que é uma superfície estritamente convexa e possui apenas um MÍNIMO GLOBAL.")
    print("Não importa quais sejam os pesos iniciais gerados aleatoriamente; o algoritmo de gradiente")
    print("descendente sempre conduzirá a solução para esse mesmo ponto de erro mínimo.")
    print("A diferença no número de épocas ocorre porque pesos iniciais sorteados muito longe do mínimo")
    print("necessitam de mais iterações para chegar lá do que pesos sorteados que por sorte nasceram")
    print("mais próximos ao vale do erro. Mas o destino final (os pesos finais) sempre será o mesmo.")
    print("="*110 + "\n")

    # Mostra o gráfico final na tela
    plt.show()

if __name__ == "__main__":
    main()
