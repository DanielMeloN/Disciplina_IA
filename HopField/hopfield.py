import os
import numpy as np
import matplotlib.pyplot as plt

# Definir semente aleatória para reprodutibilidade
np.random.seed(42)

# =====================================================================
# 1. Padrões de Referência (Imagens 9x5 = 45 Neurônios)
# =====================================================================
# Representação: -1 para pixel branco, +1 para pixel escuro (preto)

PADRAO_1 = np.array([
    -1, -1,  1,  1, -1,
    -1,  1,  1,  1, -1,
    -1, -1,  1,  1, -1,
    -1, -1,  1,  1, -1,
    -1, -1,  1,  1, -1,
    -1, -1,  1,  1, -1,
    -1, -1,  1,  1, -1,
    -1, -1,  1,  1, -1,
    -1, -1,  1,  1, -1
])

PADRAO_2 = np.array([
     1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,
    -1, -1, -1,  1,  1,
    -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,
     1,  1, -1, -1, -1,
     1,  1, -1, -1, -1,
     1,  1,  1,  1,  1,
     1,  1,  1,  1,  1
])

PADRAO_3 = np.array([
     1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,
    -1, -1, -1,  1,  1,
    -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,
    -1, -1, -1,  1,  1,
    -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,
     1,  1,  1,  1,  1
])

PADRAO_4 = np.array([
     1,  1, -1,  1,  1,
     1,  1, -1,  1,  1,
     1,  1, -1,  1,  1,
     1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,
    -1, -1, -1,  1,  1,
    -1, -1, -1,  1,  1,
    -1, -1, -1,  1,  1,
    -1, -1, -1,  1,  1
])

PADROES = [PADRAO_1, PADRAO_2, PADRAO_3, PADRAO_4]
NOMES_PADROES = ["Dígito 1", "Dígito 2", "Dígito 3", "Dígito 4"]

# =====================================================================
# 2. Implementação da Rede de Hopfield
# =====================================================================

class HopfieldNetwork:
    def __init__(self, size=45):
        self.size = size
        self.w = np.zeros((size, size))
        
    def train(self, patterns):
        """
        Treina a rede usando a regra do produto externo (Hebbiana).
        Zera a diagonal para evitar auto-conexões.
        """
        self.w = np.zeros((self.size, self.size))
        for p in patterns:
            self.w += np.outer(p, p)
        # Normalização clássica pelo número de neurônios N
        self.w = self.w / self.size
        # w_ii = 0
        np.fill_diagonal(self.w, 0)
        
    def energy(self, state):
        """
        Calcula a energia da rede para um determinado estado.
        E = -0.5 * state^T * W * state
        """
        return -0.5 * np.dot(state, np.dot(self.w, state))
        
    def update_asynchronous(self, initial_state, max_epochs=20, beta=100.0):
        """
        Atualiza o estado de forma assíncrona.
        Retorna o estado final convergido, o histórico de estados e o histórico de energia.
        """
        state = np.array(initial_state, dtype=float).copy()
        state_history = [state.copy()]
        energy_history = [self.energy(state)]
        
        for epoch in range(max_epochs):
            changed = False
            # Permutação aleatória da ordem de atualização dos neurônios
            indices = np.random.permutation(self.size)
            
            for i in indices:
                # Entrada induzida (potencial de ativação)
                u_i = np.dot(self.w[i], state)
                
                # Função de ativação Tangente Hiperbólica com beta grande
                activated_val = np.tanh(beta * u_i)
                
                # Mapeamento para bipolar
                new_val = np.sign(activated_val)
                if new_val == 0:
                    new_val = state[i] # Mantém estado anterior caso u_i = 0
                
                if new_val != state[i]:
                    state[i] = new_val
                    changed = True
            
            state_history.append(state.copy())
            energy_history.append(self.energy(state))
            
            # Se nenhum neurônio mudou de estado nesta época, a rede convergiu
            if not changed:
                break
                
        return state, state_history, energy_history

# =====================================================================
# 3. Funções Auxiliares de Ruído e Visualização
# =====================================================================

def corrupt_pattern(pattern, noise_level=0.20):
    """
    Introduz ruído invertendo exatamente a proporção dada de pixels.
    Para 20% em 45 pixels, são invertidos exatamente 9 pixels.
    """
    noisy = pattern.copy()
    num_to_flip = int(round(noise_level * len(pattern)))
    flip_indices = np.random.choice(len(pattern), num_to_flip, replace=False)
    for idx in flip_indices:
        noisy[idx] = -noisy[idx]
    return noisy

def to_ascii_lines(pattern):
    """
    Converte o padrão flat 45 em 9 linhas de caracteres ASCII (# para preto, . para branco).
    """
    lines = []
    for r in range(9):
        row = pattern[r*5 : (r+1)*5]
        row_str = "".join("#" if val == 1 else "." for val in row)
        lines.append(row_str)
    return lines

def print_ascii_comparison(original, noisy, recovered):
    """
    Imprime de forma legível no terminal o original, o ruidoso e o recuperado lado a lado.
    """
    orig_lines = to_ascii_lines(original)
    noisy_lines = to_ascii_lines(noisy)
    rec_lines = to_ascii_lines(recovered)
    
    print(f"{'Original':<7}       {'Ruidoso (20%)':<14}       {'Recuperado':<10}")
    print("-" * 45)
    for i in range(9):
        print(f" {orig_lines[i]:<5}         {noisy_lines[i]:<5}           {rec_lines[i]:<5}")
    print()

def salvar_imagem_simulacao(original, noisy, recovered, pattern_id, sit_id, save_dir="."):
    """
    Gera e salva uma visualização gráfica lado a lado da simulação.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    
    # Custom color map para estilo premium (azul aço e branco)
    cmap_custom = plt.cm.Blues
    
    # Imagem Original
    axes[0].imshow(original.reshape(9, 5), cmap=cmap_custom, vmin=-1, vmax=1)
    axes[0].set_title("Original (Limpa)", fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Imagem Ruidosa
    axes[1].imshow(noisy.reshape(9, 5), cmap=cmap_custom, vmin=-1, vmax=1)
    axes[1].set_title("Transmitida (Ruído 20%)", fontsize=10, fontweight='bold')
    axes[1].axis('off')
    
    # Imagem Recuperada
    axes[2].imshow(recovered.reshape(9, 5), cmap=cmap_custom, vmin=-1, vmax=1)
    
    # Verifica se recuperou perfeitamente
    sucesso = np.array_equal(original, recovered)
    status_str = "Recuperada (Sucesso)" if sucesso else "Recuperada (Falha)"
    title_color = "darkgreen" if sucesso else "crimson"
    
    axes[2].set_title(status_str, fontsize=10, fontweight='bold', color=title_color)
    axes[2].axis('off')
    
    plt.suptitle(f"Simulação de Transmissão: Padrão {pattern_id} (Dígito {pattern_id}) - Situação {sit_id}", 
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    filename = f"simulacao_p{pattern_id}_s{sit_id}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

# =====================================================================
# 4. Execução Principal (Treinamento e Simulações)
# =====================================================================

def main():
    print("=" * 70)
    print("      RESOLUÇÃO DO EXERCÍCIO: REDE DE HOPFIELD (45 NEURÔNIOS)")
    print("=" * 70)
    print("[!] Inicializando rede e carregando padrões...")
    
    net = HopfieldNetwork(size=45)
    net.train(PADROES)
    print("[✓] Rede treinada com a Regra de Hebb (Matriz de pesos W obtida).")
    print(f"    Auto-conexões zeradas (diagonal principal de W = 0).")
    print("-" * 70)
    
    # Executar as 12 situações de transmissão (3 para cada um dos 4 padrões)
    resultados = []
    
    print("\n[!] Simulando as 12 situações de transmissão (Ruído = 20%):")
    print("=" * 70)
    
    for p_idx, pattern in enumerate(PADROES):
        p_num = p_idx + 1
        print(f"\n>>> PADRÃO {p_num} ({NOMES_PADROES[p_idx]}) <<<")
        print("=" * 70)
        
        for sit in range(1, 4):
            # Adicionar 20% de ruído (9 bits invertidos)
            noisy = corrupt_pattern(pattern, noise_level=0.20)
            
            # Recuperar usando a rede de Hopfield
            recovered, history, energy_history = net.update_asynchronous(noisy, max_epochs=20, beta=100.0)
            
            # Verificar se foi recuperada com sucesso
            sucesso = np.array_equal(pattern, recovered)
            num_epocas = len(history) - 1
            energia_ini = energy_history[0]
            energia_fim = energy_history[-1]
            
            # Armazenar resultados
            resultados.append({
                'padrao': p_num,
                'situacao': sit,
                'original': pattern,
                'ruidoso': noisy,
                'recuperado': recovered,
                'sucesso': sucesso,
                'epocas': num_epocas,
                'energia_inicial': energia_ini,
                'energia_final': energia_fim,
                'historico_energia': energy_history
            })
            
            # Imprimir ASCII no terminal
            print(f"Situação {sit}: Convergiu em {num_epocas} épocas. "
                  f"Energia: {energia_ini:.4f} -> {energia_fim:.4f} "
                  f"[{'SUCESSO' if sucesso else 'FALHA'}]")
            print_ascii_comparison(pattern, noisy, recovered)
            
            # Salvar imagem
            salvar_imagem_simulacao(pattern, noisy, recovered, p_num, sit)
            
    # =====================================================================
    # 5. Análise de Robustez ao Ruído (Aumento Excessivo de Ruído)
    # =====================================================================
    print("\n[!] Analisando impacto do aumento excessivo de ruído...")
    print("    Avaliando níveis de ruído de 0% a 100% com 100 simulações por nível.")
    
    niveis_ruido = np.arange(0.0, 1.05, 0.05)
    taxa_sucesso = []
    
    for noise in niveis_ruido:
        sucessos_nivel = 0
        total_trials = 100
        
        for _ in range(total_trials):
            # Seleciona aleatoriamente um dos 4 padrões
            p_sel = PADROES[np.random.choice(4)]
            # Corrompe
            noisy = corrupt_pattern(p_sel, noise_level=noise)
            # Atualiza
            recovered, _, _ = net.update_asynchronous(noisy, max_epochs=15, beta=100.0)
            # Verifica sucesso
            if np.array_equal(p_sel, recovered):
                sucessos_nivel += 1
                
        taxa = (sucessos_nivel / total_trials) * 100
        taxa_sucesso.append(taxa)
        
    # Plotar gráfico de robustez
    plt.figure(figsize=(8, 5))
    plt.plot(niveis_ruido * 100, taxa_sucesso, marker='o', linewidth=2.5, color='#1f77b4', label='Taxa de Recuperação')
    plt.axvline(20, color='r', linestyle='--', label='Ruído do Exercício (20%)')
    
    # Destacar limite teórico de capacidade aproximado de Hopfield (0.138 * N)
    # Para N = 45, capacidade ~ 0.138 * 45 = 6.2 padrões.
    # Como armazenamos 4 padrões (cerca de 0.09 * N), a rede está perto da capacidade.
    # A bacia de atração típica encolhe rapidamente acima de 15-25% de ruído.
    
    plt.title("Robustez da Rede de Hopfield: Taxa de Sucesso vs. Nível de Ruído", fontsize=12, fontweight='bold')
    plt.xlabel("Nível de Ruído (% de pixels invertidos)", fontsize=10)
    plt.ylabel("Recuperação Perfeita (%)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.xlim(0, 100)
    plt.ylim(-5, 105)
    
    plt.savefig("analise_ruido.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[✓] Gráfico de robustez salvo como 'analise_ruido.png'.")
    
    # Gerar a tabela markdown para inclusão no README.md
    gerar_tabela_markdown_resultados(resultados)
    
def gerar_tabela_markdown_resultados(resultados):
    """
    Função auxiliar que gera trechos de código em Markdown para as tabelas de simulação,
    facilitando a escrita do relatório.
    """
    filepath = "tabela_simulacoes.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Resultados das Simulações (ASCII)\n\n")
        for res in resultados:
            orig_lines = to_ascii_lines(res['original'])
            noisy_lines = to_ascii_lines(res['ruidoso'])
            rec_lines = to_ascii_lines(res['recuperado'])
            
            f.write(f"### Padrão {res['padrao']} (Dígito {res['padrao']}) - Situação {res['situacao']}\n")
            f.write(f"- **Convergência**: {'Sucesso' if res['sucesso'] else 'Falha'}\n")
            f.write(f"- **Épocas**: {res['epocas']}\n")
            f.write(f"- **Variação de Energia**: {res['energia_inicial']:.4f} -> {res['energia_final']:.4f}\n\n")
            f.write("| Original | Transmitida (Ruidosa) | Recuperada |\n")
            f.write("| :---: | :---: | :---: |\n")
            for i in range(9):
                # Substituir # e . por blocos de markdown mais legíveis
                o_cell = orig_lines[i].replace("#", "⬛").replace(".", "⬜")
                n_cell = noisy_lines[i].replace("#", "⬛").replace(".", "⬜")
                r_cell = rec_lines[i].replace("#", "⬛").replace(".", "⬜")
                f.write(f"| `{o_cell}` | `{n_cell}` | `{r_cell}` |\n")
            f.write("\n---\n\n")
    print(f"[✓] Tabela markdown auxiliar gerada em '{filepath}'.")

if __name__ == '__main__':
    main()
