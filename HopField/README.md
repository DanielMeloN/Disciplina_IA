# Relatório Acadêmico: Memória Associativa com Rede de Hopfield

**Instituição**: Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG)  
**Campus**: Campus VIII – Varginha  
**Curso**: Bacharelado em Sistemas de Informação  
**Disciplina**: Laboratório de Inteligência Artificial  
**Professor**: Lázaro Eduardo da Silva  
**Data**: 28/05/2026  

---

## 1. Introdução Teórica

A **Rede de Hopfield** é um modelo clássico de rede neural recorrente de camada única com conexões simétricas e realimentação. Proposta por John Hopfield em 1982, ela funciona como uma **memória associativa endereçável por conteúdo**. Isso significa que a rede é capaz de recuperar uma informação completa e limpa memorizada previamente a partir de uma versão corrompida, ruidosa ou parcial (incompleta) dessa mesma informação.

### 1.1 Regra de Aprendizado (Regra de Hebb / Produto Externo)

Dada uma coleção de $P$ padrões binários bipolares $\{x^{\mu}\}_{\mu=1}^P$ que desejamos armazenar, onde cada padrão $x^{\mu} \in \{-1, 1\}^N$, a matriz de pesos sinápticos $W \in \mathbb{R}^{N \times N}$ é construída utilizando o aprendizado hebbiano clássico (ou regra do produto externo):

$$w_{ij} = \frac{1}{N} \sum_{\mu=1}^P x_i^{\mu} x_j^{\mu} \quad \text{para } i \neq j$$

E para evitar a auto-realimentação direta (que faria com que o neurônio simplesmente mantivesse seu estado inicial obstinadamente e impediria a correção de ruído), as conexões diagonais são zeradas:

$$w_{ii} = 0 \quad \forall i \in \{1, \dots, N\}$$

### 1.2 Dinâmica de Atualização de Estados e Função de Ativação

A dinâmica da rede é descrita pelas atualizações de estado dos neurônios ao longo do tempo. Conforme solicitado no exercício, a função de ativação utilizada é a **Tangente Hiperbólica com parâmetro $\beta$ muito grande**:

$$x_i(t+1) = \tanh\left(\beta \sum_{j=1}^N w_{ij} x_j(t)\right)$$

Quando $\beta \to \infty$, o termo $\tanh(\beta u)$ atinge a saturação bipolar quase que instantaneamente para qualquer $u \neq 0$:

- Se a entrada total $u_i = \sum_{j} w_{ij} x_j(t) > 0$, então $\tanh(\beta u_i) \to +1$.
- Se a entrada total $u_i < 0$, então $\tanh(\beta u_i) \to -1$.
- Se a entrada total $u_i = 0$, o neurônio preserva seu estado anterior $x_i(t)$ (comportamento de memória/histerese).

Isso equivale matematicamente à função degrau bipolar (ou sinal):

$$x_i(t+1) = \operatorname{sgn}\left(\sum_{j=1}^N w_{ij} x_j(t)\right)$$

A rede pode ser atualizada de duas formas:
1. **Síncrona**: Todos os neurônios são atualizados simultaneamente a cada época. Isso pode levar a oscilações e ciclos infinitos de estado.
2. **Assíncrona**: Um neurônio por vez é selecionado (geralmente em ordem aleatória) e atualizado. Essa abordagem é a mais utilizada e **garante a convergência matemática** para um estado estável (atrator).

### 1.3 Função de Energia (Função de Lyapunov)

A estabilidade da rede de Hopfield assíncrona é demonstrada matematicamente definindo uma função de energia global, análoga à energia física de sistemas de spin em termodinâmica (modelos de Ising):

$$E(x) = -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N w_{ij} x_i x_j = -\frac{1}{2} x^T W x$$

Como os pesos são simétricos ($w_{ij} = w_{ji}$) e $w_{ii} = 0$, pode-se provar analiticamente que a cada atualização de um único neurônio $x_i \to x_i'$, a variação de energia $\Delta E = E(x') - E(x)$ satisfaz:

$$\Delta E = -(x_i' - x_i) \sum_{j=1}^N w_{ij} x_j \le 0$$

Como a variação de energia é estritamente não-positiva ($\Delta E \le 0$) e a energia é limitada inferiormente por um valor mínimo finito (pois os estados são limitados a $\pm 1$), a rede assíncrona necessariamente alcançará um **mínimo de energia local** (estado estável / ponto fixo) após um número finito de passos de atualização.

---

## 2. Detalhamento do Algoritmo e Código

A solução do exercício foi totalmente implementada em Python no arquivo [hopfield.py](file:///home/alunos/Desktop/www/pasta/Disciplina_IA/HopField/hopfield.py) sem o uso de bibliotecas de aprendizado de máquina prontas (como scikit-learn). As únicas bibliotecas utilizadas foram:
- `numpy`: para as operações de álgebra linear básica (produto externo de vetores `np.outer` e produto escalar `np.dot`).
- `matplotlib`: para plotar e salvar os gráficos das simulações (`simulacao_*.png`) e o gráfico de análise de ruído (`analise_ruido.png`).

A classe `HopfieldNetwork` contém os métodos essenciais:
- `train(patterns)`: recebe a lista de 4 padrões fornecidos (cada um com dimensão 45) e calcula a matriz de pesos $W$ com a diagonal zerada.
- `energy(state)`: calcula o valor de energia atual da rede.
- `update_asynchronous(initial_state, max_epochs, beta)`: executa a dinâmica assíncrona baseada na permutação aleatória de neurônios a cada época, atualizando os estados de acordo com $\tanh(\beta u_i)$ e rastreando a energia ao longo das iterações para comprovar matematicamente o decréscimo energético até a estabilização.

---

## 3. Resultados das 12 Simulações (Ruído = 20%)

Para cada um dos 4 padrões armazenados (Dígitos **1**, **2**, **3** e **4**), foram geradas 3 simulações de transmissão independentes introduzindo exatamente 20% de ruído (inversão aleatória de exatamente 9 pixels dos 45 totais).

Abaixo estão os resultados detalhados e visualizados lado a lado. Note que nas tabelas, ⬛ representa pixels pretos ($+1$) e ⬜ representa pixels brancos ($-1$).

---

### 3.1 Padrão 1 (Dígito 1)

*   **Padrão Alvo**:
    ```
    ..##.
    .###.
    ..##.
    ..##.
    ..##.
    ..##.
    ..##.
    ..##.
    ..##.
    ```

#### Situação 1
- **Convergência**: Sucesso
- **Épocas**: 2
- **Variação de Energia**: $-7.6444 \to -21.3333$

| Original | Transmitida (Ruidosa) | Recuperada |
| :---: | :---: | :---: |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬛` | `⬜⬜⬛⬛⬜` |
| `⬜⬛⬛⬛⬜` | `⬜⬛⬛⬜⬜` | `⬜⬛⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬜⬛⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬛⬛⬛⬛⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬛⬜⬛⬛⬛` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬛⬛⬜⬜` | `⬜⬜⬛⬛⬜` |

#### Situação 2
- **Convergência**: Sucesso
- **Épocas**: 2
- **Variação de Energia**: $-8.0000 \to -21.3333$

| Original | Transmitida (Ruidosa) | Recuperada |
| :---: | :---: | :---: |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬛⬛⬛⬜` | `⬜⬛⬜⬛⬜` | `⬜⬛⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬛⬛⬛⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬛⬜⬜⬜⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬛⬛⬛⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬛` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬜⬛` | `⬜⬜⬛⬛⬜` |

#### Situação 3
- **Convergência**: Sucesso
- **Épocas**: 2
- **Variação de Energia**: $-7.5556 \to -21.3333$

| Original | Transmitida (Ruidosa) | Recuperada |
| :---: | :---: | :---: |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬛⬛⬛⬜` | `⬜⬜⬛⬛⬛` | `⬜⬛⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬜⬛⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬛` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬜⬛⬛` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬜⬜⬛⬛⬛` | `⬜⬜⬛⬛⬜` |
| `⬜⬜⬛⬛⬜` | `⬛⬜⬛⬜⬜` | `⬜⬜⬛⬛⬜` |

---

### 3.2 Padrão 2 (Dígito 2)

*   **Padrão Alvo**:
    ```
    #####
    #####
    ...##
    ...##
    #####
    ##...
    ##...
    #####
    #####
    ```

#### Situação 1
- **Convergência**: **Falha** (Divergiu para um estado misturado/espúrio)
- **Épocas**: 2
- **Variação de Energia**: $-13.7778 \to -30.6667$

| Original | Transmitida (Ruidosa) | Recuperada |
| :---: | :---: | :---: |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬜⬛` | `⬛⬛⬛⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬛⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬛⬜⬛` | `⬜⬜⬜⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬜⬜⬜` | `⬜⬛⬜⬛⬜` | `⬛⬛⬜⬜⬛` |
| `⬛⬛⬜⬜⬜` | `⬜⬛⬜⬜⬜` | `⬛⬛⬜⬜⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬜⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬜⬛⬛` | `⬛⬛⬛⬛⬛` |

*Nota acadêmica*: Observe que o canto inferior direito recuperou dois pixels pretos adicionais indesejados (`⬛⬛⬜⬜⬛`), tornando a imagem recuperada uma mistura entre o Dígito 2 e o Dígito 3.

#### Situação 2
- **Convergência**: **Falha** (Divergiu para o mesmo atrator espúrio)
- **Épocas**: 2
- **Variação de Energia**: $-11.0222 \to -30.6667$

| Original | Transmitida (Ruidosa) | Recuperada |
| :---: | :---: | :---: |
| `⬛⬛⬛⬛⬛` | `⬛⬜⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬜⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬜⬛` | `⬜⬜⬜⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬜⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬜⬜⬜` | `⬛⬛⬛⬜⬜` | `⬛⬛⬜⬜⬛` |
| `⬛⬛⬜⬜⬜` | `⬛⬜⬜⬜⬛` | `⬛⬛⬜⬜⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬜⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬜⬛⬛` | `⬛⬛⬛⬛⬛` |

#### Situação 3
- **Convergência**: Sucesso
- **Épocas**: 2
- **Variação de Energia**: $-8.8889 \to -30.4889$

| Original | Transmitida (Ruidosa) | Recuperada |
| :---: | :---: | :---: |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬜⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬜⬛` | `⬜⬜⬜⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬜⬜⬜` | `⬛⬛⬛⬜⬜` | `⬛⬛⬜⬜⬜` |
| `⬛⬛⬜⬜⬜` | `⬛⬛⬛⬛⬜` | `⬛⬛⬜⬜⬜` |
| `⬛⬛⬛⬛⬛` | `⬜⬜⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬜⬜⬛⬛` | `⬛⬛⬛⬛⬛` |

---

### 3.3 Padrão 3 (Dígito 3)

*   **Padrão Alvo**:
    ```
    #####
    #####
    ...##
    ...##
    #####
    ...##
    ...##
    #####
    #####
    ```

#### Situação 1
- **Convergência**: Sucesso
- **Épocas**: 2
- **Variação de Energia**: $-10.5778 \to -33.8667$

| Original | Transmitida (Ruidosa) | Recuperada |
| :---: | :---: | :---: |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬜⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬛⬜⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬛⬛⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬛⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬜⬛⬛⬜⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬜⬛⬛⬛⬜` | `⬛⬛⬛⬛⬛` |

#### Situação 2
- **Convergência**: Sucesso
- **Épocas**: 2
- **Variação de Energia**: $-10.7556 \to -33.8667$

| Original | Transmitida (Ruidosa) | Recuperada |
| :---: | :---: | :---: |
| `⬛⬛⬛⬛⬛` | `⬜⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬜⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬜⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬛⬜⬛⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬜⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬛⬜⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬛⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬜⬛⬛` | `⬛⬛⬛⬛⬛` |

#### Situação 3
- **Convergência**: Sucesso
- **Épocas**: 2
- **Variação de Energia**: $-11.3778 \to -33.8667$

| Original | Transmitida (Ruidosa) | Recuperada |
| :---: | :---: | :---: |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬜⬛⬜⬛` | `⬛⬛⬛⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬜⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬛⬛⬛⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬜⬛` | `⬛⬛⬛⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬛⬜⬛⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` |

---

### 3.4 Padrão 4 (Dígito 4)

*   **Padrão Alvo**:
    ```
    ##.##
    ##.##
    ##.##
    #####
    #####
    ...##
    ...##
    ...##
    ...##
    ```

#### Situação 1
- **Convergência**: Sucesso
- **Épocas**: 2
- **Variação de Energia**: $-7.2000 \to -24.8889$

| Original | Transmitida (Ruidosa) | Recuperada |
| :---: | :---: | :---: |
| `⬛⬛⬜⬛⬛` | `⬛⬛⬜⬛⬜` | `⬛⬛⬜⬛⬛` |
| `⬛⬛⬜⬛⬛` | `⬛⬛⬜⬜⬛` | `⬛⬛⬜⬛⬛` |
| `⬛⬛⬜⬛⬛` | `⬛⬜⬜⬛⬛` | `⬛⬛⬜⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬜` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬜⬛⬛⬛⬜` | `⬛⬛⬛⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬛⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬛⬜⬜⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` |

#### Situação 2
- **Convergência**: Sucesso
- **Épocas**: 2
- **Variação de Energia**: $-7.1111 \to -24.8889$

| Original | Transmitida (Ruidosa) | Recuperada |
| :---: | :---: | :---: |
| `⬛⬛⬜⬛⬛` | `⬛⬛⬜⬛⬛` | `⬛⬛⬜⬛⬛` |
| `⬛⬛⬜⬛⬛` | `⬛⬛⬜⬜⬛` | `⬛⬛⬜⬛⬛` |
| `⬛⬛⬜⬛⬛` | `⬛⬛⬜⬛⬛` | `⬛⬛⬜⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬜⬛⬜⬛⬜` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬜⬛` | `⬛⬛⬛⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬛⬜⬛⬛⬜` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬜` | `⬜⬜⬜⬛⬛` |

#### Situação 3
- **Convergência**: Sucesso
- **Épocas**: 2
- **Variação de Energia**: $-8.0889 \to -24.8889$

| Original | Transmitida (Ruidosa) | Recuperada |
| :---: | :---: | :---: |
| `⬛⬛⬜⬛⬛` | `⬛⬛⬜⬛⬛` | `⬛⬛⬜⬛⬛` |
| `⬛⬛⬜⬛⬛` | `⬜⬛⬛⬜⬛` | `⬛⬛⬜⬛⬛` |
| `⬛⬛⬜⬛⬛` | `⬛⬛⬜⬛⬛` | `⬛⬛⬜⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬜` | `⬛⬛⬛⬛⬛` |
| `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` | `⬛⬛⬛⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬛⬜⬜⬛⬜` | `⬜⬜⬜⬛⬛` |
| `⬜⬜⬜⬛⬛` | `⬜⬜⬛⬜⬜` | `⬜⬜⬜⬛⬛` |

---

## 4. Análise Científica do Comportamento sob Ruído Excessivo

### 4.1 Definição de Bacias de Atração e Estados Espúrios

Uma rede de Hopfield armazena os padrões como **mínimos locais** na superfície de energia de Lyapunov. A vizinhança de estados ao redor de um padrão que converge para ele é chamada de **bacia de atração**. 
Quando um padrão ruidoso é inserido:
1. Se o ruído for pequeno (20% por exemplo), o estado inicial ainda reside dentro da bacia de atração do padrão original. À medida que a dinâmica assíncrona reduz a energia do sistema passo a passo, a rede caminha em direção ao vale de energia correto e recupera a imagem perfeitamente (Sucesso).
2. Se o ruído for excessivo (por exemplo, 40% a 50%), o estado ruidoso é arremessado para fora da bacia de atração original. O sistema então converge para a bacia de atração de outro padrão memorizado (**interferência cruzada** ou *crosstalk*), ou cai em **estados espúrios** (*spurious states*). 

Estados espúrios são mínimos locais de energia extras criados artificialmente pela regra de Hebb, que não correspondem a nenhum dos padrões originais armazenados. Matematicamente, a regra de Hebb é linear, o que faz com que combinações lineares simétricas (como misturas dos padrões) também se tornem mínimos locais de energia estáveis.

### 4.2 O Caso Crítico do Dígito 2
Conforme observado nas simulações 1 e 2 do Dígito 2, a rede falhou na recuperação, convergindo para o estado:
```
#####
#####
...##
...##
#####
##..#
##..#
#####
#####
```
Esse estado recuperado é exatamente uma mistura entre o Dígito 2 (`##...` no canto inferior esquerdo) e o Dígito 3 (`...##` no canto inferior direito). 
**Por que isso ocorreu?**
O Dígito 2 e o Dígito 3 são imagens muito parecidas estruturalmente (altamente correlacionadas). Eles compartilham 39 pixels idênticos dos 45 totais (cerca de 87% de correlação). Isso significa que as suas respectivas bacias de atração na superfície de energia estão extremamente próximas e estreitas. Quando adicionamos 20% de ruído ao Dígito 2, a probabilidade do estado inicial ser empurrado para a bacia do Dígito 3 ou para o mínimo espúrio misturado é consideravelmente alta.

### 4.3 Capacidade Limite da Rede (Limite de Hopfield)
A capacidade de armazenamento clássica de uma rede de Hopfield com recuperação livre de erros usando o aprendizado de Hebb é delimitada por:

$$P_{max} \approx 0.138 \times N$$

Para a nossa rede de $N = 45$ neurônios, a capacidade máxima ideal é:

$$P_{max} \approx 0.138 \times 45 \approx 6.21 \text{ padrões}$$

Como armazenamos 4 padrões, estamos utilizando aproximadamente 64% da capacidade total ideal da rede. Embora estejamos abaixo do limite teórico, a forte correlação física entre as imagens dos dígitos (especialmente 2 e 3) diminui severamente a barreira de energia que separa as suas bacias de atração. Isso faz com que a bacia de atração real do dígito 2 seja muito menor que a das outras imagens, explicando a sua falha de convergência sob 20% de ruído.

### 4.4 Análise Quantitativa da Robustez ao Ruído

Para obter uma resposta empírica rigorosa de como o aumento de ruído degrada o funcionamento da rede, foi implementado um teste de robustez em [hopfield.py](file:///home/alunos/Desktop/www/pasta/Disciplina_IA/HopField/hopfield.py) que varia o ruído de 0% a 100% (rodando 100 simulações estatísticas por nível). O gráfico resultante foi salvo no arquivo `analise_ruido.png`.

A curva exibe o seguinte comportamento típico:
- **Entre 0% e 10% de ruído**: A taxa de recuperação é de 100% ou muito próxima disso. A bacia de atração protege os estados.
- **Entre 15% e 30% de ruído**: Há uma queda acentuada na taxa de recuperação (onde a taxa passa de 100% para valores em torno de 60-70%). É nesse patamar que o ruído do exercício (20%) se encontra. O Dígito 2 começa a falhar sistematicamente devido ao *crosstalk* com o Dígito 3.
- **Acima de 40% de ruído**: A taxa de sucesso cai drasticamente para próximo de zero, demonstrando que a rede é incapaz de associar o ruído extremo a qualquer atrator correto.
- **Próximo de 100% de ruído**: Se invertermos 100% dos bits, a rede teoricamente recupera a imagem "negativa" do padrão (estados invertidos $-x$ também são mínimos estáveis da função de energia devido à simetria da rede $E(x) = E(-x)$).

---

## 5. Como Executar o Código

### Pré-requisitos
Certifique-se de ter o Python 3 instalado, juntamente com as bibliotecas `numpy` e `matplotlib`:
```bash
pip install numpy matplotlib
```

### Executando a Simulação
Para rodar a simulação completa, gerar as imagens de cada situação e o gráfico estatístico de robustez ao ruído, execute o comando:
```bash
python3 hopfield.py
```

### Arquivos Gerados
A execução gerará na pasta atual:
1. `simulacao_p{1..4}_s{1..3}.png`: 12 gráficos ilustrando cada situação individual (Original, Ruidosa e Recuperada).
2. `analise_ruido.png`: Gráfico da curva de robustez (Taxa de Recuperação Perfeita vs. Nível de Ruído).
