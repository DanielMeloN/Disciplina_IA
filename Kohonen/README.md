# Relatório Acadêmico: Redes de Kohonen (Mapas Auto-Organizáveis)

**Instituição**: Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG)  
**Campus**: Campus VIII – Varginha  
**Curso**: Bacharelado em Sistemas de Informação  
**Disciplina**: Laboratório de Inteligência Artificial  
**Professor**: Lázaro Eduardo da Silva  
**Data**: 28/05/2026  

---

## 1. Introdução Teórica

O **Mapa Auto-Organizável de Kohonen** (SOM - *Self-Organizing Map*), proposto pelo finlandês Teuvo Kohonen em 1982, é um modelo de rede neural artificial que utiliza aprendizado **não supervisionado** e **competitivo**. Sua principal finalidade é a redução de dimensionalidade e visualização de dados complexos, mapeando dados de entrada tridimensionais (ou de dimensões superiores) para um grid topológico bidimensional (normalmente retangular ou hexagonal) de neurônios de saída, preservando a **relação de vizinhança geométrica** (propriedades topológicas) dos dados originais.

### 1.1 Dinâmica de Treinamento e Neurônio Vencedor (BMU)

O processo de funcionamento do algoritmo é iterativo e baseia-se na competição, cooperação e adaptação dos neurônios da camada de saída:

1. **Competição**: Para cada amostra de entrada $x = [x_1, x_2, \dots, x_D]^T$, os neurônios da grade competem entre si. O neurônio cujos pesos sinápticos $w_j = [w_{j1}, w_{j2}, \dots, w_{jD}]^T$ possuem a maior similaridade (menor Distância Euclidiana) com $x$ é declarado o **Neurônio Vencedor** ou **BMU** (*Best Matching Unit*):
   
   $$j^* = \operatorname{argmin}_j \|x - w_j\| = \operatorname{argmin}_j \sqrt{\sum_{k=1}^D (x_k - w_{jk})^2}$$

2. **Cooperação**: O neurônio vencedor determina uma vizinhança topológica $N(j^*)$ na grade de saída. Os neurônios localizados dentro de um raio de vizinhança $r$ ao redor do BMU serão atualizados em conjunto com ele.
   
3. **Adaptação**: Os pesos sinápticos do neurônio vencedor e dos seus vizinhos espaciais são atualizados para se tornarem mais semelhantes ao padrão de entrada $x$:
   
   $$w_j(t+1) = w_j(t) + \alpha \cdot h(j, j^*, t) \cdot (x - w_j(t))$$
   
   Onde:
   - $\alpha$ é a taxa de aprendizado.
   - $h(j, j^*, t)$ é a função de vizinhança, que define o grau de atualização conforme a distância topológica do neurônio $j$ até o vencedor $j^*$ na grade. No nosso caso, é uma função degrau de raio $r=1$ baseada na distância de Manhattan na grade bidimensional.

---

## 2. Configuração e Especificações do Problema

O problema industrial apresentado consiste em agrupar amostras de imperfeições em borracha de pneus com base em 3 grandezas tridimensionais $\{x_1, x_2, x_3\}$. De acordo com os parâmetros do enunciado, a rede foi configurada da seguinte forma:
- **Camada de Entrada**: 3 neurônios (variáveis $x_1, x_2, x_3$).
- **Camada de Saída**: $N_1 = 16$ neurônios organizados em uma grade bidimensional de $4 \times 4$.
- **Taxa de Aprendizado**: $\alpha = 0.001$ constante.
- **Raio de Vizinhança**: $r = 1$. A distância utilizada para determinar a vizinhança na grade foi a distância de Manhattan na matriz $4 \times 4$:
  
  $$d_{\text{Manhattan}}(j_1, j_2) = |row_1 - row_2| + |col_1 - col_2|$$
  
  Portanto, para um neurônio vencedor na coordenada $(row_{bmu}, col_{bmu})$, são atualizados apenas os neurônios localizados a uma distância máxima de 1 célula (o próprio vencedor, e as células diretamente acima, abaixo, à esquerda e à direita).
- **Número de Épocas**: 10.000 épocas com embaralhamento dos dados em cada época para garantir convergência e organização topológica ótimas.
- **Divisão das Classes de Treinamento**:
  - Amostras 1 a 20   : **Classe A**
  - Amostras 21 a 60  : **Classe B**
  - Amostras 61 a 120 : **Classe C**

---

## 3. Respostas das Questões do Trabalho

### Questão 1: Indicação dos conjuntos de neurônios no grid que respondem às Classes A, B e C

Ao mapear as 120 amostras de treinamento na grade $4 \times 4$ estabilizada após o treinamento com semente fixa (`seed = 42`), observou-se uma **organização topológica perfeita**, com separação contígua em regiões geográficas no grid e formação de células vazias nas divisas que agem como fronteiras de decisão:

*   **Classe A** (Amostras 1-20): Ocupa a região **inferior direita** do grid.
*   **Classe B** (Amostras 21-60): Ocupa a região **superior direita** do grid.
*   **Classe C** (Amostras 61-120): Ocupa toda a metade **esquerda** (colunas 0 e 1) do grid.

A tabela de mapeamento dos neurônios (índices 0 a 15 de cima para baixo, da esquerda para a direita) está distribuída conforme o diagrama topológico abaixo:

| Linha \ Coluna | Coluna 0 ($c=0$) | Coluna 1 ($c=1$) | Coluna 2 ($c=2$) | Coluna 3 ($c=3$) |
| :---: | :---: | :---: | :---: | :---: |
| **Linha 0 ($r=0$)** | **Neurônio 0**: `Classe C` | **Neurônio 1**: `Classe C` | **Neurônio 2**: `Classe B` | **Neurônio 3**: `Classe B` |
| **Linha 1 ($r=1$)** | **Neurônio 4**: `Classe C` | **Neurônio 5**: `Classe C` | **Neurônio 6**: `Classe B` | **Neurônio 7**: `Classe B` |
| **Linha 2 ($r=2$)** | **Neurônio 8**: `Classe C` | *Neurônio 9*: `Vazio` | **Neurônio 10**: `Classe A` | *Neurônio 11*: `Vazio` |
| **Linha 3 ($r=3$)** | **Neurônio 12**: `Classe C` | *Neurônio 13*: `Vazio` | **Neurônio 14**: `Classe A` | **Neurônio 15**: `Classe A` |

#### Resumo das Zonas por Classe:
- **Conjunto Classe A**: Neurônios $\{10, 14, 15\}$ (coordenadas: `(2,2)`, `(3,2)`, `(3,3)`).
- **Conjunto Classe B**: Neurônios $\{2, 3, 6, 7\}$ (coordenadas: `(0,2)`, `(0,3)`, `(1,2)`, `(1,3)`).
- **Conjunto Classe C**: Neurônios $\{0, 1, 4, 5, 8, 12\}$ (coordenadas: `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`, `(2,0)`, `(3,0)`).
- **Neurônios Inativos/Fronteira**: Neurônios $\{9, 11, 13\}$ (coordenadas: `(2,1)`, `(2,3)`, `(3,1)`).

---

### Questão 2: Classificação das Amostras de Teste

Apresentando as 12 amostras de teste à rede treinada e calculando o neurônio vencedor (BMU) correspondente para cada uma, obtivemos o mapeamento exato e a classificação indicados na tabela a seguir:

| Amostra | $x_1$ | $x_2$ | $x_3$ | Neurônio BMU | Classe do Neurônio | Classe Atribuída |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.2471 | 0.1778 | 0.2905 | `(2, 2)` | Classe A | **Classe A** |
| 2 | 0.8240 | 0.2223 | 0.7041 | `(1, 3)` | Classe B | **Classe B** |
| 3 | 0.4960 | 0.7231 | 0.5866 | `(1, 1)` | Classe C | **Classe C** |
| 4 | 0.2923 | 0.2041 | 0.2234 | `(2, 2)` | Classe A | **Classe A** |
| 5 | 0.8118 | 0.2668 | 0.7484 | `(1, 3)` | Classe B | **Classe B** |
| 6 | 0.4837 | 0.8200 | 0.4792 | `(1, 0)` | Classe C | **Classe C** |
| 7 | 0.3248 | 0.2629 | 0.2375 | `(2, 2)` | Classe A | **Classe A** |
| 8 | 0.7209 | 0.2116 | 0.7821 | `(0, 3)` | Classe B | **Classe B** |
| 9 | 0.5259 | 0.6522 | 0.5957 | `(0, 1)` | Classe C | **Classe C** |
| 10 | 0.2075 | 0.1669 | 0.1745 | `(3, 2)` | Classe A | **Classe A** |
| 11 | 0.7830 | 0.3171 | 0.7888 | `(0, 3)` | Classe B | **Classe B** |
| 12 | 0.5393 | 0.7510 | 0.5682 | `(0, 1)` | Classe C | **Classe C** |

---

### Questão 3: Demonstração da Regra de Alteração de Pesos ("Norma Euclidiana")

A **regra de atualização dos pesos sinápticos** do neurônio vencedor $j^*$ na Rede de Kohonen é comumente expressa de forma vetorial como:

$$\Delta w_{j^*} = \alpha (x - w_{j^*})$$

Abaixo, demonstra-se analiticamente que essa equação é deduzida diretamente a partir da minimização da função de erro baseada no **desvio quadrático** entre o vetor de entrada $x$ e o vetor de pesos $w_{j^*}$ do neurônio vencedor:

#### 1. Definição da Função de Custo (Erro Quadrático)
Definimos a função de erro local $E$ com relação ao neurônio vencedor $j^*$ como metade do quadrado da Norma Euclidiana da diferença de vetores:

$$E = \frac{1}{2} \|x - w_{j^*}\|^2 = \frac{1}{2} \sum_{k=1}^D (x_k - w_{j^*k})^2$$

Onde:
- $D$ é a dimensão do espaço de entrada ($D=3$ no nosso caso).
- $x_k$ é a componente $k$ do padrão de entrada $x$.
- $w_{j^*k}$ é o peso de conexão sináptica entre a componente $k$ da entrada e o neurônio vencedor $j^*$.

#### 2. Minimização via Algoritmo de Gradiente Descendente
Para minimizar a função de erro $E$ em relação a cada peso sináptico $w_{j^*k}$, utilizamos o algoritmo de otimização de gradiente descendente. O ajuste do peso deve ocorrer no sentido oposto à inclinação da derivada parcial do erro com relação a esse mesmo peso:

$$\Delta w_{j^*k} = - \alpha \frac{\partial E}{\partial w_{j^*k}}$$

Onde $\alpha$ é o passo de aprendizado (taxa de aprendizado).

#### 3. Cálculo da Derivada Parcial
Aplicando a regra da cadeia para diferenciar $E$ com relação à componente $w_{j^*k}$:

$$\frac{\partial E}{\partial w_{j^*k}} = \frac{\partial}{\partial w_{j^*k}} \left[ \frac{1}{2} \sum_{r=1}^D (x_r - w_{j^*r})^2 \right]$$

Como a derivada do somatório em relação a $w_{j^*k}$ é nula para todas as componentes $r \neq k$, restando apenas o termo $r=k$:

$$\frac{\partial E}{\partial w_{j^*k}} = \frac{1}{2} \cdot \frac{d}{d w_{j^*k}} \left[ (x_k - w_{j^*k})^2 \right]$$

Utilizando a regra da cadeia do cálculo elementar ($d(u^2) = 2u \cdot du$):

$$\frac{\partial E}{\partial w_{j^*k}} = \frac{1}{2} \cdot 2 \cdot (x_k - w_{j^*k}) \cdot \frac{d}{dw_{j^*k}}(x_k - w_{j^*k})$$

Dado que $x_k$ é independente de $w_{j^*k}$ (a entrada é fixa durante o passo):

$$\frac{\partial E}{\partial w_{j^*k}} = (x_k - w_{j^*k}) \cdot (0 - 1) = -(x_k - w_{j^*k})$$

#### 4. Aplicação do Gradiente Descendente
Substituindo a derivada parcial obtida na equação do gradiente descendente:

$$\Delta w_{j^*k} = - \alpha \left[ -(x_k - w_{j^*k}) \right]$$

$$\Delta w_{j^*k} = \alpha (x_k - w_{j^*k})$$

#### 5. Escrita em Notação Vetorial
Reunindo todos os componentes $k = 1, \dots, D$ no formato vetorial:

$$\Delta w_{j^*} = \alpha (x - w_{j^*})$$

Portanto, a regra de alteração de pesos ("Norma Euclidiana") da Rede de Kohonen é **formalmente deduzida a partir da minimização da função erro quadrático** via método clássico do gradiente descendente. **Q.E.D.**

---

## 4. Análise e Discussão dos Resultados

### 4.1 Separação Geométrica em 3D
Analisando o gráfico de dispersão gerado `dados_3d.png`, é visível que as três classes definidas formam agrupamentos perfeitamente isolados no espaço tridimensional:
- **Classe A**: Concentrada no canto inferior com valores de $x_1$ baixos/médios, $x_2$ muito baixos e $x_3$ baixos/médios.
- **Classe B**: Concentrada na região superior com valores de $x_1$ muito altos, $x_2$ baixos/médios e $x_3$ altos.
- **Classe C**: Concentrada no centro-superior esquerdo com valores de $x_1$ médios, $x_2$ muito altos e $x_3$ médios.

### 4.2 Auto-Organização Topológica do Grid
O mapa auto-organizável de Kohonen conseguiu capturar essa topologia geométrica 3D e projetá-la de forma consistente na matriz 2D de $4 \times 4$ (salvo em `grid_kohonen.png`):
- A metade esquerda do grid (`c=0` e `c=1`) representa inteiramente o território da **Classe C**.
- A metade direita superior (`c=2` e `c=3` nas linhas `r=0` e `r=1`) mapeia o território da **Classe B**.
- A metade direita inferior (`c=2` e `c=3` nas linhas `r=2` e `r=3`) mapeia o território da **Classe A**.
- Os neurônios vazios `N9(2,1)`, `N11(2,3)` e `N13(3,1)` aparecem como amortecedores/barreiras entre as regiões de A, B e C, garantindo que as fronteiras fiquem bem definidas e não haja sobreposição ou mistura de classes em uma mesma vizinhança.

Esta distribuição consistente comprova a eficácia da dinâmica competitiva não supervisionada e a capacidade de representação do grid de Kohonen.

---

## 5. Como Executar o Código

### Pré-requisitos
A execução requer Python 3 instalado com as dependências `numpy` e `matplotlib`:
```bash
pip install numpy matplotlib
```

### Executando o Script
Para treinar a rede, exibir os dados no terminal e salvar as imagens tridimensional e bidimensional dos resultados, execute:
```bash
python3 kohonen.py
```

### Arquivos Gerados
1. `dados_3d.png`: Dispersão tridimensional das 120 amostras de treino.
2. `grid_kohonen.png`: Grade topológica $4 \times 4$ ilustrando as regiões mapeadas.
