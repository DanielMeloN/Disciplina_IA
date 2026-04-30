# Trabalho de Inteligência Artificial: Perceptron
**Aluno(a):** [Seu Nome]

A partir da análise de um processo de destilação fracionada de petróleo, um Perceptron foi treinado usando a Regra de Hebb (algoritmo supervisionado) para classificar o óleo em duas classes de pureza, $C_1$ (-1) e $C_2$ (+1). A taxa de aprendizado utilizada foi $\eta = 0.01$.

## 1. Resultados de 5 Treinamentos
Para cada treinamento, o vetor de pesos foi inicializado aleatoriamente (valores entre 0 e 1).

| Treinamento | Vetor de Pesos Inicial ($w_0, w_1, w_2, w_3$) | Vetor de Pesos Final ($w_0, w_1, w_2, w_3$) | Número de Épocas |
| :---: | :--- | :--- | :---: |
| **1º (T1)** | `0.3086, 0.7302, 0.1141, 0.1997` | `-3.0514, 1.5187, 2.4525, -0.7255` | 384 |
| **2º (T2)** | `0.7142, 0.3148, 0.4276, 0.7687` | `-3.0858, 1.5544, 2.4884, -0.7365` | 420 |
| **3º (T3)** | `0.5018, 0.7895, 0.3677, 0.6528` | `-3.0982, 1.5697, 2.4802, -0.7383` | 413 |
| **4º (T4)** | `0.4865, 0.1687, 0.2498, 0.6144` | `-3.0535, 1.5507, 2.4705, -0.7298` | 405 |
| **5º (T5)** | `0.7090, 0.7951, 0.9576, 0.2966` | `-3.0310, 1.4746, 2.4633, -0.7242` | 399 |

## 2. Classificação Automática das Amostras de Óleo
Após o treinamento, o perceptron foi aplicado na classificação das amostras teste, baseando-se nos 5 vetores finais obtidos acima.

| Amostra | $x_1$ | $x_2$ | $x_3$ | $y(T_1)$ | $y(T_2)$ | $y(T_3)$ | $y(T_4)$ | $y(T_5)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | -0.3565 | 0.0620 | 5.9891 | -1 | -1 | -1 | -1 | -1 |
| **2** | -0.7842 | 1.1267 | 5.5912 | 1 | 1 | 1 | 1 | 1 |
| **3** | 0.3012 | 0.5611 | 5.8234 | 1 | 1 | 1 | 1 | 1 |
| **4** | 0.7757 | 1.0648 | 8.0677 | 1 | 1 | 1 | 1 | 1 |
| **5** | 0.1570 | 0.8028 | 6.3040 | 1 | 1 | 1 | 1 | 1 |
| **6** | -0.7014 | 1.0316 | 3.6005 | 1 | 1 | 1 | 1 | 1 |
| **7** | 0.3748 | 0.1536 | 6.1537 | -1 | -1 | -1 | -1 | -1 |
| **8** | -0.6920 | 0.9404 | 4.4058 | 1 | 1 | 1 | 1 | 1 |
| **9** | -1.3970 | 0.7141 | 4.9263 | -1 | -1 | -1 | -1 | -1 |
| **10** | -1.8842 | -0.2805 | 1.2548 | -1 | -1 | -1 | -1 | -1 |

## 3. Por que o número de épocas de treinamento varia a cada execução?
O número de épocas varia porque os pesos sinápticos iniciais do Perceptron são definidos com **valores aleatórios**.
Essa inicialização determina o ponto de partida do hiperplano de separação no espaço multidimensional. Dependendo da sorte desse ponto de partida em relação à distribuição dos dados, o hiperplano pode precisar sofrer mais ou menos ajustes (rotações e translações) pelo algoritmo até que consiga separar perfeitamente as duas classes $C_1$ e $C_2$.

## 4. Qual a principal limitação do perceptron quando aplicado em problemas de classificação de padrões?
A principal limitação estrutural de um perceptron simples é que ele **só é capaz de aprender e classificar padrões que sejam linearmente separáveis**.
Se o problema exigir uma fronteira de separação não-linear (como por exemplo o caso clássico da função lógica XOR), a regra de aprendizagem do Perceptron nunca irá convergir e ele permanecerá em um loop infinito, nunca encontrando um conjunto de pesos capaz de separar os dados perfeitamente.
