# Resumo: Perceptron Multicamadas (com conexão direta)

Neste documento, explico como funciona o algoritmo de Backpropagation para a rede neural descrita no arquivo `PMC.pdf`.

A principal diferença dessa arquitetura para um Perceptron comum é que ela tem uma **conexão direta da entrada para a saída** (representada pela matriz W3), além do caminho normal que passa pela camada escondida.

## Estrutura da Rede:
- **Camada 1 (Entrada):** Recebe 'N' sinais.
- **Camada 2 (Escondida):** Possui 'N1' neurônios.
- **Camada 3 (Saída):** Possui apenas 1 neurônio.

As matrizes de pesos são divididas assim:
- **W1:** Liga a entrada na camada escondida.
- **W2:** Liga a camada escondida na saída.
- **W3:** Liga a entrada *direto* na saída.

---

## 1. Passo Forward (Propagação)

Primeiro, o sinal entra na rede e vai avançando para gerar a resposta.

- **Na camada escondida:** Multiplicamos o sinal de entrada pelos pesos W1 e passamos o resultado na função de ativação para gerar a saída dos neurônios intermediários.
- **Na camada de saída:** O neurônio final recebe os sinais que saíram da camada escondida (multiplicados por W2) **E** os sinais que vieram direto da entrada (multiplicados por W3). Somamos tudo e passamos na função de ativação para ter a resposta final da rede.

---

## 2. Passo Backward (Ajuste dos Pesos)

Depois de ver a resposta da rede, comparamos com o valor que a gente queria que ela desse para achar o erro. Aí usamos o Backpropagation para ajustar os pesos de trás pra frente:

- **Ajustando W2 e W3 (Saída):** O erro do neurônio final é usado para corrigir essas duas matrizes. Como as duas chegam direto na saída, a conta é parecida: usamos a taxa de aprendizado, o erro da saída e o sinal que originou cada uma (a saída da escondida para W2, e a entrada original para W3).
- **Ajustando W1 (Escondida):** O erro da saída é "jogado pra trás" para os neurônios da camada escondida. Detalhe importante: esse erro só volta pelo caminho W2 (já que W3 não passa pela camada escondida). Com o erro repassado calculado, a gente consegue ajustar os pesos iniciais da matriz W1.

Ao final, atualizamos W1, W2 e W3 com esses valores novos. A gente repete esse processo inteiro para todos os dados de treinamento, várias vezes (épocas), até a rede aprender e o erro ficar bem pequeno.
