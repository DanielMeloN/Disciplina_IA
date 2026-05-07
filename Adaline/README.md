# ADALINE: Sistema Comutador de Válvulas Industriais

Este projeto apresenta a solução de um problema de classificação de sinais utilizando uma rede neural **ADALINE** e a **Regra Delta**, desenvolvido para a disciplina de Laboratório de Inteligência Artificial.

## 🐍 Como Executar (Versão Python)

Atendendo aos requisitos do projeto, a implementação matemática foi feita **100% em Python puro**, do zero, sem a utilização de bibliotecas de Machine Learning prontas como o Scikit-Learn ou PyTorch.

1. É necessário ter o Python instalado.
2. É necessário ter a biblioteca `matplotlib` instalada para a visualização dos gráficos. Caso não tenha, instale com:
   ```bash
   pip install matplotlib
   ```
3. Execute o script principal:
   ```bash
   python adaline.py
   ```
4. Ao rodar, o script irá:
   - Exibir a evolução gráfica (animação) do Erro Quadrático Médio para os treinamentos T1 e T2 ao vivo na tela.
   - Mostrar no terminal as tabelas formatadas com os pesos iniciais/finais e número de épocas dos 5 treinamentos.
   - Mostrar a tabela com os resultados da classificação das 15 amostras teste.
   - Imprimir a resposta da questão teórica solicitada no roteiro.
   - Salvar a imagem final do gráfico na mesma pasta (`grafico_eqm_t1_t2.png`).

*(Nota: Caso queira ver também a implementação alternativa interativa na web que havia sido construída antes, basta abrir o arquivo `index.html` no navegador).*

## 📝 Justificativa Teórica

**Questão: Embora o número de épocas de cada treinamento realizado seja diferente, explique por que então os valores dos pesos continuam praticamente inalterados.**

**Resposta:**
A rede ADALINE ajusta os pesos com o objetivo de minimizar a função de custo (o Erro Quadrático Médio - EQM). Para um único neurônio de ativação linear, essa função de custo desenha um hiperparaboloide, que é uma superfície estritamente convexa e possui apenas um **MÍNIMO GLOBAL**.

Não importa quais sejam os pesos iniciais gerados aleatoriamente; o algoritmo de gradiente descendente (Regra Delta) sempre conduzirá a solução para esse mesmo ponto de erro mínimo.

A diferença no número de épocas ocorre porque pesos iniciais sorteados muito longe do mínimo necessitam de mais iterações para chegar lá do que pesos sorteados que, por acaso, nasceram mais próximos ao "fundo do vale" do erro. Mas o destino final (os pesos finais da rede) sempre será o mesmo.

---
*Centro Federal de Educação Tecnológica de Minas Gerais - CEFET-MG | Campus Varginha*
