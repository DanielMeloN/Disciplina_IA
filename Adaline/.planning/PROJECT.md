# Project: Adaline - Sistema Comutador de Válvulas

## Context
O objetivo deste projeto é implementar uma rede neural ADALINE utilizando a Regra Delta para classificar sinais ruidosos de um sistema industrial, indicando se os dados devem ser encaminhados para a Válvula A (-1) ou Válvula B (+1). Além disso, o projeto possui alto rigor acadêmico, exigindo relatórios visuais da evolução do Erro Quadrático Médio (EQM) e a apresentação detalhada dos resultados dos treinamentos.

## Core Value
Garantir a correta classificação de sinais ruidosos através de 5 treinamentos independentes da rede ADALINE, fornecendo uma interface visual de altíssima qualidade para acompanhamento do treinamento, plotagem de gráficos de erro, e análise profunda dos pesos neurais para um trabalho acadêmico de excelência.

## Requirements

### Validated
(None yet)

### Active
- [ ] **REQ-1:** Implementar a rede ADALINE e o algoritmo de treinamento Regra Delta.
- [ ] **REQ-2:** Executar 5 treinamentos (T1 a T5) inicializando os pesos aleatoriamente entre 0 e 1, com taxa de aprendizado η = 0.0025 e precisão ε = 10^-6.
- [ ] **REQ-3:** Registrar os vetores de pesos iniciais, finais e número de épocas para os 5 treinamentos.
- [ ] **REQ-4:** Desenvolver um método visual em tempo real (gráfico) para observar a evolução do EQM por época para T1 e T2.
- [ ] **REQ-5:** Classificar 15 amostras de teste utilizando os pesos finais de todos os treinamentos (T1 a T5).
- [ ] **REQ-6:** Responder à questão teórica: "Por que os valores dos pesos finais continuam praticamente inalterados, mesmo com números de épocas diferentes?".
- [ ] **REQ-7:** Criar um `README.md` detalhado e profissional.

### Out of Scope
- [ ] Treinamento com outras arquiteturas neurais além do ADALINE (escopo estrito ao trabalho).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Implementação via Web App Interativo | O usuário solicitou um "método visual para ir vendo a alteração do gráfico", e uma aplicação web (HTML/JS/CSS) proporciona a melhor experiência interativa e "wow factor" para trabalhos de faculdade. | — Pending |
| Uso da biblioteca Chart.js | Facilita a animação dos gráficos de erro por época de forma leve e nativa no browser. | — Pending |

---
*Last updated: 2026-05-07 after initialization*
