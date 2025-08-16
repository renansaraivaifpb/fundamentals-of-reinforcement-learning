# Comparação de Agentes de Aprendizagem por Reforço: Bandit vs. Q-Learning no GridWorld

Este projeto apresenta uma simulação completa em Python para explorar e comparar duas abordagens fundamentais de Aprendizagem por Reforço (RL): um agente stateless (Multi-Armed Bandit) e um agente state-aware (Q-Learning) em um ambiente de navegação 2D conhecido como GridWorld.

O objetivo principal é demonstrar visualmente as limitações de uma abordagem que considera apenas recompensas imediatas (Bandit) em contraste com a robustez de uma abordagem baseada em Processos de Decisão de Markov (MDP) que planeja a longo prazo (Q-Learning).

---

## Visão Geral do Projeto

Em muitos problemas de decisão sequencial, a ação escolhida em um determinado momento não afeta apenas a recompensa imediata, mas também as situações futuras e, consequentemente, as recompensas futuras. Este projeto explora essa dinâmica através de dois agentes com "filosofias" diferentes:

* O **Agente Bandit** trata cada posição na grade como um dilema isolado, tentando aprender a melhor ação para aquela posição específica sem qualquer noção de para onde essa ação o levará.
* O **Agente Q-Learning** constrói um "mapa mental" do ambiente, aprendendo o valor de tomar uma ação em um estado, considerando a recompensa imediata e o potencial de recompensas futuras a partir do estado seguinte.

A simulação treina ambos os agentes no mesmo ambiente e plota seus resultados, oferecendo uma comparação clara de desempenho e estratégia.

## Estrutura dos Arquivos

O projeto está organizado nos seguintes arquivos:

* `analise_mdp_vs_bandit.py`: Script principal que orquestra a simulação, treina os agentes, executa a análise e gera os gráficos de resultados.
* `gridworld.py`: (Implementado dentro do script principal) Define a classe `GridWorld`, que representa o ambiente 2D, incluindo os estados, ações, transições e recompensas.
* `bandit_agent.py`: (Implementado dentro do script principal) Define a classe `BanditAgent`, que interage com o ambiente usando uma estratégia Epsilon-Greedy para cada estado de forma independente.
* `qlearning_agent.py`: (Implementado dentro do script principal) Define a classe `QLearningAgent`, que implementa o algoritmo Q-Learning para aprender uma política ótima baseada na equação de Bellman.
* `learning_curves.png`: Gráfico gerado que compara a recompensa acumulada por episódio para ambos os agentes.
* `q_learning_policy.png`: Gráfico gerado que visualiza a política final aprendida pelo agente Q-Learning.

## Como Executar

Siga os passos abaixo para rodar a simulação e gerar os resultados.

### 1. Pré-requisitos

Certifique-se de ter Python 3 instalado. Você precisará das seguintes bibliotecas, que podem ser instaladas via `pip`:

```bash
pip install numpy matplotlib seaborn tqdm
