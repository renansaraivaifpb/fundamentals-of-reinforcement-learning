## Descrição

O código simula um ambiente com 10 ações, onde cada ação tem um valor esperado desconhecido para o agente. O objetivo é maximizar a recompensa acumulada ao escolher entre as ações.

- O algoritmo ε-greedy escolhe ações com uma política greedy, mas explora aleatoriamente com probabilidade ε = 0.1.
- O algoritmo UCB usa os limites de confiança para balancear exploração e exploração de forma otimizada.

## Resultados Esperados

O gráfico deve mostrar que o algoritmo UCB geralmente tem melhor desempenho que ε-greedy em termos de recompensa média acumulada.

![Gráfico de comparação]("UCB_vs_ε-greedy.png")
