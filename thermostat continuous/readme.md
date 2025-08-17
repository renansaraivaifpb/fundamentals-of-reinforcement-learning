# Q-Learning Continuous Thermostat

## Descrição do Projeto
Este projeto implementa um **exemplo de aprendizado por reforço contínuo** usando **Q-Learning** em um ambiente simplificado de termostato.  

O objetivo é comparar o desempenho do agente com **diferentes fatores de desconto (γ)** em um ambiente contínuo com ruído.

O ambiente simula:
- **Estados**:
  - 0 = confortável
  - 1 = desconfortável (alguém reclamou)
- **Ações**:
  - 0 = aquecedor desligado
  - 1 = aquecedor ligado
- **Recompensas**:
  - 0 = ninguém reclamou
  - -1 = alguém reclamou (ruído aleatório)

## Estrutura do Código
- `ThermostatEnv`: Classe que implementa o ambiente contínuo.
- `QLearningAgent`: Classe que implementa o agente Q-Learning com política **ε-greedy**.
- `run_experiment()`: Função para treinar o agente e registrar recompensas médias.
- `q_learning_thermostat_continuous.py`: Script principal para execução do experimento.
- **Plot**: Gera gráfico da recompensa média ao longo dos episódios, comparando diferentes γ.

## Requisitos
- Python 3.7 ou superior
- Bibliotecas Python:
  ```bash
  pip install numpy matplotlib


## Experimentos

O script compara dois valores de γ:

γ = 0.3 → agente considera menos o futuro, aprendendo mais focado em recompensas imediatas.

γ = 0.99 → agente considera fortemente recompensas futuras, favorecendo políticas de longo prazo.

O gráfico resultante ajuda a visualizar como o fator de desconto afeta a aprendizagem em tarefas contínuas.
