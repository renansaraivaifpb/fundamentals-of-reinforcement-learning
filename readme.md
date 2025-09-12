
# Fundamentals of Reinforcement Learning

Este repositório contém implementações e exemplos fundamentais para aprendizado por reforço (Reinforcement Learning - RL) em Python, focando em aspectos básicos como estimativa de funções de valor, simulação de agentes e exploração-exploração em ambientes simulados.

## Conteúdo do Repositório

- **value_function_estimator.py**  
  Funções para estimar a função de valor de estado $$V(s)$$ usando simulação Monte Carlo em um ambiente GridWorld personalizado.
  
- **valor_otimista.py**  
  Simulação de agente com abordagem otimista inicial versus algoritmo epsilon-greedy para o problema do multi-armed bandit (bandido de múltiplos braços).
  
- **plotting_utils.py**  
  Funções auxiliares para visualização, incluindo plotagem da função de valor em formato heatmap sobre grades.

- **media_incremental.py**  
  Comparação entre atualização incremental com taxa de aprendizado fixa e média amostral para estimativa de valores de ações.

- **main_value_estimation.py**  
  Script principal que executa a estimação e visualização da função de valor $$V(s)$$ para o ambiente GridWorld usando uma política aleatória.

- **gridworld_value_func_env.py**  
  Definição da classe de ambiente GridWorld para uso nas simulações e estimativas de funções de valor.

- **comparacao_epsilon_e_valor_otimista.py**  
  Comparação das estratégias epsilon-greedy e otimista em ambientes não-estacionários para o problema multi-armed bandit.

## Funcionalidades

- Utiliza simulação Monte Carlo para estimar o valor esperado de estados e ações em tarefas de aprendizado por reforço.
- Implementa políticas e agentes para resolver o problema clássico do bandido de múltiplos braços, explorando diferentes estratégias de exploração-exploração.
- Fornece visualizações gráficas para análise das funções de valor e desempenho dos agentes.
- Aborda atualização incremental e média amostral para aprendizado de valores.

## Como Usar

1. Clone o repositório:

```bash
git clone https://github.com/renansaraivaifpb/fundamentals-of-reinforcement-learning.git
cd fundamentals-of-reinforcement-learning
```

2. Execute o script principal para estimar e visualizar a função de valor no GridWorld:

```bash
python main_value_estimation.py
```

3. Explore os outros scripts para entender estratégias de aprendizado para bandits e métodos de atualização incremental.

## Requisitos

- Python 3.x
- NumPy
- Matplotlib
- Seaborn
- tqdm

Instale as dependências com:

```bash
pip install numpy matplotlib seaborn tqdm
```

## Estrutura do Ambiente GridWorld

O ambiente GridWorld implementado neste repositório consiste em uma grade 5x5 com estados especiais que fornecem recompensas imediatas positivas. O agente pode executar as ações mover para cima, baixo, esquerda ou direita. Colidir com paredes gera penalidades negativas. A política usada para estimar o valor do estado é inicialmente aleatória uniforme.

## Referências

Este repositório foi desenvolvido com base em conceitos fundamentais de aprendizagem por reforço, Monte Carlo, e métodos para estimar funções de valor conforme apresentados em literatura padrão da área.

***

Esta estrutura fornece uma visão clara do conteúdo, funcionalidades e como usar o repositório para reforçar o entendimento em aprendizado por reforço básico.[1]

[1](https://github.com/renansaraivaifpb/fundamentals-of-reinforcement-learning)

[6](https://github.com/0xangelo/curso-verao-rl-ime-2020)
[7](https://repositorio.ufmg.br/handle/1843/51162)
[8](https://www.reddit.com/r/reinforcementlearning/comments/1is773d/must_read_papers_for_reinforcement_learning/)
[9](https://repositorio.bc.ufg.br/items/4338f8ca-bc3d-4fec-ab7d-a756b15b4a18)
[10](https://www.youtube.com/watch?v=1QKwP0SJK-c)
