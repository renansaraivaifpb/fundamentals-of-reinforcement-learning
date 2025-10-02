# Sistema de Controle de Ar-Condicionado com Aprendizagem por Reforço

Este projeto implementa um sistema inteligente de controle de ar-condicionado para salas de aula usando **Aprendizagem por Reforço (RL)**. O objetivo é desenvolver um agente que aprenda a controlar o sistema de climatização de forma a balancear o **conforto térmico** dos usuários e a **eficiência energética**.

## 🎯 Objetivos

- **Conforto Térmico**: Manter a temperatura da sala dentro da faixa de conforto (22-26°C)
- **Eficiência Energética**: Minimizar o consumo energético do ar-condicionado
- **Adaptabilidade**: Aprender políticas que se adaptem a diferentes cenários (ocupação, horário, temperatura externa)

## 🏗️ Arquitetura do Sistema

### Ambiente de Simulação (`classroom_ac_env.py`)
- **Estados**: Temperatura, ocupação, hora do dia, estado do AC
- **Ações**: OFF, LOW, MEDIUM, HIGH (4 níveis de potência)
- **Recompensas**: Balanceamento entre conforto térmico e consumo energético
- **Dinâmica**: Modelo térmico simplificado com transferência de calor

### Agente Q-Learning (`ac_qlearning_agent.py`)
- **Algoritmo**: Q-Learning com política ε-greedy
- **Exploração**: Decaimento adaptativo da taxa de exploração
- **Avaliação**: Métricas de performance e análise de políticas

### Sistema de Análise (`analysis_utils.py`)
- **Visualizações**: Heatmaps, gráficos de temperatura, análise de eficiência
- **Relatórios**: Análise detalhada da política aprendida
- **Dashboard**: Interface completa de análise de resultados

## 📊 Características Técnicas

### Modelagem do Ambiente
- **Sala de Aula**: 8m × 6m × 3m (144 m³)
- **Capacidade**: Até 30 ocupantes
- **Temperatura de Conforto**: 22-26°C
- **Estados Discretos**: 2,520 estados (21 temp × 6 ocupação × 20 hora)
- **Ações**: 4 níveis de potência do ar-condicionado

### Função de Recompensa
```python
# Recompensa por conforto térmico
comfort_rewards = {
    'VERY_COLD': -2.0,
    'COLD': -1.0,
    'COMFORTABLE': 1.0,
    'WARM': -1.0,
    'VERY_HOT': -2.0
}

# Penalidade por consumo energético
energy_penalty = -consumption * 0.1

# Recompensa total
total_reward = comfort_reward + energy_penalty
```

### Parâmetros de Treinamento
- **Taxa de Aprendizado (α)**: 0.1
- **Fator de Desconto (γ)**: 0.95
- **Taxa de Exploração (ε)**: 0.2 → 0.01 (decaimento)
- **Episódios**: 1,000
- **Duração do Episódio**: 24 horas (240 passos de tempo)

## 🚀 Como Usar

### Instalação
```bash
# Instalar dependências
pip install numpy matplotlib seaborn pandas

# Clonar repositório
git clone <repository-url>
cd air_conditioning_classroom
```

### Treinamento Básico
```python
from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig
from ac_qlearning_agent import ACQLearningAgent, QLearningConfig

# Cria ambiente e agente
env = ClassroomACEnvironment(ClassroomConfig())
agent = ACQLearningAgent(env.n_states, env.n_actions, QLearningConfig())

# Treina o agente
history = agent.train(env, verbose=True)

# Avalia a política
eval_stats = agent.evaluate(env, episodes=10)
```

### Execução Completa
```bash
# Executa experimentos comparativos
python main_train.py

# Cria dashboard de análise
python analysis_utils.py
```

## 📈 Resultados Esperados

### Métricas de Performance
- **Conforto Térmico**: >80% do tempo na faixa de conforto
- **Eficiência Energética**: Consumo otimizado baseado na demanda
- **Convergência**: Estabilização da política após ~500 episódios

### Comportamentos Aprendidos
- **Temperatura Baixa + Ocupação Alta**: AC em potência média/alta
- **Temperatura Alta + Ocupação Baixa**: AC em potência baixa
- **Horário Noturno**: Redução do uso do AC
- **Temperatura Confortável**: AC desligado ou baixa potência

## 🔬 Experimentos Disponíveis

### 1. Baseline
- Configuração padrão para comparação
- α=0.1, γ=0.95, ε=0.2

### 2. High Exploration
- Maior exploração inicial
- ε=0.5, decaimento mais lento

### 3. Low Learning Rate
- Taxa de aprendizado reduzida
- α=0.05 para aprendizado mais suave

### 4. High Discount Factor
- Maior consideração do futuro
- γ=0.99 para planejamento de longo prazo

## 📊 Análises Disponíveis

### 1. Heatmap da Q-Table
- Visualização dos valores Q aprendidos
- Identificação de padrões na política

### 2. Análise de Temperatura vs Conforto
- Relação entre temperatura e nível de conforto
- Efetividade do controle térmico

### 3. Análise de Eficiência Energética
- Conforto vs Consumo energético
- Otimização da razão eficiência

### 4. Relatório da Política
- Análise detalhada das regras aprendidas
- Recomendações baseadas no comportamento

## 🎛️ Configurações Avançadas

### Personalização do Ambiente
```python
config = ClassroomConfig(
    length=10.0,           # Comprimento da sala (m)
    width=8.0,             # Largura da sala (m)
    height=3.0,            # Altura da sala (m)
    max_occupancy=40,      # Capacidade máxima
    temp_comfort_min=21.0, # Temperatura mínima confortável
    temp_comfort_max=25.0  # Temperatura máxima confortável
)
```

### Personalização do Agente
```python
agent_config = QLearningConfig(
    alpha=0.15,            # Taxa de aprendizado
    gamma=0.98,            # Fator de desconto
    epsilon=0.3,           # Taxa de exploração inicial
    epsilon_min=0.005,     # Taxa de exploração mínima
    episodes=2000          # Número de episódios
)
```

## 📁 Estrutura do Projeto

```
air_conditioning_classroom/
├── classroom_ac_env.py      # Ambiente de simulação
├── ac_qlearning_agent.py    # Agente Q-Learning
├── main_train.py           # Script principal de treinamento
├── analysis_utils.py       # Utilitários de análise
├── README.md               # Este arquivo
└── results_YYYYMMDD_HHMMSS/ # Resultados dos experimentos
    ├── experiment_comparison.png
    ├── policy_behavior_analysis.png
    ├── baseline_model.pkl
    └── baseline_results.json
```

## 🔧 Dependências

- **Python**: 3.7+
- **NumPy**: Computação numérica
- **Matplotlib**: Visualizações
- **Seaborn**: Gráficos estatísticos
- **Pandas**: Manipulação de dados

## 📚 Referências

1. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction*. MIT Press.
2. **Watkins, C. J. C. H.** (1989). Learning from delayed rewards. *PhD thesis, University of Cambridge*.
3. **Mnih, V., et al.** (2015). Human-level control through deep reinforcement learning. *Nature*.

## 🤝 Contribuições

Contribuições são bem-vindas! Áreas de interesse:
- Melhorias no modelo térmico
- Novos algoritmos de RL
- Otimizações de performance
- Novas métricas de avaliação

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 👨‍💻 Autor

**Renan** - Desenvolvido com assistência de IA para fins educacionais em Aprendizagem por Reforço.

---

*Este projeto demonstra a aplicação prática de algoritmos de RL em problemas de controle de sistemas físicos, especificamente no domínio de eficiência energética e conforto térmico.*