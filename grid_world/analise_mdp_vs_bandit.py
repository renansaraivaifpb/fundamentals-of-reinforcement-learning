# -*- coding: utf-8 -*-
"""
Script completo para simular, analisar e visualizar a comparação entre um
Agente Bandit e um Agente Q-Learning em um ambiente GridWorld.

Este script irá:
1. Definir o ambiente, os agentes e a lógica de simulação.
2. Executar o treinamento para ambos os agentes.
3. Imprimir uma análise textual comparativa.
4. Gerar e salvar os gráficos em arquivos.
5. Exibir os gráficos em janelas interativas.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CLASSE DO AMBIENTE: GridWorld ---
class GridWorld:
    def __init__(self, grid_size=(5, 5)):
        self.grid_size = grid_size
        self.start_pos = (0, 0)
        self.current_pos = self.start_pos
        self.rewards = {
            (0, 4): 10,
            (1, 3): 3,
            (4, 4): 10,
            (2, 2): -100
        }
        self.terminal_states = list(self.rewards.keys())
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.num_actions = len(self.actions)

    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action):
        move = self.actions[action]
        next_pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])
        if not (0 <= next_pos[0] < self.grid_size[0] and 0 <= next_pos[1] < self.grid_size[1]):
            next_pos = self.current_pos
        self.current_pos = next_pos
        reward = self.rewards.get(self.current_pos, -0.1)
        done = self.current_pos in self.terminal_states
        return self.current_pos, reward, done

# --- CLASSE DO AGENTE 1: BanditAgent ---
class BanditAgent:
    def __init__(self, grid_size, num_actions, epsilon=0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.action_values = {}
        self.action_counts = {}

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            q_values = [self.action_values.get((state, a), 0) for a in range(self.num_actions)]
            return np.argmax(q_values)

    def update(self, state, action, reward):
        self.action_counts[(state, action)] = self.action_counts.get((state, action), 0) + 1
        count = self.action_counts[(state, action)]
        old_value = self.action_values.get((state, action), 0.0)
        new_value = old_value + (1 / count) * (reward - old_value)
        self.action_values[(state, action)] = new_value

# --- CLASSE DO AGENTE 2: QLearningAgent ---
class QLearningAgent:
    def __init__(self, grid_size, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros(grid_size + (num_actions,))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

# --- FUNÇÃO DE SIMULAÇÃO ---
def run_simulation(agent, environment, num_episodes=1000):
    total_rewards_per_episode = []
    print(f"Treinando {agent.__class__.__name__}...")
    for _ in tqdm(range(num_episodes)):
        state = environment.reset()
        total_reward = 0
        done = False
        for _ in range(100):
            action = agent.choose_action(state)
            next_state, reward, done = environment.step(action)
            if isinstance(agent, QLearningAgent):
                agent.update(state, action, reward, next_state)
            elif isinstance(agent, BanditAgent):
                agent.update(state, action, reward)
            state = next_state
            total_reward += reward
            if done:
                break
        total_rewards_per_episode.append(total_reward)
    return total_rewards_per_episode

# --- FUNÇÕES DE PLOTAGEM ---
def plot_learning_curves(bandit_rewards, q_learning_rewards, filename="learning_curves.png"):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    
    def moving_average(data, window_size=50):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    plt.plot(moving_average(bandit_rewards), label='Agente Bandit (Média Móvel)')
    plt.plot(moving_average(q_learning_rewards), label='Agente Q-Learning (Média Móvel)', linewidth=2)
    
    plt.title('Comparação de Desempenho: Recompensa por Episódio', fontsize=16)
    plt.xlabel('Episódios', fontsize=12)
    plt.ylabel('Recompensa Total Acumulada', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\nGráfico da curva de aprendizado salvo como '{filename}'")
    
    # NOVA LINHA: Exibe o gráfico em uma janela interativa
    plt.show()

def plot_q_learning_policy(agent, environment, filename="q_learning_policy.png"):
    policy = np.argmax(agent.q_table, axis=2)
    grid_size = environment.grid_size
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(np.zeros(grid_size), cmap="gray_r", linecolor="black", linewidths=1, cbar=False)

    action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            if (r, c) in environment.rewards:
                reward = environment.rewards[(r, c)]
                color = "green" if reward > 0 else "darkred"
                text = f"R={reward}"
                plt.text(c + 0.5, r + 0.5, text, ha='center', va='center', color='white', weight='bold')
                plt.gca().add_patch(plt.Rectangle((c, r), 1, 1, fill=True, color=color, alpha=0.6))
            else:
                action = policy[r, c]
                arrow = action_arrows[action]
                plt.text(c + 0.5, r + 0.5, arrow, ha='center', va='center', fontsize=20, color='blue')
    
    plt.title('Política Final Aprendida pelo Agente Q-Learning', fontsize=16)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Gráfico da política final salvo como '{filename}'")
    
    # NOVA LINHA: Exibe o mapa de política em uma janela interativa
    plt.show()

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    GRID_SIZE = (5, 5)
    NUM_EPISODES = 3000
    
    env = GridWorld(grid_size=GRID_SIZE)
    bandit_agent = BanditAgent(grid_size=GRID_SIZE, num_actions=env.num_actions, epsilon=0.1)
    q_learning_agent = QLearningAgent(grid_size=GRID_SIZE, num_actions=env.num_actions, alpha=0.1, gamma=0.9, epsilon=0.1)
    
    bandit_rewards = run_simulation(bandit_agent, env, NUM_EPISODES)
    q_learning_rewards = run_simulation(q_learning_agent, env, NUM_EPISODES)
    
    plot_learning_curves(bandit_rewards, q_learning_rewards)
    plot_q_learning_policy(q_learning_agent, env)
    
    print("\n\n--- ANÁLISE FINAL ---")
    avg_reward_bandit = np.mean(bandit_rewards[-200:])
    avg_reward_q_learning = np.mean(q_learning_rewards[-200:])
    print(f"\nRecompensa média (últimos 200 episódios) - Agente Bandit: {avg_reward_bandit:.2f}")
    print(f"Recompensa média (últimos 200 episódios) - Agente Q-Learning: {avg_reward_q_learning:.2f}")