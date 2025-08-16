# -*- coding: utf-8 -*-
# main_train.py

"""
Script principal para treinar um agente Q-Learning no ambiente do Robô de
Reciclagem e visualizar os resultados.
"""

from tqdm import tqdm
from robot_mdp_env import RecyclingRobotMDP
from q_learning_agent import QLearningAgent
from plotting_utils import plot_learning_curve, plot_q_table_heatmap
import numpy as np

if __name__ == "__main__":
    # --- Parâmetros do Ambiente ---
    ALPHA = 0.8
    BETA = 0.7
    R_SEARCH = 10
    R_WAIT = 1
    R_RESCUE = -20

    # --- Parâmetros do Treinamento ---
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.9
    EPSILON = 0.1
    NUM_EPISODES = 10000
    MAX_STEPS_PER_EPISODE = 50

    # --- Inicialização ---
    env = RecyclingRobotMDP(ALPHA, BETA, R_SEARCH, R_WAIT, R_RESCUE)
    agent = QLearningAgent(num_states=2, num_actions=3, alpha=LEARNING_RATE, gamma=DISCOUNT_FACTOR, epsilon=EPSILON)

    # --- Coleta de Dados para Plotagem ---
    rewards_history = []
    q_table_history = {0: agent.q_table.copy()} # Snapshot inicial
    snapshot_interval = NUM_EPISODES // 4 # Tira 4 snapshots durante o treino

    print("Iniciando treinamento do Agente Robô de Reciclagem...")
    
    # --- Loop de Treinamento ---
    for episode in tqdm(range(NUM_EPISODES)):
        state = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        rewards_history.append(episode_reward)

        # Salva um snapshot da Tabela Q em intervalos definidos
        if (episode + 1) % snapshot_interval == 0:
            q_table_history[episode + 1] = agent.q_table.copy()

    print("Treinamento concluído.\n")

    # --- Análise e Visualização ---
    print("--- Política Óptima Aprendida ---")
    best_action_high = env.actions_map[np.argmax(agent.q_table[env.STATE_HIGH, :2])]
    print(f"Bateria ALTA: a melhor ação é '{best_action_high}'")

    best_action_low = env.actions_map[np.argmax(agent.q_table[env.STATE_LOW])]
    print(f"Bateria BAIXA: a melhor ação é '{best_action_low}'\n")

    # Gerar e exibir os gráficos
    plot_learning_curve(rewards_history)
    plot_q_table_heatmap(q_table_history, (env.states_map, env.actions_map))