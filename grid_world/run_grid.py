# -*- coding: utf-8 -*-
"""
Script principal para simular e comparar o Agente Bandit e o Agente Q-Learning
no ambiente GridWorld.

Este script executa episódios de treinamento para ambos os agentes, coleta
as recompensas e imprime uma análise comparativa do desempenho.
"""

from grid_world.gridworld import GridWorld
from grid_world.bandit_agent import BanditAgent
from grid_world.qlearning_agent import QLearningAgent
import numpy as np

def run_simulation(agent, environment, num_episodes=1000):
    """
    Executa a simulação de treinamento para um determinado agente.

    Args:
        agent: A instância do agente a ser treinado (Bandit ou Q-Learning).
        environment: A instância do ambiente GridWorld.
        num_episodes (int): O número de episódios para treinar.

    Returns:
        Uma lista com a recompensa total de cada episódio.
    """
    total_rewards_per_episode = []
    
    for episode in range(num_episodes):
        state = environment.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = environment.step(action)
            
            # A lógica de atualização difere entre os agentes
            if isinstance(agent, QLearningAgent):
                agent.update(state, action, reward, next_state)
            elif isinstance(agent, BanditAgent):
                agent.update(state, action, reward)
            
            state = next_state
            total_reward += reward
        
        total_rewards_per_episode.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            print(f"Episódio {episode + 1}/{num_episodes} concluído.")
            
    return total_rewards_per_episode


if __name__ == "__main__":
    # --- Configurações da Simulação ---
    GRID_SIZE = (5, 5)
    NUM_EPISODES = 2000
    
    # --- Inicialização do Ambiente e Agentes ---
    env = GridWorld(grid_size=GRID_SIZE)
    
    bandit_agent = BanditAgent(
        grid_size=GRID_SIZE, 
        num_actions=env.num_actions, 
        epsilon=0.1
    )
    
    q_learning_agent = QLearningAgent(
        grid_size=GRID_SIZE, 
        num_actions=env.num_actions, 
        alpha=0.1, 
        gamma=0.9, 
        epsilon=0.1
    )
    
    # --- Execução das Simulações ---
    print("--- Treinando o Agente Bandit ---")
    bandit_rewards = run_simulation(bandit_agent, env, NUM_EPISODES)
    
    print("\n--- Treinando o Agente Q-Learning ---")
    q_learning_rewards = run_simulation(q_learning_agent, env, NUM_EPISODES)

    # --- Análise e Resultados ---
    print("\n\n--- ANÁLISE FINAL ---")
    
    avg_reward_bandit = np.mean(bandit_rewards[-100:])
    avg_reward_q_learning = np.mean(q_learning_rewards[-100:])

    print(f"\nRecompensa média (últimos 100 episódios) - Agente Bandit: {avg_reward_bandit:.2f}")
    print(f"Recompensa média (últimos 100 episódios) - Agente Q-Learning: {avg_reward_q_learning:.2f}")

    print("\nComportamento do Agente Bandit:")
    print("O Agente Bandit trata cada posição (estado) como um problema isolado. Ele pode aprender que, na posição (2,1), mover-se para a direita (para 2,2) resulta em uma recompensa imediata de -100. Ele aprenderá a evitar essa ação específica *a partir daquele estado*.")
    print("No entanto, ele não consegue 'planejar'. Ele não entende que mover-se para (1,2) pode ser ruim porque *leva* a uma vizinhança perigosa. Ele só aprende sobre uma ação depois de receber a recompensa (ou punição) direta dela.")

    print("\nComportamento do Agente Q-Learning (MDP):")
    print("O Agente Q-Learning, graças ao fator de desconto (gamma), propaga a informação sobre recompensas através da grade. O valor Q da célula (2,1) para a ação 'direita' se tornará muito negativo.")
    print("Crucialmente, a célula vizinha (1,1) também terá o valor da sua ação 'baixo' reduzido, porque a fórmula de atualização considera o valor máximo do *próximo estado*. Como o próximo estado (2,1) agora leva a um resultado ruim, o agente aprende a evitar não apenas a armadilha, mas também os caminhos que levam a ela. Ele aprende um verdadeiro 'mapa de perigo'.")
    
    print("\nConclusão:")
    print("O Agente Q-Learning (MDP) supera drasticamente o Agente Bandit em tarefas de navegação sequencial porque ele aprende a relação entre os estados, permitindo um planejamento a longo prazo. O Agente Bandit só otimiza para a recompensa imediata, o que é insuficiente quando as ações têm consequências futuras.")
    
    # Para visualizar os resultados, você pode usar matplotlib (descomente para usar)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.plot(bandit_rewards, label='Bandit Agent', alpha=0.6)
    # plt.plot(q_learning_rewards, label='Q-Learning Agent', alpha=0.6)
    # # Para suavizar a curva, pode-se plotar uma média móvel
    # moving_avg_q = np.convolve(q_learning_rewards, np.ones(100)/100, mode='valid')
    # plt.plot(moving_avg_q, label='Q-Learning (Média Móvel 100 ep.)', linewidth=2)
    # plt.title('Recompensa Total por Episódio Durante o Treinamento')
    # plt.xlabel('Episódio')
    # plt.ylabel('Recompensa Total')
    # plt.legend()
    # plt.grid(True)
    # plt.show()