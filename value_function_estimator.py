# -*- coding: utf-8 -*-
# value_function_estimator.py

"""
Funções para estimar V(s) e Q(s,a) usando simulação Monte Carlo.
"""

import numpy as np
from gridworld_value_func_env import GridWorldValueFuncEnv

def calculate_discounted_return(rewards: list, gamma: float) -> float:
    """Calcula o retorno descontado para uma lista de recompensas."""
    g = 0.0
    for k, r in enumerate(rewards):
        g += (gamma ** k) * r
    return g

def estimate_state_value(env: GridWorldValueFuncEnv, policy: np.ndarray, start_state: tuple, num_episodes: int, max_steps: int) -> float:
    """
    Estima V(s) para um estado, seguindo uma política, via simulação Monte Carlo.
    """
    episode_returns = []
    for _ in range(num_episodes):
        current_state = start_state
        episode_rewards = []
        
        for _ in range(max_steps):
            # Escolhe a ação baseada na política fornecida para o estado atual
            action_probabilities = policy[current_state[0], current_state[1]]
            action = np.random.choice(len(action_probabilities), p=action_probabilities)
            
            next_state, reward = env.step(current_state, action)
            episode_rewards.append(reward)
            current_state = next_state
        
        # Calcula o retorno descontado para o episódio e o armazena
        episode_returns.append(calculate_discounted_return(episode_rewards, env.gamma))
        
    # V(s) é a média de todos os retornos obtidos
    return np.mean(episode_returns)