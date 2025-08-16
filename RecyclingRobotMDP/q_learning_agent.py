# -*- coding: utf-8 -*-
# q_learning_agent.py

"""
Define a classe do Agente que aprende usando Q-Learning.
"""

import numpy as np
from robot_mdp_env import RecyclingRobotMDP

class QLearningAgent:
    """
    Agente que aprende a política ótima para um MDP usando Q-Learning.
    """
    def __init__(self, num_states: int, num_actions: int, alpha: float, gamma: float, epsilon: float):
        self.q_table = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

    def choose_action(self, state: int) -> int:
        """Escolhe uma ação usando a política Epsilon-Greedy."""
        if np.random.rand() < self.epsilon:
            if state == RecyclingRobotMDP.STATE_HIGH:
                return np.random.choice([RecyclingRobotMDP.ACTION_SEARCH, RecyclingRobotMDP.ACTION_WAIT])
            else:
                return np.random.randint(self.num_actions)
        else:
            if state == RecyclingRobotMDP.STATE_HIGH:
                q_values = self.q_table[state, :2]
                return np.argmax(q_values)
            else:
                return np.argmax(self.q_table[state])

    def update(self, state: int, action: int, reward: float, next_state: int):
        """Atualiza a Tabela Q com base na experiência."""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state, action] = new_value