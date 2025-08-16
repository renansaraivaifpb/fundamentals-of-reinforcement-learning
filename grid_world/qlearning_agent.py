# -*- coding: utf-8 -*-
"""
Este script implementa um Agente Q-Learning para resolver o ambiente GridWorld.

Este agente aprende uma política ótima para um Processo de Decisão de Markov (MDP)
usando a equação de Bellman para atualizar os valores de estado-ação.
"""

import numpy as np

class QLearningAgent:
    """
    Um agente que aprende a navegar no GridWorld usando Q-Learning.

    Atributos:
        q_table (np.array): Tabela que armazena os valores Q para cada
                            par estado-ação. Dimensões: (linhas, colunas, ações).
        alpha (float): Taxa de aprendizado (learning rate).
        gamma (float): Fator de desconto para recompensas futuras.
        epsilon (float): Taxa de exploração na estratégia Epsilon-Greedy.
        num_actions (int): Número de ações possíveis.
    """
    def __init__(self, grid_size: tuple, num_actions: int, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        # A tabela Q armazena o valor de cada ação em cada estado (célula da grade)
        self.q_table = np.zeros(grid_size + (num_actions,))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

    def choose_action(self, state: tuple) -> int:
        """
        Escolhe uma ação usando a estratégia Epsilon-Greedy baseada na Tabela Q.

        Args:
            state (tuple): O estado atual (posição na grade).

        Returns:
            A ação a ser tomada.
        """
        if np.random.rand() < self.epsilon:
            # Exploração
            return np.random.randint(self.num_actions)
        else:
            # Explotação: escolhe a melhor ação com base nos valores Q atuais
            return np.argmax(self.q_table[state])

    def update(self, state: tuple, action: int, reward: float, next_state: tuple):
        """
        Atualiza a Tabela Q usando a equação de atualização do Q-Learning.

        Args:
            state (tuple): O estado de partida.
            action (int): A ação tomada.
            reward (float): A recompensa recebida.
            next_state (tuple): O estado resultante.
        """
        # Pega o valor Q atual para o par (estado, ação)
        old_value = self.q_table[state][action]

        # Encontra o valor Q máximo para o próximo estado
        next_max = np.max(self.q_table[next_state])

        # Calcula o novo valor Q usando a fórmula
        # NovoValor = (1-α)*ValorAntigo + α*(Recompensa + γ*ValorMáximoFuturo)
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        
        # Atualiza a Tabela Q
        self.q_table[state][action] = new_value