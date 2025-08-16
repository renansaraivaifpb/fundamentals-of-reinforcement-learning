# -*- coding: utf-8 -*-
"""
Este script implementa um Agente Bandit para o ambiente GridWorld.

Este agente trata cada estado como um problema Multi-Armed Bandit independente,
ignorando as transições de estado. Ele é usado para demonstrar as limitações
dessa abordagem em problemas sequenciais.
"""

import numpy as np

class BanditAgent:
    """
    Um agente que usa uma abordagem de Multi-Armed Bandit para navegar.

    Este agente não tem noção de estados futuros. Para cada estado, ele
    aprende o valor de cada ação de forma independente.

    Atributos:
        grid_size (tuple): Dimensões da grade para criar a tabela de valores.
        num_actions (int): Número de ações possíveis.
        epsilon (float): Taxa de exploração (escolha de ação aleatória).
        action_values (dict): Dicionário para armazenar o valor estimado de cada
                              par (estado, ação).
        action_counts (dict): Dicionário para contar quantas vezes cada par
                              (estado, ação) foi escolhido.
    """
    def __init__(self, grid_size: tuple, num_actions: int, epsilon: float = 0.1):
        self.grid_size = grid_size
        self.num_actions = num_actions
        self.epsilon = epsilon
        
        # Estrutura para armazenar Q(s, a) - o valor estimado da ação 'a' no estado 's'
        self.action_values = {} # (estado, ação) -> valor
        self.action_counts = {} # (estado, ação) -> contagem

    def choose_action(self, state: tuple) -> int:
        """
        Escolhe uma ação usando a estratégia Epsilon-Greedy.

        Args:
            state (tuple): O estado atual do agente.

        Returns:
            A ação escolhida.
        """
        if np.random.rand() < self.epsilon:
            # Exploração: escolhe uma ação aleatória
            return np.random.randint(self.num_actions)
        else:
            # Explotação: escolhe a melhor ação conhecida para este estado
            best_action = -1
            max_value = -np.inf
            for action in range(self.num_actions):
                value = self.action_values.get((state, action), 0)
                if value > max_value:
                    max_value = value
                    best_action = action
            # Se nenhuma ação tiver sido explorada para este estado, escolhe aleatoriamente
            return best_action if best_action != -1 else np.random.randint(self.num_actions)

    def update(self, state: tuple, action: int, reward: float):
        """
        Atualiza o valor da ação com base na recompensa recebida.

        Usa uma média incremental simples.

        Args:
            state (tuple): O estado onde a ação foi tomada.
            action (int): A ação que foi tomada.
            reward (float): A recompensa recebida.
        """
        # Incrementa a contagem para o par (estado, ação)
        self.action_counts[(state, action)] = self.action_counts.get((state, action), 0) + 1
        count = self.action_counts[(state, action)]

        # Pega o valor antigo
        old_value = self.action_values.get((state, action), 0.0)

        # Atualiza o valor usando a fórmula de média incremental:
        # NovoValor = ValorAntigo + (1/N) * (Recompensa - ValorAntigo)
        new_value = old_value + (1 / count) * (reward - old_value)
        self.action_values[(state, action)] = new_value