# -*- coding: utf-8 -*-
# gridworld_value_func_env.py

"""
Define a classe de ambiente para o GridWorld usado na demonstração
de Funções de Valor.
"""

import numpy as np

class GridWorldValueFuncEnv:
    """
    Representa o ambiente GridWorld do vídeo sobre Funções de Valor.
    """
    def __init__(self, grid_size=(5, 5), gamma=0.9):
        self.grid_size = grid_size
        self.gamma = gamma

        # Posições dos estados especiais
        self.state_a_pos = (0, 1)
        self.state_a_prime_pos = (4, 1)
        self.state_b_pos = (0, 3)
        self.state_b_prime_pos = (2, 3)
        
        # Mapeamento de ações: 0: Cima, 1: Baixo, 2: Esquerda, 3: Direita
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def step(self, state: tuple, action: int) -> tuple[tuple, float]:
        """
        Executa uma ação a partir de um estado e retorna o próximo estado e a recompensa.

        Args:
            state (tuple): A posição atual (linha, coluna).
            action (int): A ação a ser tomada.

        Returns:
            Uma tupla (próximo_estado, recompensa).
        """
        # Checar se estamos em um estado especial
        if state == self.state_a_pos:
            return self.state_a_prime_pos, 10.0
        
        if state == self.state_b_pos:
            return self.state_b_prime_pos, 5.0

        # Calcular o próximo estado
        move = self.actions[action]
        next_state = (state[0] + move[0], state[1] + move[1])

        # Checar se bateu na parede
        if not (0 <= next_state[0] < self.grid_size[0] and 0 <= next_state[1] < self.grid_size[1]):
            # Bateu na parede, retorna ao mesmo estado e recebe penalidade
            return state, -1.0
        
        # Movimento normal, sem recompensa
        return next_state, 0.0