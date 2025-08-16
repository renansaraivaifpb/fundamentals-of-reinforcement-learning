# -*- coding: utf-8 -*-
"""
Este script define o ambiente GridWorld para simulações de Aprendizagem por Reforço.

O ambiente consiste em uma grade 2D onde um agente (o coelho) pode se mover.
Existem locais com recompensas positivas (comida) e negativas (perigos).
"""

import numpy as np

class GridWorld:
    """
    Representa um ambiente de grade 2D (GridWorld).

    Este ambiente define o espaço de estados, ações, transições e recompensas
    para um problema de Aprendizagem por Reforço.

    Atributos:
        grid (np.array): Matriz que representa o layout do mundo.
        start_pos (tuple): Posição inicial do agente.
        current_pos (tuple): Posição atual do agente.
    """

    def __init__(self, grid_size=(5, 5)):
        """
        Inicializa o ambiente GridWorld.

        Args:
            grid_size (tuple): As dimensões (linhas, colunas) da grade.
        """
        if not (isinstance(grid_size, tuple) and len(grid_size) == 2):
            raise ValueError("grid_size deve ser uma tupla de dois inteiros.")
        
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size)
        
        # Define a posição inicial do agente (coelho)
        self.start_pos = (0, 0)
        self.current_pos = self.start_pos

        # Define as recompensas no grid
        # Chave: posição (linha, coluna), Valor: recompensa
        self.rewards = {
            (0, 4): 10,   # Cenoura
            (1, 3): 3,    # Brócolis
            (4, 4): 10,   # Cenoura (objetivo final)
            (2, 2): -100 # Tigre
        }
        
        # Define os estados terminais (onde o episódio acaba)
        self.terminal_states = list(self.rewards.keys())

        # Define o mapa de ações possíveis
        # 0: Cima, 1: Baixo, 2: Esquerda, 3: Direita
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.num_actions = len(self.actions)

    def reset(self) -> tuple:
        """
        Reseta o ambiente para a posição inicial.

        Returns:
            A posição inicial do agente.
        """
        self.current_pos = self.start_pos
        return self.current_pos

    def get_state(self) -> tuple:
        """Retorna a posição (estado) atual do agente."""
        return self.current_pos

    def step(self, action: int) -> tuple:
        """
        Executa uma ação no ambiente.

        Args:
            action (int): A ação a ser executada (0 a 3).

        Returns:
            Uma tupla contendo (próximo_estado, recompensa, terminado).
        """
        if action not in self.actions:
            raise ValueError("Ação inválida.")

        # Calcula a mudança de posição com base na ação
        move = self.actions[action]
        next_pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])

        # Verifica se o agente bateu na parede
        # Se sim, ele permanece no mesmo lugar
        if not (0 <= next_pos[0] < self.grid_size[0] and 0 <= next_pos[1] < self.grid_size[1]):
            next_pos = self.current_pos

        # Atualiza a posição do agente
        self.current_pos = next_pos

        # Obtém a recompensa
        reward = self.rewards.get(self.current_pos, -0.1) # Penalidade pequena por movimento

        # Verifica se o estado é terminal
        done = self.current_pos in self.terminal_states

        return self.current_pos, reward, done