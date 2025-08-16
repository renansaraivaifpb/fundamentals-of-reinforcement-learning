# -*- coding: utf-8 -*-
# robot_mdp_env.py

"""
Define a classe de ambiente para o Processo de Decisão de Markov (MDP)
do problema do Robô de Reciclagem.
"""

import numpy as np

class RecyclingRobotMDP:
    """
    Representa o ambiente MDP do Robô de Reciclagem.
    """
    # Constantes para legibilidade
    STATE_HIGH = 0
    STATE_LOW = 1
    
    ACTION_SEARCH = 0
    ACTION_WAIT = 1
    ACTION_RECHARGE = 2

    def __init__(self, alpha: float, beta: float, r_search: float, r_wait: float, r_rescue: float):
        self.alpha = alpha
        self.beta = beta
        self.r_search = r_search
        self.r_wait = r_wait
        self.r_rescue = r_rescue
        
        self.current_state = self.STATE_HIGH
        self.actions_map = {0: "Search", 1: "Wait", 2: "Recharge"}
        self.states_map = {0: "High", 1: "Low"}

    def reset(self) -> int:
        """Reseta o ambiente para o estado inicial."""
        self.current_state = self.STATE_HIGH
        return self.current_state

    def step(self, action: int) -> tuple[int, float, bool]:
        """Executa uma ação no ambiente e retorna o resultado."""
        if self.current_state == self.STATE_HIGH:
            if action == self.ACTION_SEARCH:
                if np.random.rand() < self.alpha:
                    next_state = self.STATE_HIGH
                else:
                    next_state = self.STATE_LOW
                return next_state, self.r_search, False
            elif action == self.ACTION_WAIT:
                return self.STATE_HIGH, self.r_wait, False
            else:
                return self.current_state, -1, False 

        elif self.current_state == self.STATE_LOW:
            if action == self.ACTION_SEARCH:
                if np.random.rand() < self.beta:
                    return self.STATE_LOW, self.r_search, False
                else:
                    return self.STATE_HIGH, self.r_rescue, True
            elif action == self.ACTION_WAIT:
                return self.STATE_LOW, self.r_wait, False
            elif action == self.ACTION_RECHARGE:
                return self.STATE_HIGH, 0, False
        
        raise ValueError("Estado inválido encontrado.")