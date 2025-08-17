# -*- coding: utf-8 -*-
# main_value_estimation.py

"""
Script principal para estimar e visualizar a Função de Valor de Estado (V)
para o GridWorld do vídeo, usando uma política aleatória.
"""

import numpy as np
from tqdm import tqdm
from gridworld_value_func_env import GridWorldValueFuncEnv
from value_function_estimator import estimate_state_value
from plotting_utils import plot_value_function_grid # Supondo que a função foi adicionada

if __name__ == "__main__":
    # --- Configurações ---
    GRID_SIZE = (5, 5)
    GAMMA = 0.9
    NUM_ACTIONS = 4
    
    # Parâmetros da Simulação Monte Carlo
    NUM_EPISODES_PER_STATE = 1000 # Mais episódios = estimativa mais precisa
    MAX_STEPS_PER_EPISODE = 50   # Horizonte de tempo para a tarefa contínua

    # --- Inicialização ---
    env = GridWorldValueFuncEnv(grid_size=GRID_SIZE, gamma=GAMMA)

    # Definindo a política aleatória uniforme, como no vídeo
    # Para cada estado (célula), a probabilidade de cada ação é 1/4 = 0.25
    random_policy = np.full(GRID_SIZE + (NUM_ACTIONS,), 0.25)

    # Matriz para armazenar os valores de V(s) calculados
    state_value_function_v = np.zeros(GRID_SIZE)

    print("Estimando a Função de Valor de Estado V(s) para a política aleatória...")

    # --- Loop de Estimação ---
    # Iteramos por cada estado da grade para calcular seu valor
    for r in tqdm(range(GRID_SIZE[0])):
        for c in range(GRID_SIZE[1]):
            start_state = (r, c)
            v_s = estimate_state_value(
                env,
                random_policy,
                start_state,
                NUM_EPISODES_PER_STATE,
                MAX_STEPS_PER_EPISODE
            )
            state_value_function_v[start_state] = v_s
    
    print("Estimação concluída.\n")

    # --- Análise e Visualização ---
    print("--- Função de Valor de Estado V(s) Estimada ---")
    # Imprime a matriz de valores com formatação
    print(np.round(state_value_function_v, 2))
    
    # Plota o heatmap
    plot_value_function_grid(state_value_function_v, f"V(s) para Política Aleatória (γ={GAMMA})")

    print("\n--- Análise dos Resultados ---")
    print("O heatmap visualiza o quão 'bom' é estar em cada estado sob uma política aleatória.")
    print("\n1. Valores Negativos perto das Paredes:")
    print("   Os estados nas bordas (especialmente cantos) têm valores baixos porque a política aleatória tem alta chance de colidir com a parede, gerando recompensas de -1.")
    
    print("\n2. Valor do Estado A (0,1) é MENOR que sua recompensa imediata (+10):")
    print(f"   V(A) ≈ {state_value_function_v[env.state_a_pos]:.2f}. Embora a recompensa imediata seja +10, a ação leva o agente para A' (4,1), um estado de baixo valor perto da parede. O futuro provável é negativo, e o valor total (imediato + futuro descontado) é menor que 10.")

    print("\n3. Valor do Estado B (0,3) é MAIOR que sua recompensa imediata (+5):")
    print(f"   V(B) ≈ {state_value_function_v[env.state_b_pos]:.2f}. A recompensa imediata é +5, mas a ação leva o agente para B' (2,3), um estado central de alto valor. Do centro, o agente está perto de A e B e longe das paredes. O futuro provável é positivo, elevando o valor total para acima de 5.")