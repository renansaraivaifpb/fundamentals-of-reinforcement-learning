# Autor: Renan Saraiva dos Santos

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output

# --- Configuração do Ambiente Grid World ---
GRID_SIZE = 4
GOAL_STATE = 15
PENALTY_STATE = 10
STATES = np.arange(GRID_SIZE * GRID_SIZE)
ACTIONS = ['↑', '↓', '←', '→'] # Cima, Baixo, Esquerda, Direita
GAMMA = 0.9 # Fator de desconto

# --- Inicialização ---
# V(s) inicializado com zeros para todos os estados
V = np.zeros(len(STATES))

# Política Aleatória: probabilidade de 0.25 para cada ação em qualquer estado
policy = np.ones([len(STATES), len(ACTIONS)]) / len(ACTIONS)

def get_next_state_and_reward(state, action_idx):
    """
    Função de dinâmica do ambiente (determinística)
    Retorna (próximo_estado, recompensa)
    """
    if state == GOAL_STATE or state == PENALTY_STATE:
        return state, 0 # Estados terminais

    row, col = state // GRID_SIZE, state % GRID_SIZE
    action = ACTIONS[action_idx]

    if action == '↑': row = max(0, row - 1)
    elif action == '↓': row = min(GRID_SIZE - 1, row + 1)
    elif action == '←': col = max(0, col - 1)
    elif action == '→': col = min(GRID_SIZE - 1, col + 1)

    next_state = row * GRID_SIZE + col

    if next_state == GOAL_STATE:
        reward = 1
    elif next_state == PENALTY_STATE:
        reward = -1
    else:
        reward = 0

    return next_state, reward

def plot_value_function(V, iteration):
    """Função para plotar a função de valor como um heatmap."""
    V_grid = V.reshape((GRID_SIZE, GRID_SIZE))
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(V_grid, cmap='viridis')

    # Adicionar texto com os valores em cada célula
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            state_val = V_grid[i, j]
            text_color = "black" if state_val > -0.5 else "white"
            ax.text(j, i, f"{state_val:.2f}", ha="center", va="center", color=text_color, fontsize=10)
    
    # Adicionar marcadores para estados especiais
    goal_row, goal_col = GOAL_STATE // GRID_SIZE, GOAL_STATE % GRID_SIZE
    penalty_row, penalty_col = PENALTY_STATE // GRID_SIZE, PENALTY_STATE % GRID_SIZE
    ax.text(goal_col, goal_row, "🏆", ha="center", va="center", fontsize=20)
    ax.text(penalty_col, penalty_row, "🔥", ha="center", va="center", fontsize=20)

    ax.set_title(f"Função de Valor (V) - Iteração: {iteration}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8, label="Valor do Estado")
    display(fig)
    plt.savefig(f'Bellman/value_function_iteration_{iteration}.png')
    plt.close(fig)


# --- Algoritmo de Iteração de Valor (Animação) ---
MAX_ITERATIONS = 50
CONVERGENCE_THRESHOLD = 1e-4

print("Iniciando a Iteração de Valor para encontrar V(s)...")
print("Observe como os valores se propagam a partir do 🏆 (+1) e 🔥 (-1).")

plot_iterations = [0, 1, 2, 5, 10, 49]

for i in range(MAX_ITERATIONS):
    delta = 0
    V_old = V.copy()

    # Itera sobre cada estado s
    for s in STATES:
        v = V[s] # Valor antigo para comparação
        new_v = 0
        
        # Aplicando a Equação de Bellman
        # Soma sobre todas as ações 'a'
        for a_idx, action in enumerate(ACTIONS):
            prob_action = policy[s, a_idx] # pi(a|s)
            
            # Como a dinâmica é determinística, p(s', r | s, a) é 1 para um único par (s', r)
            next_s, r = get_next_state_and_reward(s, a_idx)
            
            # Bellman: r + gamma * V(s')
            term = r + GAMMA * V[next_s]
            new_v += prob_action * term
        
        V[s] = new_v
        delta = max(delta, abs(v - V[s]))

    # Plotando em iterações específicas para simular a animação
    if i in plot_iterations:
        clear_output(wait=True)
        print(f"Iteração {i+1}/{MAX_ITERATIONS} - Mudança Máxima (delta): {delta:.5f}")
        plot_value_function(V, i + 1)
        time.sleep(2) # Pausa para visualização

    # Critério de parada
    if delta < CONVERGENCE_THRESHOLD:
        print(f"\nConvergência atingida na iteração {i+1}!")
        break

clear_output(wait=True)
print("Função de Valor Final Convergida:")
plot_value_function(V, 'Final')
