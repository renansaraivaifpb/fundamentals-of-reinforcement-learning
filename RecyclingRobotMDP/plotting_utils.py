# -*- coding: utf-8 -*-
# plotting_utils.py

"""
Funções utilitárias para plotar os resultados do treinamento do agente de RL.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curve(rewards_per_episode: list, window_size: int = 100):
    """
    Plota a curva de aprendizado (recompensa média móvel por episódio).
    """
    # Calcula a média móvel para suavizar a curva
    moving_avg = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(12, 6))
    plt.plot(moving_avg)
    plt.title(f'Curva de Aprendizado (Média Móvel de {window_size} Episódios)', fontsize=16)
    plt.xlabel('Episódios', fontsize=12)
    plt.ylabel('Recompensa Média Acumulada', fontsize=12)
    plt.grid(True)
    plt.savefig('learning_curve.png')
    plt.show()

def plot_q_table_heatmap(q_table_history: dict, env_maps: tuple):
    """
    Plota a evolução da Tabela Q usando heatmaps.
    """
    states_map, actions_map = env_maps
    num_snapshots = len(q_table_history)
    
    fig, axes = plt.subplots(1, num_snapshots, figsize=(5 * num_snapshots, 4), sharey=True)
    fig.suptitle('Evolução da Tabela Q Durante o Treinamento', fontsize=18)
    
    for i, (episode, q_table) in enumerate(q_table_history.items()):
        ax = axes[i]
        sns.heatmap(q_table, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=False)
        ax.set_title(f'Episódio {episode}')
        ax.set_xticklabels(actions_map.values(), rotation=45)
    
    axes[0].set_yticklabels(states_map.values(), rotation=0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('q_table_evolution.png')
    plt.show()