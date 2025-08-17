# Adicione esta função ao seu arquivo plotting_utils.py

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ... (funções anteriores de plotagem aqui) ...

def plot_value_function_grid(value_function: np.ndarray, title: str):
    """
    Plota a função de valor de estado como um heatmap sobre a grade.
    """
    plt.figure(figsize=(8, 8))
    sns.heatmap(value_function, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
    plt.title(title, fontsize=16)
    plt.xlabel("Coluna")
    plt.ylabel("Linha")
    plt.gca().invert_yaxis() # Para o (0,0) ficar no canto superior esquerdo
    plt.savefig('value_function_grid.png')
    plt.show()