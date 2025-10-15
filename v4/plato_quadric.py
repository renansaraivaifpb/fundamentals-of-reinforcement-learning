# Salve como plot_rewards_comparison.py e execute

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Parâmetros
ideal_temp = 24.0
comfort_bonus = 5.0
comfort_sensitivity = 0.5
temp_comfort_min = 22.0
temp_comfort_max = 26.0
temperatures = np.linspace(15, 35, 400)

# Cálculo da recompensa quadrática pura
rewards_quadratic = comfort_bonus - comfort_sensitivity * ((temperatures - ideal_temp) ** 2)

# Cálculo da recompensa "Platô Quadrático"
rewards_plateau = []
for temp in temperatures:
    if temp_comfort_min <= temp <= temp_comfort_max:
        reward = comfort_bonus
    elif temp > temp_comfort_max:
        delta = temp - temp_comfort_max
        reward = comfort_bonus - comfort_sensitivity * (delta ** 2)
    else: # temp < temp_comfort_min
        delta = temp_comfort_min - temp
        reward = comfort_bonus - comfort_sensitivity * (delta ** 2)
    rewards_plateau.append(reward)

# Plotagem
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 8))
plt.plot(temperatures, rewards_quadratic, label='Quadrática Pura (pico em 24°C)', color='royalblue', lw=2, linestyle='--')
plt.plot(temperatures, rewards_plateau, label='Platô Quadrático (sua sugestão)', color='darkorange', lw=3)
plt.axvspan(temp_comfort_min, temp_comfort_max, color='green', alpha=0.1, label='Faixa de Conforto')
plt.axhline(0, color='black', linestyle=':', lw=1)
plt.title('Comparação de Funções de Recompensa de Conforto', fontsize=16)
plt.xlabel('Temperatura (°C)')
plt.ylabel('Recompensa')
plt.legend()
plt.show()