import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Parâmetros extraídos da sua classe ClassroomConfig ---
ideal_temp = 24.0
comfort_bonus = 5.0
comfort_sensitivity = 0.5
temp_comfort_min = 22.0
temp_comfort_max = 26.0

# --- 1. Geração de Dados ---
temperatures = np.linspace(15, 35, 400)
delta_temps = temperatures - ideal_temp
rewards = comfort_bonus - comfort_sensitivity * (delta_temps ** 2)

# --- 2. Plotagem do Gráfico ---
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(temperatures, rewards, label='Recompensa Quadrática', color='royalblue', lw=3)
ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.7)
ax.axvline(ideal_temp, color='green', linestyle=':', lw=2, label=f'Temperatura Ideal ({ideal_temp}°C)')
ax.axvspan(temp_comfort_min, temp_comfort_max, color='green', alpha=0.1, label='Faixa de Conforto Oficial')
ax.plot(ideal_temp, comfort_bonus, 'go', markersize=10)
ax.annotate(f'Bônus Máximo: +{comfort_bonus:.1f}', xy=(ideal_temp, comfort_bonus), xytext=(ideal_temp + 0.5, comfort_bonus - 0.5),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

# --- 3. ADICIONA PONTOS DE INTERESSE PARA MOSTRAR PENALIDADES ---
points_to_show = {
    "Borda do Conforto": 26.0,
    "Fora do Conforto": 28.0,
    "Muito Quente": 30.0
}

for label, temp in points_to_show.items():
    # Calcula a recompensa para o ponto específico
    reward_at_point = comfort_bonus - comfort_sensitivity * ((temp - ideal_temp) ** 2)
    
    # Plota o ponto no gráfico
    ax.plot(temp, reward_at_point, 'o', color='crimson', markersize=10)
    
    # Adiciona a anotação com o valor da recompensa
    ax.annotate(f'{label} ({temp}°C)\nRecompensa: {reward_at_point:.1f}',
                xy=(temp, reward_at_point),
                xytext=(temp, reward_at_point - 2.5), # Posiciona o texto um pouco abaixo do ponto
                ha='center', va='top', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

# --- 4. Configuração Final do Gráfico ---
ax.set_title('Visualização da Função de Recompensa Quadrática com Penalidades', fontsize=18, weight='bold')
ax.set_xlabel('Temperatura Atual da Sala (°C)', fontsize=14)
ax.set_ylabel('Recompensa de Conforto Calculada', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, which='both', linestyle='--')

plt.show()