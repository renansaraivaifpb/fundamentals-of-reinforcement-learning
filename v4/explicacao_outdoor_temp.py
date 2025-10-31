import numpy as np
import matplotlib.pyplot as plt

# Parâmetros simulados
dt = 0.1  # passo (h). MODIFICADO de 1.0 para 0.1 (6 minutos)
thermal_mass = 500.0
heat_transfer_coeff = 5.0
heat_gain_per_person = 80.0
ac_cooling_power = {0: 0.0, 1: 900.0, 2: -400.0} # desligado, resfriar, aquecer
occupancy = 3
temp = 25.0
temps = []
np_random = np.random.default_rng(42)

# --- MODIFICAÇÃO AQUI ---
# Define o Eixo X em 240 passos
total_steps = 240
steps = np.arange(total_steps) # Eixo X: [0, 1, 2, ..., 239]

def outdoor_temp(hour):
    base, amp = 28.0, 8.0
    phase = (hour - 14) * np.pi / 12
    return base + amp * np.cos(phase)

# Converte os passos de volta para horas para calcular a temperatura externa
hours_mapped = steps * dt 
outdoor_temps_over_steps = [outdoor_temp(h) for h in hours_mapped]
# --- FIM DA MODIFICAÇÃO ---

# Plot
plt.figure(figsize=(10,5))

# Plota Eixo X (steps) vs Eixo Y (temperatura)
plt.plot(steps, outdoor_temps_over_steps, '--', label='Temperatura Externa')

plt.axhspan(21, 23, color='green', alpha=0.1, label='Faixa Ideal')
plt.xlabel('Passos de Tempo (6 min)') # Label do Eixo X atualizado
plt.ylabel('Temperatura (°C)')
plt.title('Evolução Térmica da Sala (Simulação Simplificada)')

# Adiciona ticks customizados no Eixo X para mostrar as horas
# (Mostra um marcador a cada 20 passos = 2 horas)
ticks_at_steps = np.arange(0, total_steps + 1, 20)
labels_at_hours = [f"{int(s * dt)}h" for s in ticks_at_steps]
plt.xticks(ticks_at_steps, labels_at_hours)

plt.legend()
plt.grid(True)
plt.show()