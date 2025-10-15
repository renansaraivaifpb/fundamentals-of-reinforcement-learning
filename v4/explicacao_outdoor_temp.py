import numpy as np
import matplotlib.pyplot as plt

# Parâmetros simulados
dt = 1.0           # passo (h)
thermal_mass = 500.0
heat_transfer_coeff = 5.0
heat_gain_per_person = 80.0
ac_cooling_power = {0: 0.0, 1: 900.0, 2: -400.0}  # desligado, resfriar, aquecer
occupancy = 3
temp = 25.0
temps = []
hours = np.arange(0, 24, dt)
np_random = np.random.default_rng(42)

def outdoor_temp(hour):
    base, amp = 28.0, 8.0
    phase = (hour - 14) * np.pi / 12
    return base + amp * np.cos(phase)

for hour in hours:
    # Alterna entre estados de AC
    if 10 <= hour <= 18:
        ac_state = 1  # resfriar durante o dia
    else:
        ac_state = 0  # desligado à noite

    out = outdoor_temp(hour)
    people_heat = occupancy * heat_gain_per_person
    external_heat = heat_transfer_coeff * (out - temp)
    cooling_effect = ac_cooling_power[ac_state]
    total_heat_gain = people_heat + external_heat
    net_heat = total_heat_gain - cooling_effect

    temp_change = (net_heat / thermal_mass) * dt
    noise = np_random.normal(0, 0.05)
    temp += temp_change + noise
    temps.append(temp)

# Plot
plt.figure(figsize=(10,5))
plt.plot(hours, temps, label='Temperatura Interna', linewidth=2)
plt.plot(hours, [outdoor_temp(h) for h in hours], '--', label='Temperatura Externa')
plt.axhspan(21, 23, color='green', alpha=0.1, label='Faixa Ideal')
plt.xlabel('Hora do Dia')
plt.ylabel('Temperatura (°C)')
plt.title('Evolução Térmica da Sala (Simulação Simplificada)')
plt.legend()
plt.grid(True)
plt.show()
