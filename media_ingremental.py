import numpy as np
import matplotlib.pyplot as plt

# Parâmetros da simulação
n_steps = 100  # número de interações
true_value = 0.6  # valor real esperado da ação (ad)
initial_Q = 0.0  # estimativa inicial
alpha_constant = 0.1  # taxa de aprendizado constante
np.random.seed(42)

# Funções de atualização
def sample_reward(true_mean):
    """Simula uma recompensa com base no valor verdadeiro da ação."""
    return np.random.binomial(1, true_mean)  # recompensa 1 com probabilidade = true_mean

def incremental_update(Q, R, alpha):
    """Atualiza Q com base na regra incremental geral."""
    return Q + alpha * (R - Q)

# Simulação com dois métodos de atualização: alpha fixo vs. média amostral (1/n)
n_steps = 100
true_value = 0.6
initial_Q = 0.0

# Inicializações
Q_fixed = initial_Q
Q_avg = initial_Q
alpha = 0.1

Q_values_fixed = []
Q_values_avg = []
counts = 0  # contador de quantas vezes a ação foi escolhida

for step in range(1, n_steps + 1):
    R = sample_reward(true_value)

    # Atualização com alpha fixo
    Q_fixed = incremental_update(Q_fixed, R, alpha)
    Q_values_fixed.append(Q_fixed)

    # Atualização com média amostral
    counts += 1
    step_size = 1 / counts
    Q_avg = incremental_update(Q_avg, R, step_size)
    Q_values_avg.append(Q_avg)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(Q_values_fixed, label='α fixo (0.1)', color='blue')
plt.plot(Q_values_avg, label='Média amostral (1/n)', color='orange')
plt.axhline(true_value, linestyle='--', color='green', label='Valor real da ação')
plt.xlabel('Etapas de interação (n)')
plt.ylabel('Estimativa Q')
plt.title('Comparação: α fixo vs. Média amostral')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
