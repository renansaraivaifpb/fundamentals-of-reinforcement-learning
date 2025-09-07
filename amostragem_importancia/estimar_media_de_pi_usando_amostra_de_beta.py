# Autor: Renan Saraiva dos Santos

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. SETUP DO EXPERIMENTO: AS DUAS DISTRIBUIÇÕES
# ==============================================================================

# Usaremos distribuições discretas (como em um dado viciado)
outcomes = np.arange(0, 11) # Resultados possíveis de 0 a 10

# Distribuição Alvo (π): Queremos estimar a média desta.
# Note que ela favorece valores mais altos (pico em 6).
pi_target_probs = np.array([0.01, 0.02, 0.05, 0.07, 0.1, 0.25, 0.25, 0.1, 0.08, 0.05, 0.02])
pi_target = {x: p for x, p in zip(outcomes, pi_target_probs)}

# Distribuição de Comportamento (b): Vamos tirar amostras desta.
# Note que ela favorece valores mais baixos (pico em 2).
b_behavior_probs = np.array([0.1, 0.2, 0.3, 0.15, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
b_behavior = {x: p for x, p in zip(outcomes, b_behavior_probs)}

# --- Calculando as médias verdadeiras para nossa referência ---
true_mean_pi = sum(x * p for x, p in pi_target.items())
true_mean_b = sum(x * p for x, p in b_behavior.items())

# --- Adicionando verificação e normalização das probabilidades ---
if not np.isclose(np.sum(pi_target_probs), 1.0):
    print(f"Warning: pi_target_probs sum to {np.sum(pi_target_probs)}. Normalizing...")
    pi_target_probs = pi_target_probs / np.sum(pi_target_probs)

if not np.isclose(np.sum(b_behavior_probs), 1.0):
    print(f"Warning: b_behavior_probs sum to {np.sum(b_behavior_probs)}. Normalizing...")
    b_behavior_probs = b_behavior_probs / np.sum(b_behavior_probs)


# ==============================================================================
# 2. A SIMULAÇÃO
# ==============================================================================
N_SAMPLES = 2000

# Listas para guardar o histórico das estimativas
naive_estimates = []
is_estimates = []

# Variáveis para calcular a média incrementalmente
sum_of_samples = 0
sum_of_weighted_samples = 0

print("Iniciando a simulação...")
for i in range(1, N_SAMPLES + 1):
    # Tira uma amostra APENAS da distribuição de comportamento 'b'
    sample = np.random.choice(outcomes, p=b_behavior_probs)

    # --- Média Ingênua (errada) ---
    sum_of_samples += sample
    naive_avg = sum_of_samples / i
    naive_estimates.append(naive_avg)

    # --- Média com Amostragem de Importância (correta) ---
    # Calcula a razão de importância (o fator de correção)
    prob_pi = pi_target.get(sample, 0)
    prob_b = b_behavior.get(sample, 0)

    # Evita divisão por zero se a amostra for impossível em 'b'
    if prob_b == 0:
        rho = 0
    else:
        rho = prob_pi / prob_b

    # Pondera a amostra pela razão de importância
    weighted_sample = rho * sample
    sum_of_weighted_samples += weighted_sample
    is_avg = sum_of_weighted_samples / i
    is_estimates.append(is_avg)

print("Simulação concluída. Gerando gráficos...")
# ==============================================================================
# 3. VISUALIZAÇÃO DOS RESULTADOS
# ==============================================================================

sns.set_style("whitegrid")
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# --- Gráfico 1: As Distribuições ---
ax = axs[0]
width = 0.35
ax.bar(outcomes - width/2, pi_target_probs, width, label=f'Alvo (π) - Média Real: {true_mean_pi:.2f}', color='darkorange', alpha=0.8)
ax.bar(outcomes + width/2, b_behavior_probs, width, label=f'Comportamento (b) - Média Real: {true_mean_b:.2f}', color='steelblue', alpha=0.8)
ax.set_title('Distribuições de Probabilidade', fontsize=16)
ax.set_xlabel('Resultados Possíveis (x)')
ax.set_ylabel('Probabilidade P(x)')
ax.set_xticks(outcomes)
ax.legend()

# --- Gráfico 2: A Convergência das Estimativas ---
ax = axs[1]
ax.plot(naive_estimates, label='Estimativa com Média Ingênua', color='green')
ax.plot(is_estimates, label='Estimativa com Amostragem de Importância', color='red', linewidth=2)

ax.axhline(true_mean_pi, linestyle='--', color='darkorange', label=f'Alvo Real (Média de π): {true_mean_pi:.2f}')
ax.axhline(true_mean_b, linestyle='--', color='steelblue', label=f'Alvo Ingênuo (Média de b): {true_mean_b:.2f}')

ax.set_title('Convergência das Estimativas ao Longo do Tempo', fontsize=16)
ax.set_xlabel('Número de Amostras Tiradas de (b)')
ax.set_ylabel('Média Estimada')
ax.legend()
ax.set_ylim(0, 8) # Ajusta o eixo y para melhor visualização

plt.tight_layout()
plt.show()
