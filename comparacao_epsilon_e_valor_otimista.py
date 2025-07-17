import numpy as np
import matplotlib.pyplot as plt

# Configuração do ambiente não-estacionário
np.random.seed(42)
k = 10
n_episodes = 2000
n_steps = 1000
q_true_initial = np.random.normal(0, 1, k)

def simulate_nonstationary_agent(initial_q, epsilon, alpha=None, label=""):
    avg_rewards = np.zeros(n_steps)
    optimal_action_counts = np.zeros(n_steps)

    for episode in range(n_episodes):
        q_true = np.copy(q_true_initial)
        q_est = np.ones(k) * initial_q
        action_counts = np.zeros(k)

        optimal_action = np.argmax(q_true)

        for step in range(n_steps):
            # Variação não-estacionária: cada valor real muda um pouco
            q_true += np.random.normal(0, 0.01, k)

            if np.random.rand() < epsilon:
                action = np.random.randint(k)
            else:
                action = np.argmax(q_est)

            reward = np.random.normal(q_true[action], 1)
            action_counts[action] += 1

            if alpha is None:
                # Média amostral
                q_est[action] += (reward - q_est[action]) / action_counts[action]
            else:
                # Atualização exponencial com passo constante (melhor para ambientes não-estacionários)
                q_est[action] += alpha * (reward - q_est[action])

            avg_rewards[step] += reward
            if action == np.argmax(q_true):
                optimal_action_counts[step] += 1

    return avg_rewards / n_episodes, optimal_action_counts / n_episodes * 100

# Simulações para ambiente não-estacionário
epsilon_greedy_ns, optimal_eps_ns = simulate_nonstationary_agent(initial_q=0.0, epsilon=0.1, alpha=0.1, label="Epsilon-Greedy")
optimistic_ns, optimal_opt_ns = simulate_nonstationary_agent(initial_q=5.0, epsilon=0.0, alpha=None, label="Otimista")

# Plot
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epsilon_greedy_ns, label="Epsilon-Greedy (α=0.1)")
plt.plot(optimistic_ns, label="Otimista (α por contagem)")
plt.xlabel("Passos")
plt.ylabel("Recompensa média")
plt.legend()
plt.title("Recompensa média - Ambiente não-estacionário")

plt.subplot(1, 2, 2)
plt.plot(optimal_eps_ns, label="Epsilon-Greedy (α=0.1)")
plt.plot(optimal_opt_ns, label="Otimista (α por contagem)")
plt.xlabel("Passos")
plt.ylabel("% de ações ótimas")
plt.legend()
plt.title("Ações ótimas escolhidas - Ambiente não-estacionário")

plt.tight_layout()
plt.show()
