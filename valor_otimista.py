# Bibiliotecas importadas
import numpy as np
import matplotlib.pyplot as plt

# Configuração do ambiente simulado (Bandit de 10 braços)
np.random.seed(42)
k = 10
q_true = np.random.normal(0, 1, k)  # valores reais das ações

# Parâmetros de simulação
n_episodes = 2000  # número de execuções independentes
n_steps = 1000     # número de passos por execução

# Função para simular um agente
def simulate_agent(initial_q, epsilon, label):
    avg_rewards = np.zeros(n_steps)
    optimal_action_counts = np.zeros(n_steps)

    for episode in range(n_episodes):
        q_est = np.ones(k) * initial_q
        action_counts = np.zeros(k)

        rewards = []
        optimal_action = np.argmax(q_true)

        for step in range(n_steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(k)
            else:
                action = np.argmax(q_est)

            reward = np.random.normal(q_true[action], 1)
            action_counts[action] += 1
            q_est[action] += (reward - q_est[action]) / action_counts[action]

            rewards.append(reward)
            if action == optimal_action:
                optimal_action_counts[step] += 1
            avg_rewards[step] += reward

    return avg_rewards / n_episodes, optimal_action_counts / n_episodes * 100

# Simulações
greedy_optimistic, optimal_optimistic = simulate_agent(initial_q=5.0, epsilon=0.0, label="Otimista")
epsilon_greedy, optimal_eps = simulate_agent(initial_q=0.0, epsilon=0.1, label="Epsilon-Greedy")

# Plot
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(greedy_optimistic, label="Otimista (Q=5, ε=0)")
plt.plot(epsilon_greedy, label="Epsilon-Greedy (Q=0, ε=0.1)")
plt.xlabel("Passos")
plt.ylabel("Recompensa média")
plt.legend()
plt.title("Recompensa média ao longo do tempo")

plt.subplot(1, 2, 2)
plt.plot(optimal_optimistic, label="Otimista (Q=5, ε=0)")
plt.plot(optimal_eps, label="Epsilon-Greedy (Q=0, ε=0.1)")
plt.xlabel("Passos")
plt.ylabel("% de ações ótimas")
plt.legend()
plt.title("Porcentagem de ações ótimas escolhidas")

plt.tight_layout()
plt.show()
