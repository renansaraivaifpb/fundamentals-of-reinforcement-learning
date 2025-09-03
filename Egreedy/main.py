# Autor: Renan Saraiva dos Santos

# importando bibliotecas
import numpy as np
import matplotlib.pyplot as plt

# Configurações do ambiente
n_actions = 10
n_steps = 1000
n_runs = 2000
epsilons = [0, 0.01, 0.1]

# Função para executar uma simulação do agente epsilon-greedy
def run_bandit(epsilon, n_runs, n_steps, n_actions):
    avg_rewards = np.zeros(n_steps)

    for run in range(n_runs):
        q_true = np.random.normal(0, 1, n_actions)  # valor real de cada ação (q*)
        q_est = np.zeros(n_actions)  # estimativas iniciais
        counts = np.zeros(n_actions)  # contador de seleções
        rewards = []

        for step in range(n_steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(q_est)

            reward = np.random.normal(q_true[action], 1)
            rewards.append(reward)

            counts[action] += 1
            alpha = 1 / counts[action]
            q_est[action] += alpha * (reward - q_est[action])

        avg_rewards += np.array(rewards)

    avg_rewards /= n_runs
    return avg_rewards

# Executar simulações para diferentes valores de epsilon
results = {epsilon: run_bandit(epsilon, n_runs, n_steps, n_actions) for epsilon in epsilons}

# Plot dos resultados
plt.figure(figsize=(12, 6))
for epsilon, rewards in results.items():
    label = f"ε = {epsilon}"
    plt.plot(rewards, label=label)

plt.title("Recompensa média por passo - Epsilon-Greedy (10-arm bandit)")
plt.xlabel("Etapas")
plt.ylabel("Recompensa média")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Egreedy/rewards_plot.png')  # Salva antes do show
plt.show()
