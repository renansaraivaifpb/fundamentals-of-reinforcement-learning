import numpy as np
import matplotlib.pyplot as plt

# Configurações do ambiente
np.random.seed(42)
n_actions = 10
n_steps = 1000
n_runs = 2000  # número de experimentos independentes

# Parâmetros
epsilon = 0.1
c = 2  # parâmetro de exploração do UCB

def run_bandit_ucb(epsilon=False):
    rewards = np.zeros((n_runs, n_steps))
    
    for run in range(n_runs):
        q_true = np.random.normal(0, 1, n_actions)  # valor verdadeiro das ações
        q_est = np.zeros(n_actions)
        action_counts = np.zeros(n_actions)
        
        for t in range(n_steps):
            if epsilon:
                # epsilon-greedy
                if np.random.rand() < 0.1:
                    action = np.random.randint(n_actions)
                else:
                    action = np.argmax(q_est)
            else:
                # UCB
                if t < n_actions:
                    action = t
                else:
                    ucb_values = q_est + c * np.sqrt(np.log(t + 1) / (action_counts + 1e-5))
                    action = np.argmax(ucb_values)

            reward = np.random.normal(q_true[action], 1)
            action_counts[action] += 1
            q_est[action] += (reward - q_est[action]) / action_counts[action]
            rewards[run, t] = reward

    return rewards.mean(axis=0)

# Simulações
avg_rewards_ucb = run_bandit_ucb(epsilon=False)
avg_rewards_eps = run_bandit_ucb(epsilon=True)

# Plotagem
plt.figure(figsize=(10, 6))
plt.plot(avg_rewards_ucb, label='UCB (c=2)', linewidth=2)
plt.plot(avg_rewards_eps, label='ε-greedy (ε=0.1)', linewidth=2)
plt.xlabel('Passo')
plt.ylabel('Recompensa média')
plt.title('Comparação entre UCB e ε-greedy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
