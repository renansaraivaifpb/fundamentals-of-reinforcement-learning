import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Ambiente contínuo estilo Gym (Termostato)
# --------------------------------------------------
class ThermostatEnv:
    """
    Ambiente simplificado de termostato.

    Estado:
        0 = confortável
        1 = desconfortável (alguém reclamou)

    Ações:
        0 = aquecedor off
        1 = aquecedor on

    Recompensas:
        0  (ninguém reclamou)
       -1 (alguém reclamou)
    """
    def __init__(self, p_noise=0.2):
        self.p = p_noise  # Probabilidade de ruído: alguém reclama aleatoriamente
        self.state = 0

    def reset(self):
        """Reseta o ambiente para o estado inicial confortável"""
        self.state = 0
        return self.state

    def step(self, action):
        """
        Executa uma ação no ambiente.

        Args:
            action (int): 0 = off, 1 = on
        
        Returns:
            state (int): novo estado
            reward (int): recompensa obtida
            done (bool): flag de término (sempre False, ambiente contínuo)
            info (dict): informações adicionais (vazio)
        """
        # Aleatoriamente, alguém pode reclamar (ruído)
        if np.random.rand() < self.p:
            reward = -1
            self.state = 1
        else:
            reward = 0
            self.state = 0

        done = False  # tarefa contínua
        return self.state, reward, done, {}

# --------------------------------------------------
# Agente Q-Learning
# --------------------------------------------------
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Args:
            n_states (int): número de estados possíveis
            n_actions (int): número de ações possíveis
            alpha (float): taxa de aprendizado
            gamma (float): fator de desconto
            epsilon (float): taxa de exploração
        """
        self.Q = np.zeros((n_states, n_actions))  # inicializa a tabela Q
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        """Escolhe uma ação usando política ε-greedy"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)  # explora
        else:
            return np.argmax(self.Q[state])  # explora conhecimento

    def learn(self, s, a, r, s_):
        """Atualiza a tabela Q usando o Q-Learning"""
        predict = self.Q[s, a]
        target = r + self.gamma * np.max(self.Q[s_])
        self.Q[s, a] += self.alpha * (target - predict)

# --------------------------------------------------
# Treinamento comparando dois valores de γ
# --------------------------------------------------
def run_experiment(gamma, env, episodes=5000):
    """
    Executa um experimento de aprendizado Q-Learning.

    Args:
        gamma (float): fator de desconto
        env (ThermostatEnv): ambiente
        episodes (int): número de episódios

    Returns:
        avg_rewards (list): recompensa média por episódio
    """
    agent = QLearningAgent(n_states=2, n_actions=2, gamma=gamma, alpha=0.1, epsilon=0.1)
    avg_rewards = []
    total_reward = 0
    s = env.reset()

    for ep in range(episodes):
        a = agent.choose_action(s)
        s_, r, d, _ = env.step(a)
        agent.learn(s, a, r, s_)
        s = s_
        total_reward += r
        avg_rewards.append(total_reward / (ep+1))
    return avg_rewards

# --------------------------------------------------
# Executar experimentos
# --------------------------------------------------
env = ThermostatEnv(p_noise=0.2)
episodes = 5000
gammas = [0.3, 0.99]  # comparação de fator de desconto
results = {}

for g in gammas:
    results[g] = run_experiment(g, env, episodes)

# --------------------------------------------------
# Plot dos resultados
# --------------------------------------------------
plt.figure(figsize=(10,5))
for g in gammas:
    plt.plot(results[g], label=f"γ = {g}")
plt.title("Aprendizado Q-Learning no Termostato (tarefa contínua)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa média")
plt.legend()
plt.grid(True)
plt.show()
