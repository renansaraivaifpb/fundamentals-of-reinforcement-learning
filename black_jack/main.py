# Autor: Renan Saraiva dos Santos

import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# (A classe MonteCarloExploringStartsAgent e as funções de plotagem permanecem as mesmas,
# apenas o método 'train' e a chamada principal serão ligeiramente modificados)

class MonteCarloExploringStartsAgent:
    def __init__(self, env, gamma=1.0):
        # ... (código do __init__ igual ao anterior) ...
        self.env = env; self.gamma = gamma; self.Q = defaultdict(float)
        self.returns = defaultdict(list); self.policy = defaultdict(lambda: self.env.action_space.sample())
        self.learning_progress_rewards = []

    def train(self, num_episodes=500000, monitored_states=None, print_every=50000):
        """
        MODIFICADO: Agora aceita uma lista de estados para monitorar.
        """
        if monitored_states is None:
            monitored_states = []

        for episode_num in tqdm(range(num_episodes)):
            episode = []
            state, info = self.env.reset()
            action = self.env.action_space.sample()
            done = False

            while not done:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode.append((state, action, reward))
                state = next_state
                if not done:
                    action = self.policy[state]

            self.learning_progress_rewards.append(reward)

            G = 0.0
            visited_pairs = set()
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                if (state, action) not in visited_pairs:
                    self.returns[(state, action)].append(G)
                    self.Q[(state, action)] = np.mean(self.returns[(state, action)])
                    q_values_for_state = [self.Q.get((state, a), 0.0) for a in range(self.env.action_space.n)]
                    self.policy[state] = np.argmax(q_values_for_state)
                    visited_pairs.add((state, action))

            # === BLOCO DE IMPRESSÃO MODIFICADO ===
            if (episode_num + 1) % print_every == 0:
                avg_reward = np.mean(self.learning_progress_rewards[-print_every:])
                print(f"\n--- Progresso no Episódio: {episode_num + 1}/{num_episodes} ---")
                print(f"  Recompensa Média (últimos {print_every} episódios): {avg_reward:.4f}")
                print(f"  Tamanho da Política (estados conhecidos): {len(self.policy)}")
                print("-" * 60)
                print("  Política Aprendida para Cenários Monitorados:")

                # Itera sobre a lista de estados para ver a política de cada um
                for s_track in monitored_states:
                    policy_s_int = self.policy.get(s_track, -1)
                    policy_s_str = "Hit" if policy_s_int == 1 else "Stick" if policy_s_int == 0 else "N/A (nunca visitado)"
                    print(f"    - Estado {str(s_track):<25}: Ação -> {policy_s_str}")
                print("-" * 60)

    def get_policy(self):
        return self.policy

# (As funções de plotagem - plot_learning_progress e plot_blackjack_policy - são as mesmas)
def plot_learning_progress(rewards, window_size=5000):
    plt.figure(figsize=(12, 6)); df = pd.DataFrame(rewards, columns=['reward'])
    moving_average = df['reward'].rolling(window=window_size, min_periods=1).mean()
    plt.plot(moving_average); plt.title(f'Progresso de Aprendizagem (Média Móvel de {window_size} Episódios)')
    plt.xlabel('Episódios'); plt.ylabel('Recompensa Média Móvel'); plt.grid(True); plt.ylim(-0.5, 0.1); plt.show()

def plot_blackjack_policy(policy):
    player_sums = range(12, 22); dealer_cards = range(1, 11)
    policy_no_ace = np.zeros((len(player_sums), len(dealer_cards))); policy_usable_ace = np.zeros((len(player_sums), len(dealer_cards)))
    for i, p_sum in enumerate(player_sums):
        for j, d_card in enumerate(dealer_cards):
            policy_no_ace[i, j] = policy.get((p_sum, d_card, False), 0)
            policy_usable_ace[i, j] = policy.get((p_sum, d_card, True), 0)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'title': 'Política Ótima Aprendida'})
    axs[0].set_title('Sem Ás Utilizável'); axs[1].set_title('Com Ás Utilizável')
    im1 = axs[0].imshow(policy_no_ace, origin='lower', extent=[0.5, 10.5, 11.5, 21.5], cmap='viridis')
    im2 = axs[1].imshow(policy_usable_ace, origin='lower', extent=[0.5, 10.5, 11.5, 21.5], cmap='viridis')
    for ax in axs:
        ax.set_xticks(range(1, 11), labels=['A', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        ax.set_yticks(range(12, 22)); ax.set_xlabel("Carta Visível do Dealer"); ax.set_ylabel("Soma do Jogador")
    fig.colorbar(im1, ax=axs, ticks=[0,1], label="Ação (0=Stick, 1=Hit)")
    plt.show()

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    env = gym.make('Blackjack-v1')
    agent = MonteCarloExploringStartsAgent(env)

    # MODIFICADO: Lista de estados que queremos observar
    CENARIOS_PARA_MONITORAR = [
        (20, 10, False), # Mão forte
        (12, 3, False),  # Mão fraca
        (16, 10, False), # Decisão difícil
        (13, 2, False),  # Decisão difícil
        (17, 9, False),  # Decisão na margem
        (19, 10, True),  # Mão "soft" forte (Ás+8)
        (17, 6, True),   # Mão "soft" 17 (Ás+6)
        (15, 4, True),   # Mão "soft" 15 (Ás+4)
        (13, 10, True),  # Mão "soft" 13 (Ás+2)
        (21, 1, False),  # Mão perfeita
    ]

    agent.train(num_episodes=500000, monitored_states=CENARIOS_PARA_MONITORAR)

    print("\n--- Treinamento Concluído. Gerando gráficos finais... ---")
    plot_learning_progress(agent.learning_progress_rewards)
    final_policy = agent.get_policy()
    plot_blackjack_policy(final_policy)
    env.close()
