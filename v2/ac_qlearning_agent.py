# ac_qlearning_agent.py
# -*- coding: utf-8 -*-
"""
Agente Q-Learning Aprimorado para Controle de Ar-Condicionado. (Versão Final Corrigida)
"""

import numpy as np
from dataclasses import dataclass, asdict
import pickle
import logging
from collections import deque
import matplotlib.pyplot as plt

@dataclass
class QLearningConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995
    episodes: int = 10000
    early_stopping_patience: int = 500
    early_stopping_threshold: float = 0.01

class ACQLearningAgent:
    """Agente Q-Learning tabular."""
    
    def __init__(self, n_states: int, n_actions: int, config: QLearningConfig = None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config or QLearningConfig()
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.epsilon = self.config.epsilon
        self.training_history = {}

    def choose_action(self, state: int, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool):
        current_q = self.Q[state, action]
        target_q = reward if done else reward + self.config.gamma * np.max(self.Q[next_state])
        new_q = current_q + self.config.alpha * (target_q - current_q)
        self.Q[state, action] = np.clip(new_q, -100, 100)

    def train(self, env, verbose: bool = True):
        self.training_history = {
            'episode_rewards': [], 'comfort_percentages': [], 'energy_consumptions': [], 'epsilon_values': []
        }
        rewards_window = deque(maxlen=self.config.early_stopping_patience)
        best_avg_reward = -np.inf
        patience_counter = 0

        logging.info(f"Iniciando treinamento Q-Learning para {self.config.episodes} episódios...")
        
        for episode in range(self.config.episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.learn(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
            
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
            
            stats = env.get_statistics()
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['comfort_percentages'].append(stats['comfort_percentage'])
            self.training_history['energy_consumptions'].append(stats['total_energy_consumption'])
            self.training_history['epsilon_values'].append(self.epsilon)

            rewards_window.append(episode_reward)
            if len(rewards_window) == self.config.early_stopping_patience:
                current_avg_reward = np.mean(rewards_window)
                if current_avg_reward > best_avg_reward + self.config.early_stopping_threshold:
                    best_avg_reward = current_avg_reward
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    logging.info(f"Early stopping ativado no episódio {episode+1}.")
                    break
            
            if verbose and (episode + 1) % 500 == 0:
                avg_r = np.mean(self.training_history['episode_rewards'][-100:])
                logging.info(f"Episódio {episode+1}/{self.config.episodes} | Recompensa Média (100ep): {avg_r:.2f} | Epsilon: {self.epsilon:.3f}")

        logging.info("Treinamento Q-Learning concluído.")
        return self.training_history

    def evaluate(self, env, episodes: int = 10):
        all_stats = {'episode_rewards': [], 'comfort_percentages': [], 'energy_consumptions': []}
        logging.info(f"Iniciando avaliação Q-Learning por {episodes} episódios...")
        for _ in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.choose_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                episode_reward += reward
            stats = env.get_statistics()
            all_stats['episode_rewards'].append(episode_reward)
            all_stats['comfort_percentages'].append(stats['comfort_percentage'])
            all_stats['energy_consumptions'].append(stats['total_energy_consumption'])
        final_stats = {
            'avg_reward': np.mean(all_stats['episode_rewards']),
            'avg_comfort': np.mean(all_stats['comfort_percentages']),
            'avg_energy': np.mean(all_stats['energy_consumptions'])
        }
        logging.info(f"Avaliação Q-Learning concluída: Recompensa Média = {final_stats['avg_reward']:.2f}")
        return final_stats

    def save_model(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'q_table': self.Q, 'config': asdict(self.config)}, f)
        logging.info(f"Modelo Q-Learning salvo em: {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.Q = data['q_table']
        self.config = QLearningConfig(**data['config'])
        self.n_states, self.n_actions = self.Q.shape
        logging.info(f"Modelo Q-Learning carregado de: {filepath}")
    
    def plot_training_progress(self, save_path: str = None):
        """Plota o progresso do treinamento"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Recompensas por episódio
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_title('Recompensas por Episódio')
        axes[0, 0].set_xlabel('Episódio')
        axes[0, 0].set_ylabel('Recompensa Total')
        axes[0, 0].grid(True)
        
        # Média móvel das recompensas
        window = 100
        if len(self.training_history['episode_rewards']) >= window:
            moving_avg = np.convolve(self.training_history['episode_rewards'], 
                                   np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.training_history['episode_rewards'])), 
                           moving_avg, 'r-', linewidth=2, label=f'Média Móvel ({window})')
            axes[0, 0].legend()
        
        # Taxa de exploração
        axes[0, 1].plot(self.training_history['epsilon_history'])
        axes[0, 1].set_title('Taxa de Exploração (ε)')
        axes[0, 1].set_xlabel('Episódio')
        axes[0, 1].set_ylabel('ε')
        axes[0, 1].grid(True)
        
        # Mudanças na Q-table
        axes[0, 2].plot(self.training_history['q_table_changes'])
        axes[0, 2].set_title('Mudanças na Q-Table')
        axes[0, 2].set_xlabel('Episódio')
        axes[0, 2].set_ylabel('Mudança Média')
        axes[0, 2].grid(True)
        
        # Percentual de conforto
        axes[1, 0].plot(self.training_history['comfort_percentages'])
        axes[1, 0].set_title('Percentual de Conforto Térmico')
        axes[1, 0].set_xlabel('Episódio')
        axes[1, 0].set_ylabel('Conforto (%)')
        axes[1, 0].grid(True)
        
        # Consumo energético
        axes[1, 1].plot(self.training_history['energy_consumptions'])
        axes[1, 1].set_title('Consumo Energético Total')
        axes[1, 1].set_xlabel('Episódio')
        axes[1, 1].set_ylabel('Energia (kW)')
        axes[1, 1].grid(True)
        
        # Distribuição de ações
        action_names = ['OFF', 'LOW', 'MEDIUM', 'HIGH']
        action_counts = [self.action_counts[i] for i in range(len(action_names))]
        axes[1, 2].bar(action_names, action_counts)
        axes[1, 2].set_title('Distribuição de Ações')
        axes[1, 2].set_xlabel('Ação')
        axes[1, 2].set_ylabel('Frequência')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig
    
    # Cria ambiente e agente
    config = ClassroomConfig()
    env = ClassroomACEnvironment(config)
    
    agent_config = QLearningConfig(
        alpha=0.1,
        gamma=0.95,
        epsilon=0.2,
        episodes=500
    )
    
    agent = ACQLearningAgent(env.n_states, env.n_actions, agent_config)
    
    # Treina o agente
    print("Treinando agente...")
    history = agent.train(env, verbose=True)
    
    # Avalia a política
    print("\nAvaliando política...")
    eval_stats = agent.evaluate(env, episodes=5, render=True)
    
    # Plota progresso
    agent.plot_training_progress()
    
    # Salva modelo
    agent.save_model("ac_agent_model.pkl")