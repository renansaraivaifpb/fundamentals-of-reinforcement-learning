# ac_dqn_agent.py
# -*- coding: utf-8 -*-
"""
Agente Deep Q-Network (DQN) para Controle de Ar-Condicionado.

Este agente utiliza uma rede neural para aproximar a função Q, permitindo
o uso de um espaço de estados contínuo. Ele implementa conceitos chave como:
- Experience Replay para quebrar correlações entre amostras.
- Target Network para estabilizar o treinamento.

Autor: Gemini Mentor
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
from dataclasses import dataclass, asdict
import pickle
import logging

# Estrutura para armazenar transições
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Rede Neural para aproximar a função Q
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# Buffer de Replay para armazenar experiências
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

@dataclass
class DQNConfig:
    state_size: int = 4 # [temp, occupancy, sin(hour), cos(hour)]
    action_size: int = 4 # [OFF, LOW, MED, HIGH]
    episodes: int = 2000 # DQN geralmente converge mais rápido
    buffer_size: int = 100000
    batch_size: int = 64
    gamma: float = 0.99
    alpha: float = 1e-3 # Learning rate para o otimizador
    tau: float = 1e-3 # Para soft update da target network
    update_every: int = 4 # Com que frequência atualizar a rede
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01

class ACDQNAgent:
    """Agente que interage e aprende com o ambiente usando DQN."""
    
    def __init__(self, config: DQNConfig = None):
        self.config = config or DQNConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network_local = QNetwork(self.config.state_size, self.config.action_size).to(self.device)
        self.q_network_target = QNetwork(self.config.state_size, self.config.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=self.config.alpha)
        
        self.memory = ReplayBuffer(self.config.buffer_size)
        self.t_step = 0
        self.epsilon = self.config.epsilon_start
        self.training_history = {}
    def evaluate(self, env, episodes: int = 10):
        """Avalia a política aprendida do agente DQN sem exploração."""
        all_stats = {'episode_rewards': [], 'comfort_percentages': [], 'energy_consumptions': []}
        logging.info(f"Iniciando avaliação DQN por {episodes} episódios...")

        for _ in range(episodes):
            info = env.reset()[1]
            state = info['continuous_state']
            done = False
            episode_reward = 0
            
            while not done:
                action = self.choose_action(state, training=False) # Sem exploração
                _, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = info['continuous_state']
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
        logging.info(f"Avaliação DQN concluída: Recompensa Média = {final_stats['avg_reward']:.2f}")
        return final_stats
    
    def step(self, state, action, reward, next_state, done):
        """Salva a experiência no buffer e aciona o aprendizado."""
        self.memory.push(state, action, next_state, reward, done)
        
        self.t_step = (self.t_step + 1) % self.config.update_every
        if self.t_step == 0:
            if len(self.memory) > self.config.batch_size:
                experiences = self.memory.sample(self.config.batch_size)
                self.learn(experiences)

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Escolhe uma ação baseada no estado atual (epsilon-greedy)."""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state_tensor)
        self.q_network_local.train()
        
        if training and random.random() < self.epsilon:
            return random.choice(np.arange(self.config.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences):
        """Atualiza os pesos da rede neural usando um batch de experiências."""
        states, actions, next_states, rewards, dones = zip(*experiences)
        
        # Converte para tensores
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # Calcula Q-targets
        q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (self.config.gamma * q_targets_next * (1 - dones))
        
        # Calcula Q-expected
        q_expected = self.q_network_local(states).gather(1, actions)
        
        # Calcula o loss
        loss = nn.MSELoss()(q_expected, q_targets)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Atualiza a target network
        self.soft_update_target_network()

    def soft_update_target_network(self):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(self.q_network_target.parameters(), self.q_network_local.parameters()):
            target_param.data.copy_(self.config.tau * local_param.data + (1.0 - self.config.tau) * target_param.data)
            
    def train(self, env, verbose: bool = True):
        """Treina o agente DQN."""
        self.training_history = {
            'episode_rewards': [], 'comfort_percentages': [], 'energy_consumptions': [], 'epsilon_values': []
        }
        logging.info(f"Iniciando treinamento DQN para {self.config.episodes} episódios...")
        
        for episode in range(self.config.episodes):
            _, info = env.reset()
            state = info['continuous_state']
            episode_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state, training=True)
                _, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = info['continuous_state']
                
                self.step(state, np.array([action]), np.array([reward]), next_state, np.array([done]))
                
                state = next_state
                episode_reward += reward

            # Atualiza epsilon
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
            
            # Log
            stats = env.get_statistics()
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['comfort_percentages'].append(stats['comfort_percentage'])
            self.training_history['energy_consumptions'].append(stats['total_energy_consumption'])
            self.training_history['epsilon_values'].append(self.epsilon)

            if verbose and (episode + 1) % 100 == 0:
                avg_r = np.mean(self.training_history['episode_rewards'][-100:])
                logging.info(f"Episódio {episode+1}/{self.config.episodes} | Recompensa Média (100ep): {avg_r:.2f} | Epsilon: {self.epsilon:.3f}")
        
        logging.info("Treinamento DQN concluído.")
        return self.training_history

    def save_model(self, filepath: str):
        """Salva os pesos da rede e a configuração."""
        torch.save(self.q_network_local.state_dict(), filepath)
        logging.info(f"Modelo DQN salvo em: {filepath}")

    def load_model(self, filepath: str):
        """Carrega os pesos da rede."""
        self.q_network_local.load_state_dict(torch.load(filepath, map_location=self.device))
        self.q_network_target.load_state_dict(torch.load(filepath, map_location=self.device))
        logging.info(f"Modelo DQN carregado de: {filepath}")