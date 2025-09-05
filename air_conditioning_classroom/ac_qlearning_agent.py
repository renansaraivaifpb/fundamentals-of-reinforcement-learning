# -*- coding: utf-8 -*-
"""
Agente Q-Learning para Controle de Ar-Condicionado em Sala de Aula

Este módulo implementa um agente de aprendizagem por reforço usando Q-Learning
para controlar o sistema de ar-condicionado de uma sala de aula, otimizando
o balanceamento entre conforto térmico e eficiência energética.

Autor: Renan (com assistência de IA)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle
import json
from datetime import datetime

@dataclass
class QLearningConfig:
    """Configuração do agente Q-Learning"""
    alpha: float = 0.1          # Taxa de aprendizado
    gamma: float = 0.95         # Fator de desconto
    epsilon: float = 0.1        # Taxa de exploração inicial
    epsilon_min: float = 0.01   # Taxa de exploração mínima
    epsilon_decay: float = 0.995 # Decaimento da exploração
    episodes: int = 1000        # Número de episódios de treinamento
    
    # Parâmetros específicos para controle de AC
    temperature_penalty: float = 0.5    # Penalidade por temperatura inadequada
    energy_penalty: float = 0.1         # Penalidade por consumo energético
    comfort_bonus: float = 2.0          # Bônus por conforto térmico
    
    # Estratégias de exploração
    use_epsilon_decay: bool = True
    use_optimistic_init: bool = True
    optimistic_value: float = 1.0

class ACQLearningAgent:
    """
    Agente Q-Learning especializado para controle de ar-condicionado.
    
    Este agente aprende uma política ótima para controlar o ar-condicionado
    considerando múltiplos fatores:
    - Temperatura atual da sala
    - Número de ocupantes
    - Hora do dia
    - Estado atual do ar-condicionado
    """
    
    def __init__(self, n_states: int, n_actions: int, config: QLearningConfig = None):
        """
        Inicializa o agente Q-Learning.
        
        Args:
            n_states: Número de estados possíveis
            n_actions: Número de ações possíveis
            config: Configuração do agente
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config or QLearningConfig()
        
        # Inicializa tabela Q
        if self.config.use_optimistic_init:
            self.Q = np.full((n_states, n_actions), self.config.optimistic_value)
        else:
            self.Q = np.zeros((n_states, n_actions))
        
        # Parâmetros de aprendizado
        self.alpha = self.config.alpha
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon
        self.epsilon_min = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay
        
        # Histórico de treinamento
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_history': [],
            'q_table_changes': [],
            'comfort_percentages': [],
            'energy_consumptions': []
        }
        
        # Contadores para análise
        self.action_counts = np.zeros(n_actions)
        self.state_visits = np.zeros(n_states)
        
    def choose_action(self, state: int, training: bool = True) -> int:
        """
        Escolhe uma ação usando política ε-greedy.
        
        Args:
            state: Estado atual
            training: Se está em modo de treinamento
            
        Returns:
            Ação escolhida
        """
        self.state_visits[state] += 1
        
        if training and np.random.random() < self.epsilon:
            # Exploração: escolhe ação aleatória
            action = np.random.randint(self.n_actions)
        else:
            # Exploração: escolhe melhor ação conhecida
            action = np.argmax(self.Q[state])
        
        self.action_counts[action] += 1
        return action
    
    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool = False):
        """
        Atualiza a tabela Q usando a equação de Bellman.
        
        Args:
            state: Estado atual
            action: Ação executada
            reward: Recompensa recebida
            next_state: Próximo estado
            done: Se o episódio terminou
        """
        # Valor atual da ação
        current_q = self.Q[state, action]
        
        # Valor alvo (Bellman equation)
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.Q[next_state])
        
        # Atualiza Q-value
        self.Q[state, action] += self.alpha * (target_q - current_q)
    
    def update_epsilon(self):
        """Atualiza a taxa de exploração se habilitado"""
        if self.config.use_epsilon_decay:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env, episodes: int = None, verbose: bool = True) -> Dict:
        """
        Treina o agente no ambiente.
        
        Args:
            env: Ambiente de treinamento
            episodes: Número de episódios (usa config se None)
            verbose: Se deve imprimir progresso
            
        Returns:
            Histórico de treinamento
        """
        episodes = episodes or self.config.episodes
        
        if verbose:
            print(f"Iniciando treinamento por {episodes} episódios...")
            print(f"Configuração: α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")
        
        # Reset histórico
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_history': [],
            'q_table_changes': [],
            'comfort_percentages': [],
            'energy_consumptions': []
        }
        
        # Inicializa Q-table se necessário
        if self.config.use_optimistic_init:
            self.Q = np.full((self.n_states, self.n_actions), self.config.optimistic_value)
        else:
            self.Q = np.zeros((self.n_states, self.n_actions))
        
        # Reset parâmetros
        self.epsilon = self.config.epsilon
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            q_table_before = self.Q.copy()
            
            while True:
                # Escolhe ação
                action = self.choose_action(state, training=True)
                
                # Executa ação
                next_state, reward, done, info = env.step(action)
                
                # Aprende
                self.learn(state, action, reward, next_state, done)
                
                # Atualiza contadores
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Atualiza epsilon
            self.update_epsilon()
            
            # Calcula mudanças na Q-table
            q_table_change = np.mean(np.abs(self.Q - q_table_before))
            
            # Registra histórico
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['epsilon_history'].append(self.epsilon)
            self.training_history['q_table_changes'].append(q_table_change)
            
            # Estatísticas do ambiente
            stats = env.get_statistics()
            self.training_history['comfort_percentages'].append(stats['comfort_percentage'])
            self.training_history['energy_consumptions'].append(stats['total_energy_consumption'])
            
            # Progresso
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                avg_comfort = np.mean(self.training_history['comfort_percentages'][-100:])
                print(f"Episódio {episode + 1}: Recompensa média = {avg_reward:.2f}, "
                      f"Conforto = {avg_comfort:.1f}%, ε = {self.epsilon:.3f}")
        
        if verbose:
            print("Treinamento concluído!")
        
        return self.training_history
    
    def evaluate(self, env, episodes: int = 10, render: bool = False) -> Dict:
        """
        Avalia a política aprendida.
        
        Args:
            env: Ambiente de avaliação
            episodes: Número de episódios de avaliação
            render: Se deve renderizar o ambiente
            
        Returns:
            Estatísticas de avaliação
        """
        print(f"Avaliando política por {episodes} episódios...")
        
        evaluation_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'comfort_percentages': [],
            'energy_consumptions': [],
            'ac_usage_percentages': [],
            'temperature_stats': []
        }
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Escolhe melhor ação (sem exploração)
                action = self.choose_action(state, training=False)
                
                # Executa ação
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Coleta estatísticas
            stats = env.get_statistics()
            evaluation_stats['episode_rewards'].append(episode_reward)
            evaluation_stats['episode_lengths'].append(episode_length)
            evaluation_stats['comfort_percentages'].append(stats['comfort_percentage'])
            evaluation_stats['energy_consumptions'].append(stats['total_energy_consumption'])
            evaluation_stats['ac_usage_percentages'].append(stats['ac_usage_percentage'])
            evaluation_stats['temperature_stats'].append({
                'mean': stats['avg_temperature'],
                'std': stats['temp_std']
            })
            
            if render:
                env.render()
        
        # Calcula estatísticas agregadas
        final_stats = {
            'avg_reward': np.mean(evaluation_stats['episode_rewards']),
            'std_reward': np.std(evaluation_stats['episode_rewards']),
            'avg_comfort': np.mean(evaluation_stats['comfort_percentages']),
            'std_comfort': np.std(evaluation_stats['comfort_percentages']),
            'avg_energy': np.mean(evaluation_stats['energy_consumptions']),
            'std_energy': np.std(evaluation_stats['energy_consumptions']),
            'avg_ac_usage': np.mean(evaluation_stats['ac_usage_percentages']),
            'avg_temperature': np.mean([s['mean'] for s in evaluation_stats['temperature_stats']]),
            'temperature_std': np.mean([s['std'] for s in evaluation_stats['temperature_stats']])
        }
        
        print(f"Resultados da avaliação:")
        print(f"  Recompensa média: {final_stats['avg_reward']:.2f} ± {final_stats['std_reward']:.2f}")
        print(f"  Conforto médio: {final_stats['avg_comfort']:.1f}% ± {final_stats['std_comfort']:.1f}%")
        print(f"  Consumo energético médio: {final_stats['avg_energy']:.2f} ± {final_stats['std_energy']:.2f} kW")
        print(f"  Uso do AC: {final_stats['avg_ac_usage']:.1f}%")
        print(f"  Temperatura média: {final_stats['avg_temperature']:.1f}°C ± {final_stats['temperature_std']:.1f}°C")
        
        return final_stats
    
    def get_policy(self) -> np.ndarray:
        """Retorna a política aprendida (ação ótima para cada estado)"""
        return np.argmax(self.Q, axis=1)
    
    def get_state_values(self) -> np.ndarray:
        """Retorna os valores de estado (V(s) = max_a Q(s,a))"""
        return np.max(self.Q, axis=1)
    
    def analyze_policy(self, env) -> Dict:
        """
        Analisa a política aprendida em detalhes.
        
        Args:
            env: Ambiente para análise
            
        Returns:
            Análise detalhada da política
        """
        policy = self.get_policy()
        state_values = self.get_state_values()
        
        # Análise por tipo de estado
        analysis = {
            'policy_summary': {},
            'action_distribution': {},
            'state_value_stats': {
                'mean': np.mean(state_values),
                'std': np.std(state_values),
                'min': np.min(state_values),
                'max': np.max(state_values)
            },
            'q_table_stats': {
                'mean': np.mean(self.Q),
                'std': np.std(self.Q),
                'min': np.min(self.Q),
                'max': np.max(self.Q)
            }
        }
        
        # Distribuição de ações
        unique, counts = np.unique(policy, return_counts=True)
        for action, count in zip(unique, counts):
            analysis['action_distribution'][action] = {
                'count': count,
                'percentage': count / len(policy) * 100
            }
        
        # Análise por faixas de temperatura, ocupação, etc.
        # (implementação específica depende da estrutura do estado)
        
        return analysis
    
    def save_model(self, filepath: str):
        """Salva o modelo treinado"""
        model_data = {
            'Q': self.Q,
            'config': self.config,
            'training_history': self.training_history,
            'action_counts': self.action_counts,
            'state_visits': self.state_visits,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str):
        """Carrega um modelo treinado"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.Q = model_data['Q']
        self.config = model_data['config']
        self.training_history = model_data['training_history']
        self.action_counts = model_data['action_counts']
        self.state_visits = model_data['state_visits']
        
        print(f"Modelo carregado de: {filepath}")
    
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