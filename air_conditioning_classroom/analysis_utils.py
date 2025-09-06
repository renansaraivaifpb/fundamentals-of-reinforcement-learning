# -*- coding: utf-8 -*-
"""
Utilitários de Análise e Visualização para o Sistema de Controle de Ar-Condicionado

Este módulo contém funções para análise detalhada do comportamento do agente,
visualização de políticas aprendidas e métricas de performance.

Autor: Renan (com assistência de IA)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import json

from classroom_ac_env import ClassroomACEnvironment, ACState, ComfortLevel
from ac_qlearning_agent import ACQLearningAgent

def plot_q_table_heatmap(agent: ACQLearningAgent, env: ClassroomACEnvironment, 
                        save_path: str = None, figsize: Tuple[int, int] = (15, 10)):
    """
    Plota heatmap da tabela Q para análise visual da política aprendida.
    
    Args:
        agent: Agente treinado
        env: Ambiente de treinamento
        save_path: Caminho para salvar a figura
        figsize: Tamanho da figura
    """
    Q = agent.Q
    policy = agent.get_policy()
    
    # Cria figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Heatmap da Q-table (média por ação)
    q_mean = np.mean(Q, axis=1).reshape(-1, 1)
    sns.heatmap(q_mean, ax=axes[0, 0], cmap='viridis', cbar=True)
    axes[0, 0].set_title('Valores Q Médios por Estado')
    axes[0, 0].set_xlabel('Ação')
    axes[0, 0].set_ylabel('Estado')
    
    # 2. Heatmap da política (ação ótima por estado)
    policy_reshaped = policy.reshape(-1, 1)
    sns.heatmap(policy_reshaped, ax=axes[0, 1], cmap='Set1', cbar=True)
    axes[0, 1].set_title('Política Aprendida (Ação Ótima)')
    axes[0, 1].set_xlabel('Ação')
    axes[0, 1].set_ylabel('Estado')
    
    # 3. Distribuição de valores Q por ação
    action_names = ['OFF', 'LOW', 'MEDIUM', 'HIGH']
    q_by_action = [Q[:, i] for i in range(4)]
    
    axes[1, 0].boxplot(q_by_action, labels=action_names)
    axes[1, 0].set_title('Distribuição de Valores Q por Ação')
    axes[1, 0].set_ylabel('Valor Q')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Frequência de visitas por estado
    state_visits = agent.state_visits
    axes[1, 1].hist(state_visits, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribuição de Visitas por Estado')
    axes[1, 1].set_xlabel('Número de Visitas')
    axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_temperature_comfort_analysis(env: ClassroomACEnvironment, 
                                    agent: ACQLearningAgent,
                                    episodes: int = 5,
                                    save_path: str = None):
    """
    Analisa a relação entre temperatura e conforto ao longo do tempo.
    
    Args:
        env: Ambiente de simulação
        agent: Agente treinado
        episodes: Número de episódios para análise
        save_path: Caminho para salvar a figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    all_temps = []
    all_comforts = []
    all_ac_states = []
    all_occupancies = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_temps = []
        episode_comforts = []
        episode_ac_states = []
        episode_occupancies = []
        
        for step in range(100):  # 10 horas de simulação
            action = agent.choose_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            episode_temps.append(info['temperature'])
            episode_comforts.append(ComfortLevel[info['comfort_level']].value)
            episode_ac_states.append(action)
            episode_occupancies.append(info['occupancy'])
            
            state = next_state
            
            if done:
                break
        
        all_temps.extend(episode_temps)
        all_comforts.extend(episode_comforts)
        all_ac_states.extend(episode_ac_states)
        all_occupancies.extend(episode_occupancies)
    
    # 1. Temperatura vs Conforto
    comfort_colors = ['red', 'orange', 'green', 'orange', 'red']
    comfort_labels = ['Very Cold', 'Cold', 'Comfortable', 'Warm', 'Very Hot']
    
    for i in range(5):
        mask = np.array(all_comforts) == i
        if np.any(mask):
            axes[0, 0].scatter(np.array(all_temps)[mask], 
                             np.array(all_comforts)[mask], 
                             c=comfort_colors[i], alpha=0.6, 
                             label=comfort_labels[i])
    
    axes[0, 0].axvline(x=22, color='g', linestyle='--', alpha=0.7, label='Conforto Min')
    axes[0, 0].axvline(x=26, color='g', linestyle='--', alpha=0.7, label='Conforto Max')
    axes[0, 0].set_xlabel('Temperatura (°C)')
    axes[0, 0].set_ylabel('Nível de Conforto')
    axes[0, 0].set_title('Temperatura vs Conforto Térmico')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Estado do AC vs Temperatura
    ac_colors = ['blue', 'cyan', 'yellow', 'red']
    ac_labels = ['OFF', 'LOW', 'MEDIUM', 'HIGH']
    
    for i in range(4):
        mask = np.array(all_ac_states) == i
        if np.any(mask):
            axes[0, 1].scatter(np.array(all_temps)[mask], 
                             np.array(all_ac_states)[mask], 
                             c=ac_colors[i], alpha=0.6, 
                             label=ac_labels[i])
    
    axes[0, 1].axvline(x=22, color='g', linestyle='--', alpha=0.7, label='Conforto Min')
    axes[0, 1].axvline(x=26, color='g', linestyle='--', alpha=0.7, label='Conforto Max')
    axes[0, 1].set_xlabel('Temperatura (°C)')
    axes[0, 1].set_ylabel('Estado do AC')
    axes[0, 1].set_title('Estado do AC vs Temperatura')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Ocupação vs Temperatura
    axes[1, 0].scatter(all_occupancies, all_temps, alpha=0.6, c='purple')
    axes[1, 0].axhline(y=22, color='g', linestyle='--', alpha=0.7, label='Conforto Min')
    axes[1, 0].axhline(y=26, color='g', linestyle='--', alpha=0.7, label='Conforto Max')
    axes[1, 0].set_xlabel('Ocupação')
    axes[1, 0].set_ylabel('Temperatura (°C)')
    axes[1, 0].set_title('Ocupação vs Temperatura')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Distribuição de temperaturas
    axes[1, 1].hist(all_temps, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    axes[1, 1].axvline(x=22, color='g', linestyle='--', alpha=0.7, label='Conforto Min')
    axes[1, 1].axvline(x=26, color='g', linestyle='--', alpha=0.7, label='Conforto Max')
    axes[1, 1].axvline(x=np.mean(all_temps), color='r', linestyle='-', alpha=0.7, 
                      label=f'Média: {np.mean(all_temps):.1f}°C')
    axes[1, 1].set_xlabel('Temperatura (°C)')
    axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].set_title('Distribuição de Temperaturas')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_energy_efficiency(env: ClassroomACEnvironment, 
                            agent: ACQLearningAgent,
                            episodes: int = 10,
                            save_path: str = None):
    """
    Analisa a eficiência energética do agente.
    
    Args:
        env: Ambiente de simulação
        agent: Agente treinado
        episodes: Número de episódios para análise
        save_path: Caminho para salvar a figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    energy_data = []
    comfort_data = []
    ac_usage_data = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_energy = []
        episode_comfort = []
        episode_ac_usage = []
        
        for step in range(100):  # 10 horas de simulação
            action = agent.choose_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            episode_energy.append(info['energy_consumption'])
            episode_comfort.append(1 if info['comfort_level'] == 'COMFORTABLE' else 0)
            episode_ac_usage.append(1 if action != 0 else 0)  # 1 se AC ligado, 0 se desligado
            
            state = next_state
            
            if done:
                break
        
        energy_data.append(episode_energy)
        comfort_data.append(episode_comfort)
        ac_usage_data.append(episode_ac_usage)
    
    # 1. Consumo energético ao longo do tempo
    for i, episode_energy in enumerate(energy_data):
        axes[0, 0].plot(episode_energy, alpha=0.7, label=f'Episódio {i+1}' if i < 5 else '')
    
    axes[0, 0].set_title('Consumo Energético por Episódio')
    axes[0, 0].set_xlabel('Passos de Tempo')
    axes[0, 0].set_ylabel('Consumo (kW)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Eficiência: Conforto vs Consumo
    total_energy = [np.sum(ep) for ep in energy_data]
    total_comfort = [np.mean(ep) * 100 for ep in comfort_data]
    
    scatter = axes[0, 1].scatter(total_energy, total_comfort, 
                                c=range(len(total_energy)), 
                                cmap='viridis', s=100, alpha=0.7)
    axes[0, 1].set_xlabel('Consumo Energético Total (kW)')
    axes[0, 1].set_ylabel('Percentual de Conforto (%)')
    axes[0, 1].set_title('Eficiência: Conforto vs Consumo')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Adiciona linha de tendência
    z = np.polyfit(total_energy, total_comfort, 1)
    p = np.poly1d(z)
    axes[0, 1].plot(total_energy, p(total_energy), "r--", alpha=0.8)
    
    # 3. Uso do AC ao longo do tempo
    for i, episode_ac in enumerate(ac_usage_data):
        axes[1, 0].plot(episode_ac, alpha=0.7, label=f'Episódio {i+1}' if i < 5 else '')
    
    axes[1, 0].set_title('Uso do Ar-Condicionado por Episódio')
    axes[1, 0].set_xlabel('Passos de Tempo')
    axes[1, 0].set_ylabel('AC Ligado (1) / Desligado (0)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Estatísticas de eficiência
    avg_energy = np.mean(total_energy)
    avg_comfort = np.mean(total_comfort)
    efficiency_ratio = avg_comfort / avg_energy if avg_energy > 0 else 0
    
    stats_text = f"""
    Estatísticas de Eficiência:
    
    Consumo Energético Médio: {avg_energy:.2f} kW
    Conforto Médio: {avg_comfort:.1f}%
    Razão Eficiência: {efficiency_ratio:.2f} %/kW
    
    Desvio Padrão Energia: {np.std(total_energy):.2f} kW
    Desvio Padrão Conforto: {np.std(total_comfort):.1f}%
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                   fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def generate_policy_report(agent: ACQLearningAgent, env: ClassroomACEnvironment, 
                          save_path: str = None):
    """
    Gera relatório detalhado da política aprendida.
    
    Args:
        agent: Agente treinado
        env: Ambiente de treinamento
        save_path: Caminho para salvar o relatório
    """
    policy = agent.get_policy()
    state_values = agent.get_state_values()
    
    # Análise da política
    action_names = ['OFF', 'LOW', 'MEDIUM', 'HIGH']
    action_counts = np.bincount(policy, minlength=4)
    action_percentages = action_counts / len(policy) * 100
    
    # Análise por faixas de temperatura
    temp_ranges = [
        (15, 20, "Muito Frio"),
        (20, 22, "Frio"),
        (22, 26, "Confortável"),
        (26, 30, "Quente"),
        (30, 35, "Muito Quente")
    ]
    
    policy_by_temp = {}
    for temp_min, temp_max, label in temp_ranges:
        # Simula estados para diferentes temperaturas
        temp_states = []
        for temp in np.linspace(temp_min, temp_max, 10):
            for occ in [0, 15, 30]:  # Diferentes ocupações
                for hour in [8, 14, 20]:  # Diferentes horas
                    # Cria estado temporário para análise
                    env.current_temp = temp
                    env.occupancy = occ
                    env.hour_of_day = hour
                    state = env._discretize_state()
                    temp_states.append(state)
        
        if temp_states:
            actions_in_range = policy[temp_states]
            action_dist = np.bincount(actions_in_range, minlength=4)
            policy_by_temp[label] = action_dist / len(actions_in_range) * 100
    
    # Gera relatório
    report = f"""
# Relatório da Política Aprendida - Controle de Ar-Condicionado

## Resumo Geral
- Total de Estados: {len(policy)}
- Estados Visitados: {np.sum(agent.state_visits > 0)}
- Taxa de Exploração Final: {agent.epsilon:.3f}

## Distribuição de Ações
"""
    
    for i, (name, count, percentage) in enumerate(zip(action_names, action_counts, action_percentages)):
        report += f"- {name}: {count} estados ({percentage:.1f}%)\n"
    
    report += f"""
## Análise por Faixa de Temperatura
"""
    
    for temp_range, action_dist in policy_by_temp.items():
        report += f"\n### {temp_range}\n"
        for i, (name, percentage) in enumerate(zip(action_names, action_dist)):
            report += f"- {name}: {percentage:.1f}%\n"
    
    report += f"""
## Estatísticas da Q-Table
- Valor Médio: {np.mean(agent.Q):.3f}
- Desvio Padrão: {np.std(agent.Q):.3f}
- Valor Mínimo: {np.min(agent.Q):.3f}
- Valor Máximo: {np.max(agent.Q):.3f}

## Valores de Estado
- Valor Médio: {np.mean(state_values):.3f}
- Desvio Padrão: {np.std(state_values):.3f}
- Valor Mínimo: {np.min(state_values):.3f}
- Valor Máximo: {np.max(state_values):.3f}

## Análise de Convergência
- Mudança Média na Q-Table (últimos 100 episódios): {np.mean(agent.training_history['q_table_changes'][-100:]):.6f}
- Recompensa Média (últimos 100 episódios): {np.mean(agent.training_history['episode_rewards'][-100:]):.3f}

## Recomendações
"""
    
    # Análise de padrões
    if action_percentages[0] > 50:  # Muitos estados com AC desligado
        report += "- A política tende a manter o AC desligado, possivelmente priorizando eficiência energética.\n"
    
    if action_percentages[3] > 30:  # Muitos estados com AC alto
        report += "- A política usa frequentemente o AC em potência alta, indicando necessidade de refrigeração intensa.\n"
    
    if np.std(action_percentages) < 10:  # Distribuição uniforme
        report += "- A política é relativamente equilibrada entre as diferentes ações.\n"
    
    report += f"""
## Data de Geração
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Relatório salvo em: {save_path}")
    else:
        print(report)
    
    return report

def create_dashboard(agent: ACQLearningAgent, env: ClassroomACEnvironment, 
                    save_dir: str = "dashboard"):
    """
    Cria dashboard completo com todas as análises.
    
    Args:
        agent: Agente treinado
        env: Ambiente de treinamento
        save_dir: Diretório para salvar as figuras
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("Criando dashboard de análise...")
    
    # 1. Heatmap da Q-table
    print("1. Gerando heatmap da Q-table...")
    plot_q_table_heatmap(agent, env, os.path.join(save_dir, "q_table_heatmap.png"))
    
    # 2. Análise de temperatura e conforto
    print("2. Analisando temperatura e conforto...")
    plot_temperature_comfort_analysis(agent, env, episodes=5, 
                                    save_path=os.path.join(save_dir, "temperature_comfort_analysis.png"))
    
    # 3. Análise de eficiência energética
    print("3. Analisando eficiência energética...")
    analyze_energy_efficiency(agent, env, episodes=10, 
                            save_path=os.path.join(save_dir, "energy_efficiency_analysis.png"))
    
    # 4. Relatório da política
    print("4. Gerando relatório da política...")
    generate_policy_report(agent, env, os.path.join(save_dir, "policy_report.md"))
    
    # 5. Progresso de treinamento
    print("5. Plotando progresso de treinamento...")
    agent.plot_training_progress(os.path.join(save_dir, "training_progress.png"))
    
    print(f"Dashboard criado em: {save_dir}")

# Exemplo de uso
if __name__ == "__main__":
    from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig
    from ac_qlearning_agent import ACQLearningAgent, QLearningConfig
    
    # Cria ambiente e agente
    config = ClassroomConfig()
    env = ClassroomACEnvironment(config)
    
    agent_config = QLearningConfig(episodes=500)
    agent = ACQLearningAgent(env.n_states, env.n_actions, agent_config)
    
    # Treina agente
    print("Treinando agente...")
    agent.train(env, verbose=True)
    
    # Cria dashboard
    create_dashboard(agent, env)