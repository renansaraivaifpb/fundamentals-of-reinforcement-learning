# -*- coding: utf-8 -*-
"""
Script Principal para Treinamento e Avaliação do Agente de Controle de Ar-Condicionado

Este script orquestra o treinamento de um agente Q-Learning para controlar
o sistema de ar-condicionado de uma sala de aula, incluindo:
- Treinamento do agente
- Avaliação da política aprendida
- Análise de resultados
- Visualizações

Autor: Renan (com assistência de IA)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json

from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig, ACState, ComfortLevel
from ac_qlearning_agent import ACQLearningAgent, QLearningConfig

def create_experiment_configs():
    """Cria diferentes configurações de experimento para comparação"""
    configs = {
        'baseline': {
            'env_config': ClassroomConfig(),
            'agent_config': QLearningConfig(
                alpha=0.1,
                gamma=0.95,
                epsilon=0.2,
                episodes=1000
            ),
            'name': 'Baseline'
        },
        'high_exploration': {
            'env_config': ClassroomConfig(),
            'agent_config': QLearningConfig(
                alpha=0.1,
                gamma=0.95,
                epsilon=0.5,
                epsilon_decay=0.99,
                episodes=1000
            ),
            'name': 'High Exploration'
        },
        'low_learning_rate': {
            'env_config': ClassroomConfig(),
            'agent_config': QLearningConfig(
                alpha=0.05,
                gamma=0.95,
                epsilon=0.2,
                episodes=1000
            ),
            'name': 'Low Learning Rate'
        },
        'high_discount': {
            'env_config': ClassroomConfig(),
            'agent_config': QLearningConfig(
                alpha=0.1,
                gamma=0.99,
                epsilon=0.2,
                episodes=1000
            ),
            'name': 'High Discount Factor'
        }
    }
    return configs

def run_experiment(env_config, agent_config, name, save_dir):
    """Executa um experimento completo"""
    print(f"\n{'='*50}")
    print(f"Executando experimento: {name}")
    print(f"{'='*50}")
    
    # Cria ambiente e agente
    env = ClassroomACEnvironment(env_config)
    agent = ACQLearningAgent(env.n_states, env.n_actions, agent_config)
    
    # Treina o agente
    print("Treinando agente...")
    start_time = datetime.now()
    history = agent.train(env, verbose=True)
    training_time = datetime.now() - start_time
    
    # Avalia a política
    print("\nAvaliando política...")
    eval_stats = agent.evaluate(env, episodes=10, render=False)
    
    # Analisa a política
    policy_analysis = agent.analyze_policy(env)
    
    # Salva resultados
    results = {
        'name': name,
        'config': {
            'env': {
                'length': env_config.length,
                'width': env_config.width,
                'height': env_config.height,
                'max_occupancy': env_config.max_occupancy,
                'temp_comfort_min': env_config.temp_comfort_min,
                'temp_comfort_max': env_config.temp_comfort_max
            },
            'agent': {
                'alpha': agent_config.alpha,
                'gamma': agent_config.gamma,
                'epsilon': agent_config.epsilon,
                'episodes': agent_config.episodes
            }
        },
        'training_time': str(training_time),
        'training_history': history,
        'evaluation_stats': eval_stats,
        'policy_analysis': policy_analysis,
        'timestamp': datetime.now().isoformat()
    }
    
    # Salva modelo e resultados
    model_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
    results_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_results.json")
    
    agent.save_model(model_path)
    
    # Converte numpy arrays para listas para JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Salva resultados em JSON
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert_numpy)
    
    print(f"Resultados salvos em: {results_path}")
    print(f"Modelo salvo em: {model_path}")
    
    return results

def plot_comparison(results_list, save_dir):
    """Plota comparação entre diferentes experimentos"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Cores para diferentes experimentos
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_list)))
    
    # 1. Recompensas por episódio
    for i, results in enumerate(results_list):
        rewards = results['training_history']['episode_rewards']
        axes[0, 0].plot(rewards, alpha=0.7, color=colors[i], label=results['name'])
    
    axes[0, 0].set_title('Recompensas por Episódio')
    axes[0, 0].set_xlabel('Episódio')
    axes[0, 0].set_ylabel('Recompensa Total')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Média móvel das recompensas
    window = 100
    for i, results in enumerate(results_list):
        rewards = results['training_history']['episode_rewards']
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(rewards)), moving_avg, 
                           color=colors[i], linewidth=2, label=results['name'])
    
    axes[0, 1].set_title(f'Recompensas - Média Móvel ({window})')
    axes[0, 1].set_xlabel('Episódio')
    axes[0, 1].set_ylabel('Recompensa Média')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Percentual de conforto
    for i, results in enumerate(results_list):
        comfort = results['training_history']['comfort_percentages']
        axes[0, 2].plot(comfort, alpha=0.7, color=colors[i], label=results['name'])
    
    axes[0, 2].set_title('Conforto Térmico por Episódio')
    axes[0, 2].set_xlabel('Episódio')
    axes[0, 2].set_ylabel('Conforto (%)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # 4. Consumo energético
    for i, results in enumerate(results_list):
        energy = results['training_history']['energy_consumptions']
        axes[1, 0].plot(energy, alpha=0.7, color=colors[i], label=results['name'])
    
    axes[1, 0].set_title('Consumo Energético por Episódio')
    axes[1, 0].set_xlabel('Episódio')
    axes[1, 0].set_ylabel('Energia (kW)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 5. Comparação de métricas finais
    names = [r['name'] for r in results_list]
    final_rewards = [r['evaluation_stats']['avg_reward'] for r in results_list]
    final_comfort = [r['evaluation_stats']['avg_comfort'] for r in results_list]
    final_energy = [r['evaluation_stats']['avg_energy'] for r in results_list]
    
    x = np.arange(len(names))
    width = 0.25
    
    axes[1, 1].bar(x - width, final_rewards, width, label='Recompensa', alpha=0.8)
    axes[1, 1].set_xlabel('Experimento')
    axes[1, 1].set_ylabel('Recompensa Média')
    axes[1, 1].set_title('Métricas Finais - Recompensa')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Comparação de conforto e energia
    ax2 = axes[1, 2]
    ax3 = ax2.twinx()
    
    bars1 = ax2.bar(x - width/2, final_comfort, width, label='Conforto (%)', alpha=0.8, color='green')
    bars2 = ax3.bar(x + width/2, final_energy, width, label='Energia (kW)', alpha=0.8, color='red')
    
    ax2.set_xlabel('Experimento')
    ax2.set_ylabel('Conforto (%)', color='green')
    ax3.set_ylabel('Energia (kW)', color='red')
    ax2.set_title('Conforto vs Consumo Energético')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45)
    
    # Adiciona legendas
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Salva figura
    comparison_path = os.path.join(save_dir, 'experiment_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparação salva em: {comparison_path}")

def analyze_policy_behavior(agent, env, save_dir):
    """Analisa o comportamento da política aprendida"""
    print("\nAnalisando comportamento da política...")
    
    # Testa diferentes cenários
    scenarios = [
        {'name': 'Sala Vazia (Manhã)', 'occupancy': 0, 'hour': 8},
        {'name': 'Sala Cheia (Manhã)', 'occupancy': 30, 'hour': 8},
        {'name': 'Sala Vazia (Tarde)', 'occupancy': 0, 'hour': 14},
        {'name': 'Sala Cheia (Tarde)', 'occupancy': 30, 'hour': 14},
        {'name': 'Sala Vazia (Noite)', 'occupancy': 0, 'hour': 20},
        {'name': 'Sala Cheia (Noite)', 'occupancy': 30, 'hour': 20}
    ]
    
    scenario_results = []
    
    for scenario in scenarios:
        print(f"\nTestando cenário: {scenario['name']}")
        
        # Configura cenário específico
        env.current_temp = 24.0
        env.occupancy = scenario['occupancy']
        env.hour_of_day = scenario['hour']
        
        # Executa episódio
        state = env.reset()
        episode_data = {
            'scenario': scenario['name'],
            'temperatures': [],
            'ac_states': [],
            'comfort_levels': [],
            'rewards': []
        }
        
        for step in range(100):  # 10 horas de simulação
            action = agent.choose_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            episode_data['temperatures'].append(info['temperature'])
            episode_data['ac_states'].append(action)
            episode_data['comfort_levels'].append(info['comfort_level'])
            episode_data['rewards'].append(reward)
            
            state = next_state
            
            if done:
                break
        
        scenario_results.append(episode_data)
        
        # Estatísticas do cenário
        stats = env.get_statistics()
        print(f"  Temperatura média: {stats['avg_temperature']:.1f}°C")
        print(f"  Conforto: {stats['comfort_percentage']:.1f}%")
        print(f"  Uso do AC: {stats['ac_usage_percentage']:.1f}%")
        print(f"  Consumo energético: {stats['total_energy_consumption']:.1f} kW")
    
    # Plota análise de cenários
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, result in enumerate(scenario_results):
        row = i // 3
        col = i % 3
        
        # Temperatura
        axes[row, col].plot(result['temperatures'], label='Temperatura', color='blue')
        axes[row, col].axhline(y=22, color='g', linestyle='--', alpha=0.7, label='Conforto Min')
        axes[row, col].axhline(y=26, color='g', linestyle='--', alpha=0.7, label='Conforto Max')
        
        # Estado do AC (eixo secundário)
        ax2 = axes[row, col].twinx()
        ac_states = np.array(result['ac_states'])
        ax2.plot(ac_states, label='Estado AC', color='red', alpha=0.7)
        ax2.set_ylabel('Estado AC (0=OFF, 1=LOW, 2=MED, 3=HIGH)', color='red')
        ax2.set_ylim(-0.5, 3.5)
        
        axes[row, col].set_title(result['scenario'])
        axes[row, col].set_xlabel('Passos de Tempo')
        axes[row, col].set_ylabel('Temperatura (°C)', color='blue')
        axes[row, col].legend(loc='upper left')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva análise
    analysis_path = os.path.join(save_dir, 'policy_behavior_analysis.png')
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Análise de comportamento salva em: {analysis_path}")
    
    return scenario_results

def main():
    """Função principal"""
    print("Sistema de Controle de Ar-Condicionado com Aprendizagem por Reforço")
    print("="*70)
    
    # Cria diretório para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Resultados serão salvos em: {save_dir}")
    
    # Configurações de experimento
    experiment_configs = create_experiment_configs()
    
    # Executa experimentos
    results_list = []
    
    for config_name, config in experiment_configs.items():
        try:
            results = run_experiment(
                config['env_config'],
                config['agent_config'],
                config['name'],
                save_dir
            )
            results_list.append(results)
        except Exception as e:
            print(f"Erro no experimento {config_name}: {e}")
            continue
    
    # Plota comparação
    if len(results_list) > 1:
        print("\nGerando comparação entre experimentos...")
        plot_comparison(results_list, save_dir)
    
    # Análise detalhada do melhor experimento
    if results_list:
        best_result = max(results_list, key=lambda x: x['evaluation_stats']['avg_reward'])
        print(f"\nMelhor experimento: {best_result['name']}")
        print(f"Recompensa média: {best_result['evaluation_stats']['avg_reward']:.2f}")
        print(f"Conforto médio: {best_result['evaluation_stats']['avg_comfort']:.1f}%")
        
        # Carrega o melhor modelo para análise
        from ac_qlearning_agent import ACQLearningAgent
        best_agent = ACQLearningAgent(0, 0)  # Valores temporários
        model_path = os.path.join(save_dir, f"{best_result['name'].lower().replace(' ', '_')}_model.pkl")
        best_agent.load_model(model_path)
        
        # Cria ambiente para análise
        best_env = ClassroomACEnvironment(best_result['config']['env'])
        
        # Analisa comportamento da política
        analyze_policy_behavior(best_agent, best_env, save_dir)
    
    print(f"\nExperimentos concluídos! Resultados salvos em: {save_dir}")

if __name__ == "__main__":
    main()