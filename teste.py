# -*- coding: utf-8 -*-
"""
Script Principal para Treinamento e Avaliação do Agente de Controle de Ar-Condicionado
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
from enum import Enum
from dataclasses import asdict, is_dataclass

# Certifique-se de que os nomes dos arquivos importados correspondem aos seus
from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig, ACState, ComfortLevel
from ac_qlearning_agent import ACQLearningAgent, QLearningConfig


def create_experiment_configs():
    """Cria diferentes configurações de experimento para comparação"""
    configs = {
        'baseline': {
            'env_config': ClassroomConfig(),
            'agent_config': QLearningConfig(alpha=0.1, gamma=0.95, epsilon=0.2, episodes=1000),
            'name': 'Baseline'
        },
        'high_exploration': {
            'env_config': ClassroomConfig(),
            'agent_config': QLearningConfig(alpha=0.1, gamma=0.95, epsilon=0.5, epsilon_decay=0.99, episodes=1000),
            'name': 'High Exploration'
        },
        'low_learning_rate': {
            'env_config': ClassroomConfig(),
            'agent_config': QLearningConfig(alpha=0.05, gamma=0.95, epsilon=0.2, episodes=1000),
            'name': 'Low Learning Rate'
        },
        'high_discount': {
            'env_config': ClassroomConfig(),
            'agent_config': QLearningConfig(alpha=0.1, gamma=0.99, epsilon=0.2, episodes=1000),
            'name': 'High Discount Factor'
        }
    }
    return configs


class RobustJSONEncoder(json.JSONEncoder):
    """
    Um encoder JSON customizado para lidar com tipos complexos
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Enum):
            return obj.name
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (set, tuple)):
            return list(obj)
        return super(RobustJSONEncoder, self).default(obj)


def convert_keys_to_str(obj):
    """Recursivamente converte chaves Enum para strings em dicionários"""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, Enum):
                k = k.name  # 🔧 usa o .name do Enum (ex.: "HIGH", "OFF")
            new_dict[str(k)] = convert_keys_to_str(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_keys_to_str(i) for i in obj]
    else:
        return obj


def run_experiment(env_config, agent_config, name, save_dir):
    """Executa um experimento completo"""
    print(f"\n{'='*50}")
    print(f"Executando experimento: {name}")
    print(f"{'='*50}")
    
    env = ClassroomACEnvironment(env_config)
    agent = ACQLearningAgent(env.n_states, env.n_actions, agent_config)
    
    print("Treinando agente...")
    start_time = datetime.now()
    history = agent.train(env, verbose=True)
    training_time = datetime.now() - start_time
    
    print("\nAvaliando política...")
    eval_stats = agent.evaluate(env, episodes=10, render=False)
    
    policy_analysis = agent.analyze_policy(env)
    
    results = {
        'name': name,
        'config': {
            'env': asdict(env_config),
            'agent': asdict(agent_config)
        },
        'training_time': str(training_time),
        'training_history': history,
        'evaluation_stats': eval_stats,
        'policy_analysis': policy_analysis,
        'timestamp': datetime.now().isoformat()
    }
    
    # 🔧 Converte as chaves Enum antes de salvar
    results = convert_keys_to_str(results)
    
    model_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
    results_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_results.json")
    
    agent.save_model(model_path)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, cls=RobustJSONEncoder)
    
    print(f"Resultados salvos em: {results_path}")
    print(f"Modelo salvo em: {model_path}")
    
    return results


# As demais funções (plot_comparison, analyze_policy_behavior, main) seguem inalteradas...


def plot_comparison(results_list, save_dir):
    """Plota comparação entre diferentes experimentos"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Comparativo Entre Experimentos", fontsize=16, y=0.99)
    
    colors = plt.cm.get_cmap('Set1', len(results_list))
    window = 100

    # Recompensas
    for i, results in enumerate(results_list):
        rewards = results['training_history']['episode_rewards']
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, color=colors(i), linewidth=2, label=results['name'])
    axes[0, 0].set_title(f'Recompensas - Média Móvel ({window})')
    axes[0, 0].set_xlabel('Episódio'); axes[0, 0].set_ylabel('Recompensa Média')
    axes[0, 0].legend(); axes[0, 0].grid(True)

    # Decaimento epsilon
    for i, results in enumerate(results_list):
        if 'epsilons' in results['training_history']:
            epsilons = results['training_history']['epsilons']
            axes[0, 1].plot(epsilons, color=colors(i), label=results['name'])
    axes[0, 1].set_title('Decaimento do Epsilon')
    axes[0, 1].set_xlabel('Episódio'); axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].legend(); axes[0, 1].grid(True)

    # Conforto térmico
    for i, results in enumerate(results_list):
        comfort = results['training_history']['comfort_percentages']
        if len(comfort) >= window:
            moving_avg_comfort = np.convolve(comfort, np.ones(window)/window, mode='valid')
            axes[0, 2].plot(range(window-1, len(comfort)), moving_avg_comfort, color=colors(i), label=results['name'])
    axes[0, 2].set_title(f'Conforto Térmico - Média Móvel ({window})')
    axes[0, 2].set_xlabel('Episódio'); axes[0, 2].set_ylabel('Conforto (%)')
    axes[0, 2].legend(); axes[0, 2].grid(True)

    # Consumo energético
    for i, results in enumerate(results_list):
        energy = results['training_history']['energy_consumptions']
        if len(energy) >= window:
            moving_avg_energy = np.convolve(energy, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(energy)), moving_avg_energy, color=colors(i), label=results['name'])
    axes[1, 0].set_title(f'Consumo Energético - Média Móvel ({window})')
    axes[1, 0].set_xlabel('Episódio'); axes[1, 0].set_ylabel('Energia (kW)')
    axes[1, 0].legend(); axes[1, 0].grid(True)

    # Métricas finais - recompensa
    names = [r['name'] for r in results_list]
    final_rewards = [r['evaluation_stats']['avg_reward'] for r in results_list]
    axes[1, 1].bar(names, final_rewards, color=[colors(i) for i in range(len(results_list))])
    axes[1, 1].set_ylabel('Recompensa Média')
    axes[1, 1].set_title('Métricas Finais - Recompensa')
    plt.setp(axes[1, 1].get_xticklabels(), rotation=30, ha="right")
    axes[1, 1].grid(True, axis='y', alpha=0.5)

    # Conforto vs energia
    final_comfort = [r['evaluation_stats']['avg_comfort'] for r in results_list]
    final_energy = [r['evaluation_stats']['avg_energy'] for r in results_list]
    x = np.arange(len(names))
    width = 0.35
    
    ax2 = axes[1, 2]
    ax3 = ax2.twinx()
    ax2.bar(x - width/2, final_comfort, width, label='Conforto (%)', color='green')
    ax3.bar(x + width/2, final_energy, width, label='Energia (kW)', color='red')
    
    ax2.set_xlabel('Experimento')
    ax2.set_ylabel('Conforto (%)', color='green'); ax2.tick_params(axis='y', labelcolor='green')
    ax3.set_ylabel('Energia (kW)', color='red'); ax3.tick_params(axis='y', labelcolor='red')
    ax2.set_title('Conforto vs Consumo Energético')
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30, ha="right")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    comparison_path = os.path.join(save_dir, 'experiment_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparação salva em: {comparison_path}")


def analyze_policy_behavior(agent, env, save_dir, experiment_name):
    """Analisa o comportamento da política aprendida"""
    print(f"\nAnalisando comportamento da política: {experiment_name}")
    
    scenarios = [
        {'name': 'Sala Vazia (Manhã)', 'occupancy': 0, 'hour': 8, 'temp': 20},
        {'name': 'Sala Cheia (Manhã)', 'occupancy': 30, 'hour': 8, 'temp': 20},
        {'name': 'Sala Vazia (Tarde Quente)', 'occupancy': 0, 'hour': 14, 'temp': 30},
        {'name': 'Sala Cheia (Tarde Quente)', 'occupancy': 30, 'hour': 14, 'temp': 30},
        {'name': 'Sala Vazia (Noite)', 'occupancy': 0, 'hour': 20, 'temp': 18},
        {'name': 'Sala Cheia (Noite)', 'occupancy': 30, 'hour': 20, 'temp': 18}
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
    fig.suptitle(f'Análise de Comportamento da Política - {experiment_name}', fontsize=16)

    for i, scenario in enumerate(scenarios):
        env.reset() 
        env.current_temp = scenario['temp']; env.occupancy = scenario['occupancy']; env.hour_of_day = scenario['hour']
        state = env._discretize_state()
        
        for _ in range(100):
            action = agent.choose_action(state, training=False)
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done: break
        
        row, col = i // 3, i % 3
        ax1 = axes[row, col]
        ax1.plot(env.temp_history, label='Temperatura', color='blue')
        ax1.axhline(y=env.config.temp_comfort_min, c='g', ls='--'); ax1.axhline(y=env.config.temp_comfort_max, c='g', ls='--')
        ax1.set_title(scenario['name']); ax1.set_ylabel('Temperatura (°C)', color='blue'); ax1.tick_params(axis='y', labelcolor='blue'); ax1.grid(True)
        
        ax2 = ax1.twinx()
        ax2.plot([s.value for s in env.ac_state_history], label='Estado AC', color='red', alpha=0.7, drawstyle='steps-post')
        ax2.set_ylabel('Estado AC', color='red'); ax2.tick_params(axis='y', labelcolor='red'); ax2.set_yticks(range(4)); ax2.set_yticklabels(['OFF', 'LOW', 'MED', 'HIGH'])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    analysis_path = os.path.join(save_dir, f"{experiment_name.lower().replace(' ', '_')}_policy_analysis.png")
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Análise de comportamento salva em: {analysis_path}")


def main():
    """Função principal"""
    print("Sistema de Controle de Ar-Condicionado com Aprendizagem por Reforço")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Resultados serão salvos em: {save_dir}")
    
    experiment_configs = create_experiment_configs()
    results_list = []
    
    for config_name, config in experiment_configs.items():
        try:
            results = run_experiment(config['env_config'], config['agent_config'], config['name'], save_dir)
            results_list.append(results)
        except Exception as e:
            print(f"Erro no experimento {config_name}: {e}")
            continue
    
    if len(results_list) > 1:
        print("\nGerando comparação entre experimentos...")
        plot_comparison(results_list, save_dir)
    
    if results_list:
        try:
            best_result = max(results_list, key=lambda x: x['evaluation_stats']['avg_reward'])
            print(f"\nAnalisando em detalhe o melhor experimento: {best_result['name']}")
            
            best_agent = ACQLearningAgent(0, 0)
            model_path = os.path.join(save_dir, f"{best_result['name'].lower().replace(' ', '_')}_model.pkl")
            best_agent.load_model(model_path)
            
            # Recria a configuração do ambiente a partir dos resultados salvos
            env_config_data = best_result['config']['agent']
            config_args = {key: val for key, val in env_config_data.items() if key in ClassroomConfig.__annotations__}
            best_env_config = ClassroomConfig(**config_args)
            best_env = ClassroomACEnvironment(best_env_config)
            
            analyze_policy_behavior(best_agent, best_env, save_dir, best_result['name'])
        except Exception as e:
            print(f"\nErro ao analisar o melhor experimento: {e}")

    print(f"\nExperimentos concluídos! Resultados salvos em: {save_dir}")


if __name__ == "__main__":
    main()
