# -*- coding: utf-8 -*-
"""
Script Principal para Treinamento e Avaliação do Agente de Controle de Ar-Condicionado

Este script orquestra o treinamento de um agente Q-Learning para controlar
o sistema de ar-condicionado de uma sala de aula, incluindo:
- Treinamento do agente
- Avaliação da política aprendida
- Análise de resultados
- Visualizações

Autor: Renan Saraiva dos Santos
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import dataclasses
import traceback
import pickle

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Supondo que estes arquivos existem e estão corretos
from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig
from ac_qlearning_agent import ACQLearningAgent, QLearningConfig

def create_experiment_configs():
    EPISODES = 10000
    configs = {
        'baseline': {
            'env_config': ClassroomConfig(),
            'agent_config': QLearningConfig(alpha=0.1, gamma=0.95, epsilon=0.2, episodes=EPISODES),
            'name': 'Baseline'
        },
        'high_exploration': {
            'env_config': ClassroomConfig(),
            'agent_config': QLearningConfig(alpha=0.1, gamma=0.95, epsilon=0.5, epsilon_decay=0.99, episodes=EPISODES),
            'name': 'High Exploration'
        },
        'low_learning_rate': {
            'env_config': ClassroomConfig(),
            'agent_config': QLearningConfig(alpha=0.05, gamma=0.95, epsilon=0.2, episodes=EPISODES),
            'name': 'Low Learning Rate'
        },
        'high_discount': {
            'env_config': ClassroomConfig(),
            'agent_config': QLearningConfig(alpha=0.1, gamma=0.99, epsilon=0.2, episodes=EPISODES),
            'name': 'High Discount Factor'
        }
    }
    return configs

# ALTERAÇÃO 1: Criamos uma função de sanitização robusta e recursiva.
# Esta função substitui a antiga 'convert_numpy'.
def sanitize_for_json(obj):
    """
    Converte recursivamente um objeto para que seja serializável em JSON.
    - Converte chaves de dicionário para strings.
    - Converte tipos numéricos e arrays do NumPy para tipos nativos do Python.
    """
    if isinstance(obj, dict): return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [sanitize_for_json(elem) for elem in obj]
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def run_experiment(env_config, agent_config, name, save_dir):
    """Executa um experimento completo com rastreamento MLflow seguro."""
    
    # O bloco 'with' garante que mlflow.end_run() seja chamado automaticamente no final,
    # mesmo que ocorram erros, resolvendo o problema de "run ativa".
    with mlflow.start_run(run_name=name):
        print(f"\n{'='*50}")
        print(f"Executando experimento: {name}")
        print(f"{'='*50}")
        
        if MLFLOW_AVAILABLE:
            mlflow.log_params(dataclasses.asdict(agent_config))
            env_params = {f"env_{k}": v for k, v in dataclasses.asdict(env_config).items() if not isinstance(v, dict)}
            mlflow.log_params(env_params)

        env = ClassroomACEnvironment(env_config)
        agent = ACQLearningAgent(env.n_states, env.n_actions, agent_config)
        
        print("Treinando agente...")
        start_time = datetime.now()
        history = agent.train(env, verbose=True)
        training_time = datetime.now() - start_time
        
        print("\nAvaliando política...")
        eval_stats = agent.evaluate(env, episodes=10, render=False)
        
        policy_analysis = agent.analyze_policy(env)
        
        # O dicionário 'results' está incompleto no seu script, vamos usar dataclasses como sugerido antes
        results = {
            'name': name,
            'config': {
                'env': dataclasses.asdict(env_config),
                'agent': dataclasses.asdict(agent_config)
            },
            'training_time': str(training_time),
            'training_history': history,
            'evaluation_stats': eval_stats,
            'policy_analysis': policy_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        model_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
        results_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_results.json")
        
        agent.save_model(model_path)
        
        sanitized_results = sanitize_for_json(results)
        with open(results_path, 'w') as f:
            json.dump(sanitized_results, f, indent=2)
        
        print(f"Resultados salvos em: {results_path}")
        print(f"Modelo salvo em: {model_path}")

        if MLFLOW_AVAILABLE:
            mlflow.log_metrics(eval_stats)
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(results_path)
            print(f"Resultados para '{name}' registrados com sucesso no MLflow.")
        
    return results # Retorna os resultados originais (com tipos numpy) para os plots

def plot_comparison(results_list, save_dir):
    """Plota comparação entre diferentes experimentos"""
    # Adicionando um estilo visual mais agradável
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Análise Comparativa dos Experimentos de RL', fontsize=16, weight='bold')

    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))
    
    # 1. Recompensas por episódio (com média móvel)
    ax = axes[0, 0]
    window = 50
    for i, results in enumerate(results_list):
        rewards = results['training_history']['episode_rewards']
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), moving_avg, 
                    color=colors[i], linewidth=2, label=results['name'])
    ax.set_title(f'Recompensas (Média Móvel de {window} Episódios)')
    ax.set_xlabel('Episódio')
    ax.set_ylabel('Recompensa Média')
    ax.legend()
    
    # 2. Percentual de conforto (com média móvel)
    ax = axes[0, 1]
    for i, results in enumerate(results_list):
        comfort = results['training_history']['comfort_percentages']
        if len(comfort) >= window:
            moving_avg = np.convolve(comfort, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(comfort)), moving_avg, 
                    color=colors[i], linewidth=2, label=results['name'])
    ax.set_title(f'Conforto Térmico (Média Móvel de {window} Episódios)')
    ax.set_xlabel('Episódio')
    ax.set_ylabel('Conforto Médio (%)')
    ax.legend()
    
    # 3. Consumo energético (com média móvel)
    ax = axes[0, 2]
    for i, results in enumerate(results_list):
        energy = results['training_history']['energy_consumptions']
        if len(energy) >= window:
            moving_avg = np.convolve(energy, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(energy)), moving_avg, 
                    color=colors[i], linewidth=2, label=results['name'])
    ax.set_title(f'Consumo Energético (Média Móvel de {window} Episódios)')
    ax.set_xlabel('Episódio')
    ax.set_ylabel('Energia Média (kW)')
    ax.legend()
    
    # 4. Métricas de Avaliação Final (Recompensa)
    ax = axes[1, 0]
    names = [r['name'] for r in results_list]
    final_rewards = [r['evaluation_stats']['avg_reward'] for r in results_list]
    reward_std = [r['evaluation_stats']['std_reward'] for r in results_list]
    ax.bar(names, final_rewards, yerr=reward_std, capsize=5, color=colors, alpha=0.8)
    ax.set_title('Recompensa Média na Avaliação')
    ax.set_ylabel('Recompensa')
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    
    # 5. Métricas de Avaliação Final (Conforto)
    ax = axes[1, 1]
    final_comfort = [r['evaluation_stats']['avg_comfort'] for r in results_list]
    comfort_std = [r['evaluation_stats']['std_comfort'] for r in results_list]
    ax.bar(names, final_comfort, yerr=comfort_std, capsize=5, color=colors, alpha=0.8)
    ax.set_title('Conforto Médio na Avaliação')
    ax.set_ylabel('Conforto (%)')
    ax.set_ylim(0, 100)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # 6. Métricas de Avaliação Final (Energia)
    ax = axes[1, 2]
    final_energy = [r['evaluation_stats']['avg_energy'] for r in results_list]
    energy_std = [r['evaluation_stats']['std_energy'] for r in results_list]
    ax.bar(names, final_energy, yerr=energy_std, capsize=5, color=colors, alpha=0.8)
    ax.set_title('Consumo de Energia na Avaliação')
    ax.set_ylabel('Energia (kW)')
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Salva figura
    comparison_path = os.path.join(save_dir, 'experiment_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparação salva em: {comparison_path}")

# Esta função permanece a mesma, pois as correções não a afetam.
def analyze_policy_behavior(agent, env, save_dir):
    """Analisa o comportamento da política aprendida em cenários mais realistas."""
    print("\nAnalisando comportamento da política...")
    
    scenarios = [
        {'name': 'Manhã Fria, Sala Vazia',   'start_temp': 18.0, 'occupancy': 0,  'hour': 8},
        {'name': 'Manhã Agradável, Sala Cheia', 'start_temp': 23.0, 'occupancy': 30, 'hour': 9},
        {'name': 'Tarde Quente, Sala Vazia',  'start_temp': 28.0, 'occupancy': 0,  'hour': 14},
        {'name': 'Tarde Quente, Sala Cheia',  'start_temp': 28.0, 'occupancy': 30, 'hour': 14},
        {'name': 'Noite Agradável, Sala Vazia', 'start_temp': 24.0, 'occupancy': 0,  'hour': 20},
        {'name': 'Noite Quente, Sala Cheia',  'start_temp': 27.0, 'occupancy': 20, 'hour': 20}
    ]
    
    scenario_results = []
    for scenario in scenarios:
        print(f"\nTestando cenário: {scenario['name']} (Iniciando em {scenario['start_temp']}°C)")
        state = env.reset(start_temp=scenario['start_temp'])
        env.occupancy, env.hour_of_day = scenario['occupancy'], scenario['hour']
        state = env._discretize_state()
        
        episode_data = {'temperatures': [], 'ac_states': []}
        for _ in range(100):
            action = agent.choose_action(state, training=False)
            next_state, _, done, info = env.step(action)
            episode_data['temperatures'].append(info['temperature'])
            episode_data['ac_states'].append(action)
            state = next_state
            if done: break
        
        scenario_results.append((scenario['name'], episode_data))
        stats = env.get_statistics()
        print(f"  Temperatura média: {stats['avg_temperature']:.1f}°C, Conforto: {stats['comfort_percentage']:.1f}%, "
              f"Uso do AC: {stats['ac_usage_percentage']:.1f}%, Energia: {stats['total_energy_consumption']:.1f} kW")
    
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
    fig.suptitle('Análise de Comportamento da Política em Diferentes Cenários', fontsize=16, weight='bold')

    # --- INÍCIO DA CORREÇÃO DO BUG ---
    # O loop agora desempacota a tupla em 'name' e 'data'
    for i, (name, data) in enumerate(scenario_results):
        row, col = i // 3, i % 3
        ax1 = axes[row, col]
        
        # Acessa os dados usando a variável 'data'
        ax1.plot(data['temperatures'], label='Temperatura (°C)', color='royalblue', linewidth=2)
        ax1.axhline(y=22, color='green', linestyle='--', alpha=0.7, label='Faixa Conforto')
        ax1.axhline(y=26, color='green', linestyle='--', alpha=0.7)
        ax1.set_title(name, weight='bold') # Usa a variável 'name'
        ax1.set_xlabel('Passos de Tempo (6 min cada)'); ax1.set_ylabel('Temperatura (°C)', color='royalblue')
        ax1.tick_params(axis='y', labelcolor='royalblue'); ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(data['ac_states'], label='Estado AC', color='crimson', alpha=0.6, drawstyle='steps-pre')
        ax2.set_ylabel('Estado do AC', color='crimson'); ax2.tick_params(axis='y', labelcolor='crimson')
        ax2.set_yticks([0, 1, 2, 3]); ax2.set_yticklabels(['OFF', 'LOW', 'MED', 'HIGH'])
        ax2.legend(loc='upper right')
    # --- FIM DA CORREÇÃO DO BUG ---

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    analysis_path = os.path.join(save_dir, 'policy_behavior_analysis.png')
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Análise de comportamento salva em: {analysis_path}")
    return scenario_results


def main():
    """Função principal"""
    if MLFLOW_AVAILABLE:
        print("✅ MLflow está disponível. Os experimentos serão rastreados.")
        mlflow.set_experiment("Controle de AC - Treinamento V3")
    else:
        print("⚠️  Aviso: MLflow não foi encontrado.")
    print("Sistema de Controle de Ar-Condicionado com Aprendizagem por Reforço\n" + "="*70)
    
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
            traceback.print_exc()
            continue
    
    if len(results_list) > 1:
        print("\nGerando comparação entre experimentos...")
        plot_comparison(results_list, save_dir)
    
    if results_list:
        best_result = max(results_list, key=lambda x: x['evaluation_stats']['avg_reward'])
        print(f"\nAnalisando o melhor experimento: {best_result['name']}")
        print(f"  Recompensa média de avaliação: {best_result['evaluation_stats']['avg_reward']:.2f}")
        print(f"  Conforto médio de avaliação: {best_result['evaluation_stats']['avg_comfort']:.1f}%")
        
        best_agent = ACQLearningAgent(0, 0)
        model_path = os.path.join(save_dir, f"{best_result['name'].lower().replace(' ', '_')}_model.pkl")
        best_agent.load_model(model_path)
        
        best_env_config_dict = best_result['config']['env']
        best_env_config = ClassroomConfig(**best_env_config_dict)
        best_env = ClassroomACEnvironment(best_env_config)
        
        analyze_policy_behavior(best_agent, best_env, save_dir)
    
    print(f"\nExperimentos concluídos! Resultados salvos em: {save_dir}")

if __name__ == "__main__":
    main()