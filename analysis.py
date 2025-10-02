# -*- coding: utf-8 -*-
"""
Script de Análise dos Resultados de Experimentos de Aprendizado por Reforço (Versão Final)

Este script realiza uma análise completa dos resultados dos experimentos:
1. Resume as condições iniciais do experimento (parâmetros de agente e ambiente).
2. Carrega os dados de resultado dos arquivos .json.
3. Realiza análise estatística descritiva das métricas finais de avaliação.
4. Plota as curvas de aprendizado do histórico de treinamento.
5. Carrega os modelos .pkl treinados, simula um cenário desafiador e plota a
   comparação do controle de temperatura.

Como usar:
$ python analysis.py <caminho_para_a_pasta_de_resultados>
"""

import os
import json
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importa as classes do ambiente e do agente, se disponíveis
try:
    from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig
    from ac_qlearning_agent import ACQLearningAgent
except ImportError:
    print("Aviso: Não foi possível importar 'ClassroomACEnvironment' ou 'ACQLearningAgent'.")
    print("A simulação de comportamento de temperatura será pulada.")
    ClassroomACEnvironment = None
    ACQLearningAgent = None

# --- INÍCIO DA SEÇÃO ADICIONADA: RESUMO DAS CONDIÇÕES DO EXPERIMENTO ---

def plot_hyperparameters(df_agents: pd.DataFrame, save_dir: str):
    """Plota um gráfico de barras comparando os hiperparâmetros dos agentes."""
    print("\n🎨 Gerando gráfico de comparação de hiperparâmetros...")
    df_plot = df_agents[['alpha', 'gamma', 'epsilon']].copy()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    df_plot.plot(kind='bar', ax=ax, colormap='viridis', rot=0)
    
    ax.set_xlabel('Configuração do Agente', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valores dos Hiperparâmetros', fontsize=12, fontweight='bold')
    ax.set_title('Comparação dos Hiperparâmetros Iniciais dos Agentes', fontsize=15, fontweight='bold')
    ax.legend(title='Hiperparâmetro')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'analysis_hyperparameters_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✅ Gráfico de hiperparâmetros salvo em: {save_path}")

def summarize_experiment_conditions(results_data: dict, results_dir: str):
    """Exibe tabelas e gráficos resumindo as condições iniciais dos experimentos."""
    print("\n" + "="*70)
    print("📋 RESUMO DAS CONDIÇÕES DO EXPERIMENTO")
    print("="*70)

    # 1. Tabela de Hiperparâmetros dos Agentes (Variáveis)
    print("\n1. Hiperparâmetros dos Agentes (Valores Variáveis)")
    print("-" * 50)
    agent_configs_list = []
    for name, data in results_data.items():
        config = data.get('config', {}).get('agent', {})
        config['Experimento'] = data.get('name', name)
        agent_configs_list.append(config)
    
    df_agents = pd.DataFrame(agent_configs_list).set_index('Experimento')
    print(df_agents.to_markdown())
    plot_hyperparameters(df_agents, results_dir)

    # 2. Tabela de Parâmetros do Ambiente (Constantes)
    print("\n2. Parâmetros do Ambiente (Valores Constantes)")
    print("-" * 50)
    first_result = list(results_data.values())[0]
    env_config = first_result.get('config', {}).get('env', {})
    
    # Adiciona placeholders para variáveis não salvas no JSON
    env_config['thermal_mass'] = 'Não disponível no JSON'
    env_config['heat_gain_per_person'] = 'Não disponível no JSON'
    
    df_env = pd.DataFrame.from_dict(env_config, orient='index', columns=['Valor'])
    df_env.index.name = 'Parâmetro'
    print(df_env.to_markdown())
    print("\nNota: Para exibir todos os parâmetros do ambiente, eles precisariam ser salvos no arquivo results.json durante o treinamento.")

    # 3. Tabela de Cenários de Análise Pós-Treinamento
    print("\n3. Cenários de Análise (usados para testar o melhor agente)")
    print("-" * 50)
    scenarios = {
        'Sala Vazia (Manhã)': {'Ocupação': 0, 'Hora do Dia': 8},
        'Sala Cheia (Manhã)': {'Ocupação': 30, 'Hora do Dia': 8},
        'Sala Vazia (Tarde)': {'Ocupação': 0, 'Hora do Dia': 14},
        'Sala Cheia (Tarde)': {'Ocupação': 30, 'Hora do Dia': 14},
    }
    df_scenarios = pd.DataFrame.from_dict(scenarios, orient='index')
    df_scenarios.index.name = 'Cenário'
    print(df_scenarios.to_markdown())
    print("="*70)

# --- FIM DA SEÇÃO ADICIONADA ---


def load_experiment_data(results_dir: str) -> dict:
    """Carrega todos os arquivos de resultado .json de um diretório."""
    print(f"\n🔎 Carregando dados do diretório: {results_dir}")
    # ... (código restante da função é o mesmo)
    # ... (código das outras funções: perform_descriptive_analysis, plot_training_history, etc. permanecem os mesmos)
    print(f"\n🔎 Carregando arquivos de resultado de: {results_dir}")
    json_files = glob.glob(os.path.join(results_dir, '*_results.json'))
    if not json_files:
        print(f"⚠️ Nenhum arquivo '*_results.json' encontrado em '{results_dir}'.")
        return {}

    results_data = {}
    for file_path in sorted(json_files):
        experiment_name = os.path.basename(file_path).replace('_results.json', '')
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results_data[experiment_name] = data
                print(f"  - Arquivo '{os.path.basename(file_path)}' carregado.")
        except Exception as e:
            print(f"  - Erro ao carregar {file_path}: {e}")
    return results_data

def perform_descriptive_analysis(results_data: dict):
    """Realiza e exibe uma análise estatística descritiva das métricas de avaliação."""
    print("\n" + "="*60)
    print("📊 Análise Estatística Descritiva (Métricas de Avaliação)")
    print("="*60)

    eval_stats_list = [
        {**data.get('evaluation_stats', {}), 'experiment': data.get('name', name)}
        for name, data in results_data.items()
    ]
    if not eval_stats_list:
        print("Nenhuma estatística de avaliação para analisar.")
        return

    df_eval = pd.DataFrame(eval_stats_list).set_index('experiment')
    desired_columns = ['avg_reward', 'avg_comfort', 'avg_energy', 'avg_temp']
    columns_to_analyze = [col for col in desired_columns if col in df_eval.columns]
    
    if 'avg_temp' in columns_to_analyze:
        df_eval.rename(columns={'avg_temp': 'avg_temp_c'}, inplace=True)
        columns_to_analyze[columns_to_analyze.index('avg_temp')] = 'avg_temp_c'

    print("Analisando colunas disponíveis:", columns_to_analyze)
    print("\nResumo Estatístico:")
    print(df_eval[columns_to_analyze].describe().round(2))
    print("\nValores por Experimento:")
    print(df_eval[columns_to_analyze].round(2))
    print("="*60 + "\n")

def plot_training_history(results_data: dict, save_dir: str):
    """Plota as curvas de aprendizado do histórico de treinamento."""
    print("🎨 Gerando gráficos do histórico de treinamento...")
    if not results_data:
        print("Dados insuficientes para gerar gráficos.")
        return

    sns.set_style("darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Análise Comparativa do Desempenho Durante o Treinamento', fontsize=18, weight='bold')
    
    palette = dict(zip([data['name'] for data in results_data.values()], sns.color_palette("viridis", len(results_data))))
    window = 50 

    ax_reward, ax_comfort = axes[0, 0], axes[0, 1]
    ax_energy, ax_epsilon = axes[1, 0], axes[1, 1]

    for name, data in results_data.items():
        # Recompensa
        rewards = data['training_history']['episode_rewards']
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax_reward.plot(range(window-1, len(rewards)), moving_avg, label=data['name'], color=palette[data['name']], lw=2)
        # Conforto
        comfort = data['training_history']['comfort_percentages']
        moving_avg_comfort = np.convolve(comfort, np.ones(window)/window, mode='valid')
        ax_comfort.plot(range(window-1, len(comfort)), moving_avg_comfort, label=data['name'], color=palette[data['name']], lw=2)
        # Energia
        energy = data['training_history']['energy_consumptions']
        moving_avg_energy = np.convolve(energy, np.ones(window)/window, mode='valid')
        ax_energy.plot(range(window-1, len(energy)), moving_avg_energy, label=data['name'], color=palette[data['name']], lw=2)
        # Epsilon
        if 'epsilon_history' in data['training_history']:
            epsilons = data['training_history']['epsilon_history']
            ax_epsilon.plot(epsilons, label=data['name'], color=palette[data['name']], lw=2, alpha=0.8)

    ax_reward.set_title(f'Recompensa Média (Média Móvel de {window})', fontsize=12)
    ax_reward.legend()
    ax_comfort.set_title(f'Conforto Térmico (Média Móvel de {window})', fontsize=12)
    ax_comfort.legend()
    ax_energy.set_title(f'Consumo de Energia (Média Móvel de {window})', fontsize=12)
    ax_energy.legend()
    ax_epsilon.set_title('Decaimento da Taxa de Exploração (Epsilon)', fontsize=12)
    ax_epsilon.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = os.path.join(save_dir, 'analysis_training_performance.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✅ Gráfico de histórico de treinamento salvo em: {save_path}")

def run_evaluation_episode(agent: ACQLearningAgent, env: ClassroomACEnvironment, start_temp: float) -> list:
    state = env.reset(start_temp=start_temp) # Passa a temperatura para o reset
    temp_history = [env.current_temp]
    SIMULATION_HORIZON = 240  # 240 passos * 6 min/passo = 24 horas de simulação

    for _ in range(SIMULATION_HORIZON): 
        action = agent.choose_action(state, training=False)
        next_state, _, done, info = env.step(action)
        temp_history.append(info['temperature'])
        state = next_state
        if done:
            break
    return temp_history

def plot_temperature_comparison(results: dict, config: ClassroomConfig, save_dir: str):
    """Plota as curvas de temperatura de múltiplos agentes."""
    print("🎨 Gerando gráfico de comparação de controle de temperatura...")
    plt.figure(figsize=(12, 7))
    
    plt.axhline(y=config.temp_comfort_min, color='green', linestyle='--', alpha=0.8, label=f'Conforto Min ({config.temp_comfort_min}°C)')
    plt.axhline(y=config.temp_comfort_max, color='green', linestyle='--', alpha=0.8, label=f'Conforto Max ({config.temp_comfort_max}°C)')

    for name, temp_history in results.items():
        plt.plot(temp_history, label=name, lw=2, alpha=0.9)

    plt.title('Comparação do Controle de Temperatura (Cenário Desafiador)', fontsize=16)
    plt.xlabel('Passos de Tempo (intervalos de 6 min)', fontsize=12)
    plt.ylabel('Temperatura da Sala (°C)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.ylim(config.temp_comfort_min - 2, config.temp_comfort_max + 4)
    
    save_path = os.path.join(save_dir, 'analysis_temperature_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✅ Gráfico de controle de temperatura salvo em: {save_path}")

def main(results_dir: str):
    """Função principal para orquestrar a análise."""
    results_data = load_experiment_data(results_dir)
    if not results_data:
        print("Análise encerrada por falta de dados.")
        return

    # 1. Resumo das Condições Iniciais do Experimento
    summarize_experiment_conditions(results_data, results_dir)

    # 2. Análise Estatística dos Resultados
    perform_descriptive_analysis(results_data)
    
    # 3. Plotagem do Histórico de Treinamento
    plot_training_history(results_data, results_dir)
    
    # 4. Análise de Comportamento por Simulação
    if ClassroomACEnvironment is None or ACQLearningAgent is None:
        print("\n pulando simulação de temperatura devido a erro na importação das classes.")
    else:
        print("\n" + "="*60)
        print("🔄 Iniciando simulação para análise de temperatura")
        print("="*60)
        
        temp_histories = {}
        env_config = ClassroomConfig()
        env = ClassroomACEnvironment(env_config)
        initial_temp_for_sim = 27
        env.hour_of_day, env.occupancy, env.current_temp = 14, 35, initial_temp_for_sim
        
        for name, data in results_data.items():
            model_path = os.path.join(results_dir, f"{name}_model.pkl")
            if os.path.exists(model_path):
                print(f"  - Carregando e simulando para: '{data['name']}'")
                try:
                    agent = ACQLearningAgent(n_states=0, n_actions=0, config=None)
                    agent.load_model(model_path)
                    history = run_evaluation_episode(agent, env, start_temp=initial_temp_for_sim)
                    temp_histories[data['name']] = history
                except Exception as e:
                    print(f"    └─ Erro ao simular modelo '{name}': {e}")
            else:
                print(f"  - ⚠️ Modelo '{model_path}' não encontrado. Pulando simulação.")

        if temp_histories:
            plot_temperature_comparison(temp_histories, env_config, results_dir)
        else:
            print("Nenhum histórico de temperatura foi gerado para plotagem.")
            
    print("\n🎉 Análise concluída com sucesso!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analisa os resultados de experimentos de RL.")
    parser.add_argument("results_dir", type=str, help="Caminho para o diretório de resultados.")
    args = parser.parse_args()
    if not os.path.isdir(args.results_dir):
        print(f"Erro: O diretório '{args.results_dir}' não foi encontrado.")
    else:
        main(args.results_dir)