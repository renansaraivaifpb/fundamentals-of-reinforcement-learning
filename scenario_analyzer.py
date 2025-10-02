# scenario_analyzer_all.py
# -*- coding: utf-8 -*-
"""
Script para Análise de Comportamento de TODOS os Agentes de RL em Múltiplos Cenários

Este script carrega TODOS os agentes treinados de uma pasta de resultados e
os avalia em um conjunto predefinido de cenários operacionais.

Como usar:
$ python scenario_analyzer_all.py <caminho_para_a_pasta_de_resultados>
Exemplo:
$ python scenario_analyzer_all.py results_20251002_174127
"""

import os
import json
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importa as classes do ambiente e do agente
try:
    from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig
    from ac_qlearning_agent import ACQLearningAgent, QLearningConfig
except ImportError:
    print("ERRO: Certifique-se de que os arquivos 'classroom_ac_env.py' e 'ac_qlearning_agent.py' estão no mesmo diretório.")
    exit()

def load_agent_and_config(results_dir: str, model_name: str) -> tuple:
    """Carrega um modelo de agente salvo e sua configuração de ambiente correspondente."""
    model_path = os.path.join(results_dir, f"{model_name}_model.pkl")
    config_path = os.path.join(results_dir, f"{model_name}_results.json")

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"ERRO: Arquivos de modelo ('{model_path}') ou de resultados ('{config_path}') não encontrados.")
        return None, None

    print(f"🔎 Carregando modelo de '{os.path.basename(model_path)}'...")
    agent = ACQLearningAgent(n_states=0, n_actions=0)
    agent.load_model(model_path)

    print(f"🔎 Carregando configuração de ambiente de '{os.path.basename(config_path)}'...")
    with open(config_path, 'r') as f:
        results = json.load(f)
        env_config_dict = results['config']['env']
    
    env_config = ClassroomConfig(**env_config_dict)
    
    return agent, env_config

def run_simulation_for_scenario(agent: ACQLearningAgent, env: ClassroomACEnvironment, scenario: dict) -> pd.DataFrame:
    """Executa uma simulação para um único cenário e retorna o histórico."""
    print(f"  -> Simulando cenário: '{scenario['name']}'...")
    
    state = env.reset(start_temp=scenario['start_temp'])
    env.occupancy = scenario['occupancy']
    env.hour_of_day = scenario['hour']
    state = env._discretize_state()

    history = []
    simulation_steps = 120

    for step in range(simulation_steps):
        action = agent.choose_action(state, training=False)
        next_state, reward, done, info = env.step(action)
        
        history.append({
            'step': step,
            'temperature': info['temperature'],
            'ac_action': action,
            'comfort_level': 1 if info['comfort_level'] == 'COMFORTABLE' else 0,
            'energy_consumption': info['energy_consumption'],
            'reward': reward
        })
        state = next_state
        if done:
            break
            
    return pd.DataFrame(history)

def plot_scenario_results(all_results: dict, env_config: ClassroomConfig, save_dir: str, model_name: str):
    """Plota os resultados de todos os cenários, criando uma nova figura para cada grupo de 5."""
    print(f"\n🎨 Gerando visualizações para o agente '{model_name.title()}'...")
    sns.set_style("darkgrid")
    
    scenarios = list(all_results.items())
    chunk_size = 5

    for i in range(0, len(scenarios), chunk_size):
        chunk = scenarios[i:i + chunk_size]
        page_num = (i // chunk_size) + 1
        
        print(f"  -> Gerando página de gráficos nº {page_num}...")

        ncols = 2 if len(chunk) > 1 else 1
        nrows = (len(chunk) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 5 * nrows), squeeze=False)
        axes = axes.flatten()

        fig.suptitle(f'Análise do Agente "{model_name.title()}" - Cenários (Parte {page_num})', fontsize=18, weight='bold')

        for j, (scenario_name, df_history) in enumerate(chunk):
            ax1 = axes[j]
            ax1.plot(df_history['step'], df_history['temperature'], label='Temperatura (°C)', color='royalblue', lw=2.5)
            ax1.axhspan(env_config.temp_comfort_min, env_config.temp_comfort_max, color='green', alpha=0.15, label='Faixa de Conforto')
            ax1.set_ylabel('Temperatura (°C)', color='royalblue', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='royalblue')
            ax1.set_title(scenario_name, fontsize=14, weight='bold')
            ax1.grid(True, which='major', linestyle='--', linewidth=0.5)

            ax2 = ax1.twinx()
            ax2.step(df_history['step'], df_history['ac_action'], where='post', label='Ação do AC', color='crimson', alpha=0.7, lw=2)
            ax2.set_ylabel('Ação do AC', color='crimson', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='crimson')
            ax2.set_yticks([0, 1, 2, 3]); ax2.set_yticklabels(['OFF', 'LOW', 'MED', 'HIGH'])
            ax2.set_ylim(-0.5, 3.5)
            
            lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        for k in range(j + 1, len(axes)):
            axes[k].set_visible(False)

        fig.text(0.5, 0.01, 'Passos de Tempo (intervalos de 6 min)', ha='center', va='center', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        save_path = os.path.join(save_dir, f'analysis_scenarios_{model_name.replace(" ", "_")}_part{page_num}.png')
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"✅ Gráfico salvo em: {save_path}")

# ALTERAÇÃO: A lógica principal foi movida para esta nova função
def analyze_single_agent(results_dir: str, model_name: str):
    """Orquestra o carregamento, simulação e análise de um único agente."""
    
    print("\n" + "="*80)
    print(f"ANÁLISE DO AGENTE: {model_name.upper()}")
    print("="*80)

    agent, env_config = load_agent_and_config(results_dir, model_name)
    if agent is None:
        return

    env = ClassroomACEnvironment(env_config)

    scenarios = [
        {'name': '1: Manhã Fria, Sala Vazia', 'start_temp': 19.0, 'occupancy': 0, 'hour': 8},
        {'name': '2: Manhã Agradável, Sala Enchendo', 'start_temp': 23.0, 'occupancy': 25, 'hour': 9},
        {'name': '3: Tarde Quente, Sala Cheia', 'start_temp': 28.0, 'occupancy': 30, 'hour': 14},
        {'name': '4: Tarde Quente, Sala Superlotada', 'start_temp': 29.0, 'occupancy': 40, 'hour': 15},
        {'name': '5: Fim de Tarde, Sala Esvaziando', 'start_temp': 25.0, 'occupancy': 5, 'hour': 18},
        {'name': '6: Noite, Sala Vazia', 'start_temp': 24.0, 'occupancy': 0, 'hour': 22},
        {'name': '7: Tarde Agradável, Sala Vazia (Manutenção)', 'start_temp': 24.5, 'occupancy': 0, 'hour': 16},
        {'name': '8: Onda de Calor Extrema, Sala Cheia (Estresse)', 'start_temp': 32.0, 'occupancy': 30, 'hour': 15},
        {'name': '9: Fim de Expediente (Eficiência Energética)', 'start_temp': 25.5, 'occupancy': 0, 'hour': 19},
        {'name': '10: Inverno Hipotético (Caso de Borda/Inação)', 'start_temp': 12.0, 'occupancy': 10, 'hour': 10},
    ]

    all_scenario_results = {}
    for scenario in scenarios:
        df_result = run_simulation_for_scenario(agent, env, scenario)
        all_scenario_results[scenario['name']] = df_result

    plot_scenario_results(all_scenario_results, env_config, results_dir, model_name)

    summary_data = []
    for name, df in all_scenario_results.items():
        summary_data.append({
            'Cenário': name,
            'Temp. Média (°C)': df['temperature'].mean(),
            'Tempo em Conforto (%)': df['comfort_level'].mean() * 100,
            'Energia Total (kW)': df['energy_consumption'].sum(),
            'Recompensa Média': df['reward'].mean(),
            'Ação Predominante': df['ac_action'].mode().iloc[0] if not df['ac_action'].empty else 0
        })
    
    df_summary = pd.DataFrame(summary_data).set_index('Cenário')
    df_summary['Ação Predominante'] = df_summary['Ação Predominante'].map({0: 'OFF', 1: 'LOW', 2: 'MED', 3: 'HIGH'})

    print("\n" + "-"*80)
    print(f"📊 RESUMO QUANTITATIVO DO DESEMPENHO DO AGENTE '{model_name.title()}'")
    print("-" * 80)
    print(df_summary.round(2).to_markdown())

# ALTERAÇÃO: A função main agora descobre e itera sobre todos os modelos
def main(results_dir: str):
    """Descobre todos os modelos na pasta de resultados e executa a análise para cada um."""
    
    # Encontra todos os arquivos de modelo .pkl no diretório
    model_files = glob.glob(os.path.join(results_dir, '*_model.pkl'))
    
    if not model_files:
        print(f"Nenhum arquivo '*_model.pkl' encontrado no diretório '{results_dir}'.")
        return

    # Extrai os nomes dos modelos dos nomes dos arquivos
    model_names = [os.path.basename(f).replace('_model.pkl', '') for f in model_files]
    
    print(f"Encontrados {len(model_names)} modelos para analisar: {', '.join(model_names)}")
    
    # Executa a análise para cada modelo encontrado
    for model_name in sorted(model_names):
        analyze_single_agent(results_dir, model_name)

    print("\n\n🎉 Análise completa de todos os agentes concluída com sucesso!")


if __name__ == '__main__':
    # ALTERAÇÃO: O script agora espera apenas um argumento: o diretório de resultados
    parser = argparse.ArgumentParser(description="Analisa o comportamento de TODOS os agentes de RL em múltiplos cenários.")
    parser.add_argument("results_dir", type=str, help="Caminho para o diretório de resultados que contém os modelos.")
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"Erro: O diretório '{args.results_dir}' não foi encontrado.")
    else:
        main(args.results_dir)