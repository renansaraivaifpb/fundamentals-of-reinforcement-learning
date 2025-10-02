# autodoc_analyzer.py
# -*- coding: utf-8 -*-
"""
Script para Geração Automática de Relatório de Análise de Cenários

Este script carrega todos os agentes de uma pasta de resultados, executa uma
bateria de testes de cenário para cada um, e compila os resultados 
(tabelas e gráficos) em um único arquivo README.md dentro da pasta analisada.

Como usar:
$ python autodoc_analyzer.py <caminho_para_a_pasta_de_resultados>
Exemplo:
$ python autodoc_analyzer.py results_20251002_174127
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
    from ac_qlearning_agent import ACQLearningAgent
except ImportError:
    print("ERRO: Certifique-se de que os arquivos 'classroom_ac_env.py' e 'ac_qlearning_agent.py' estão no mesmo diretório.")
    exit()

def load_agent_and_config(results_dir: str, model_name: str) -> tuple:
    # (Esta função permanece a mesma)
    model_path = os.path.join(results_dir, f"{model_name}_model.pkl")
    config_path = os.path.join(results_dir, f"{model_name}_results.json")
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        return None, None
    agent = ACQLearningAgent(n_states=0, n_actions=0)
    agent.load_model(model_path)
    with open(config_path, 'r') as f:
        results = json.load(f)
        env_config_dict = results['config']['env']
    env_config = ClassroomConfig(**env_config_dict)
    return agent, env_config

def run_simulation_for_scenario(agent: ACQLearningAgent, env: ClassroomACEnvironment, scenario: dict) -> pd.DataFrame:
    # (Esta função permanece a mesma)
    state = env.reset(start_temp=scenario['start_temp'])
    env.occupancy, env.hour_of_day = scenario['occupancy'], scenario['hour']
    state = env._discretize_state()
    history = []
    for step in range(120):
        action = agent.choose_action(state, training=False)
        next_state, reward, done, info = env.step(action)
        history.append({
            'step': step, 'temperature': info['temperature'], 'ac_action': action,
            'comfort_level': 1 if info['comfort_level'] == 'COMFORTABLE' else 0,
            'energy_consumption': info['energy_consumption'], 'reward': reward
        })
        state = next_state
        if done: break
    return pd.DataFrame(history)

# ALTERAÇÃO: A função de plot agora não mostra os gráficos e retorna os caminhos dos arquivos salvos
def plot_scenario_results(all_results: dict, env_config: ClassroomConfig, save_dir: str, model_name: str) -> list:
    """Plota os resultados e retorna uma lista com os caminhos dos arquivos de imagem gerados."""
    print(f"🎨 Gerando gráficos para o agente '{model_name.title()}'...")
    sns.set_style("darkgrid")
    
    scenarios = list(all_results.items())
    chunk_size = 5
    saved_plot_paths = []

    for i in range(0, len(scenarios), chunk_size):
        chunk = scenarios[i:i + chunk_size]
        page_num = (i // chunk_size) + 1
        
        ncols = 2 if len(chunk) > 1 else 1
        nrows = (len(chunk) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 5 * nrows), squeeze=False)
        axes = axes.flatten()

        fig.suptitle(f'Análise do Agente "{model_name.title()}" - Cenários (Parte {page_num})', fontsize=18, weight='bold')

        for j, (scenario_name, df_history) in enumerate(chunk):
            ax1 = axes[j]
            ax1.plot(df_history['step'], df_history['temperature'], label='Temperatura (°C)', color='royalblue', lw=2.5)
            ax1.axhspan(env_config.temp_comfort_min, env_config.temp_comfort_max, color='green', alpha=0.15, label='Faixa de Conforto')
            ax1.set_ylabel('Temperatura (°C)', color='royalblue', fontsize=12); ax1.set_title(scenario_name, fontsize=14, weight='bold')
            ax2 = ax1.twinx()
            ax2.step(df_history['step'], df_history['ac_action'], where='post', label='Ação do AC', color='crimson', alpha=0.7, lw=2)
            ax2.set_ylabel('Ação do AC', color='crimson', fontsize=12)
            ax2.set_yticks([0, 1, 2, 3]); ax2.set_yticklabels(['OFF', 'LOW', 'MED', 'HIGH']); ax2.set_ylim(-0.5, 3.5)
            lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        for k in range(j + 1, len(axes)): axes[k].set_visible(False)
        fig.text(0.5, 0.01, 'Passos de Tempo (intervalos de 6 min)', ha='center', va='center', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        save_path = os.path.join(save_dir, f'analysis_scenarios_{model_name.replace(" ", "_")}_part{page_num}.png')
        plt.savefig(save_path, dpi=300)
        plt.close(fig) # Fecha a figura para não exibi-la na tela
        
        saved_plot_paths.append(save_path)
        print(f"  -> Gráfico salvo em: {os.path.basename(save_path)}")
        
    return saved_plot_paths

# ALTERAÇÃO: Nova função para gerar o conteúdo do README.md
def generate_readme(analysis_results: list, results_dir: str):
    """Gera um arquivo README.md compilando todas as análises."""
    print("\n📝 Gerando arquivo README.md com o relatório completo...")
    
    readme_content = f"# Relatório de Análise de Agentes - {os.path.basename(results_dir)}\n\n"
    readme_content += "Este relatório documenta o comportamento de cada agente treinado sob um conjunto de 10 cenários de teste.\n"

    for result in analysis_results:
        model_name = result['name']
        summary_table = result['summary_table_md']
        plot_paths = result['plot_paths']

        readme_content += f"\n---\n\n## 🔎 Análise do Agente: `{model_name}`\n\n"
        readme_content += "### Resumo Quantitativo de Desempenho\n\n"
        readme_content += summary_table + "\n\n"
        readme_content += "### Gráficos de Comportamento em Cenários\n\n"

        for path in plot_paths:
            # Usa o caminho relativo para a imagem funcionar no README
            relative_path = os.path.basename(path)
            readme_content += f"![Gráfico de Análise para {model_name}]({relative_path})\n"

    readme_path = os.path.join(results_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
        
    print(f"✅ Relatório completo salvo em: {readme_path}")


def main(results_dir: str):
    """Descobre, analisa todos os modelos e gera um relatório README.md."""
    
    model_files = glob.glob(os.path.join(results_dir, '*_model.pkl'))
    if not model_files:
        print(f"Nenhum arquivo '*_model.pkl' encontrado no diretório '{results_dir}'.")
        return
    model_names = sorted([os.path.basename(f).replace('_model.pkl', '') for f in model_files])
    print(f"Encontrados {len(model_names)} modelos para analisar: {', '.join(model_names)}\n")

    all_analyses = []
    scenarios = [
        {'name': '1: Manhã Fria, Sala Vazia', 'start_temp': 19.0, 'occupancy': 0, 'hour': 8},
        {'name': '2: Manhã Agradável, Sala Enchendo', 'start_temp': 23.0, 'occupancy': 25, 'hour': 9},
        # ... (lista completa de cenários) ...
        {'name': '10: Inverno Hipotético (Caso de Borda/Inação)', 'start_temp': 12.0, 'occupancy': 10, 'hour': 10},
    ]

    for model_name in model_names:
        print(f"--- Iniciando análise do agente: {model_name.upper()} ---")
        agent, env_config = load_agent_and_config(results_dir, model_name)
        if agent is None:
            print(f"Não foi possível carregar o agente {model_name}. Pulando.")
            continue
        env = ClassroomACEnvironment(env_config)
        
        all_scenario_results = {}
        for scenario in scenarios:
            df_result = run_simulation_for_scenario(agent, env, scenario)
            all_scenario_results[scenario['name']] = df_result

        plot_paths = plot_scenario_results(all_scenario_results, env_config, results_dir, model_name)

        summary_data = []
        for name, df in all_scenario_results.items():
            summary_data.append({
                'Cenário': name, 'Temp. Média (°C)': df['temperature'].mean(), 'Tempo em Conforto (%)': df['comfort_level'].mean() * 100,
                'Energia Total (kW)': df['energy_consumption'].sum(), 'Recompensa Média': df['reward'].mean(),
                'Ação Predominante': df['ac_action'].mode().iloc[0] if not df['ac_action'].empty else 0
            })
        
        df_summary = pd.DataFrame(summary_data).set_index('Cenário')
        df_summary['Ação Predominante'] = df_summary['Ação Predominante'].map({0: 'OFF', 1: 'LOW', 2: 'MED', 3: 'HIGH'})

        all_analyses.append({
            'name': model_name,
            'summary_table_md': df_summary.round(2).to_markdown(),
            'plot_paths': plot_paths
        })
        print(f"--- Análise de {model_name.upper()} concluída ---\n")

    # Gera o arquivo README.md final com todos os resultados
    generate_readme(all_analyses, results_dir)
    
    print("\n🎉 Processo de documentação automática concluído com sucesso!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gera um relatório README.md com a análise de cenários para todos os agentes em uma pasta de resultados.")
    parser.add_argument("results_dir", type=str, help="Caminho para o diretório de resultados que contém os modelos.")
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"Erro: O diretório '{args.results_dir}' não foi encontrado.")
    else:
        main(args.results_dir)