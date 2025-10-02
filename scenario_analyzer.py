# analyzer.py
# -*- coding: utf-8 -*-
"""
Script Unificado para Análise de Agentes de RL e Geração de Relatórios

Este script é a ferramenta central para analisar os resultados dos treinamentos.
Ele pode operar em dois modos:

1. MODO RELATÓRIO COMPLETO (Padrão):
   - Analisa TODOS os agentes na pasta de resultados.
   - Gera um arquivo README.md com gráficos e tabelas para cada um.
   - Uso: python analyzer.py <caminho_para_a_pasta_de_resultados>

2. MODO ANÁLISE DETALHADA (Interativo):
   - Analisa um ÚNICO agente especificado.
   - Exibe os gráficos de cenário interativamente na tela.
   - Uso: python analyzer.py <caminho_para_a_pasta_de_resultados> --model <nome_do_modelo>

Exemplo 1: Gerar relatório completo para a pasta 'results_20251002_174127'
$ python analyzer.py results_20251002_174127

Exemplo 2: Analisar interativamente apenas o agente 'baseline'
$ python analyzer.py results_20251002_174127 --model baseline
"""

import os
import json
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig
    from ac_qlearning_agent import ACQLearningAgent
except ImportError:
    print("ERRO: Certifique-se de que os arquivos 'classroom_ac_env.py' e 'ac_qlearning_agent.py' estão no mesmo diretório.")
    exit()

def load_agent_and_config(results_dir: str, model_name: str) -> tuple:
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
    print(f"  -> Simulando cenário: '{scenario['name']}'...")
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

# ALTERAÇÃO: Adicionado parâmetro 'interactive' para controlar a exibição dos plots
def plot_scenario_results(all_results: dict, env_config: ClassroomConfig, save_dir: str, model_name: str, interactive: bool = False) -> list:
    print(f"\n🎨 Gerando visualizações para o agente '{model_name.title()}'...")
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
        saved_plot_paths.append(save_path)
        print(f"  -> Gráfico salvo em: {os.path.basename(save_path)}")

        if interactive:
            plt.show() # Mostra o gráfico na tela apenas no modo interativo
        
        plt.close(fig)
        
    return saved_plot_paths

def generate_readme(analysis_results: list, results_dir: str):
    print("\n📝 Gerando arquivo README.md com o relatório completo...")
    readme_content = f"# Relatório de Análise de Agentes - {os.path.basename(results_dir)}\n\n"
    readme_content += "Este relatório documenta o comportamento de cada agente treinado sob um conjunto de 10 cenários de teste.\n"
    for result in analysis_results:
        readme_content += f"\n---\n\n## 🔎 Análise do Agente: `{result['name']}`\n\n"
        readme_content += "### Resumo Quantitativo\n\n" + result['summary_table_md'] + "\n\n"
        readme_content += "### Gráficos de Comportamento\n\n"
        for path in result['plot_paths']:
            readme_content += f"![Gráfico de Análise para {result['name']}]({os.path.basename(path)})\n"
    readme_path = os.path.join(results_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f: f.write(readme_content)
    print(f"✅ Relatório completo salvo em: {readme_path}")

# ALTERAÇÃO: Esta função agora retorna os resultados e controla a interatividade
def analyze_single_agent(results_dir: str, model_name: str, interactive: bool = False):
    """Orquestra a análise de um único agente."""
    
    print("\n" + "="*80)
    print(f"INICIANDO ANÁLISE DO AGENTE: {model_name.upper()}")
    print("="*80)

    agent, env_config = load_agent_and_config(results_dir, model_name)
    if agent is None:
        print(f"Não foi possível carregar o agente {model_name}. Pulando.")
        return None

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

    all_scenario_results = {s['name']: run_simulation_for_scenario(agent, env, s) for s in scenarios}
    plot_paths = plot_scenario_results(all_scenario_results, env_config, results_dir, model_name, interactive=interactive)

    summary_data = []
    for name, df in all_scenario_results.items():
        summary_data.append({
            'Cenário': name, 'Temp. Média (°C)': df['temperature'].mean(), 'Tempo em Conforto (%)': df['comfort_level'].mean() * 100,
            'Energia Total (kW)': df['energy_consumption'].sum(), 'Recompensa Média': df['reward'].mean(),
            'Ação Predominante': df['ac_action'].mode().iloc[0] if not df['ac_action'].empty else 0
        })
    df_summary = pd.DataFrame(summary_data).set_index('Cenário')
    df_summary['Ação Predominante'] = df_summary['Ação Predominante'].map({0: 'OFF', 1: 'LOW', 2: 'MED', 3: 'HIGH'})
    summary_table_md = df_summary.round(2).to_markdown()

    if interactive:
        print("\n" + "="*80 + f"\n📊 RESUMO QUANTITATIVO DO AGENTE '{model_name.title()}'\n" + "="*80)
        print(summary_table_md)
        
    return {'name': model_name, 'summary_table_md': summary_table_md, 'plot_paths': plot_paths}

# ALTERAÇÃO: A função main agora decide o modo de operação (single vs all)
def main(args):
    """Decide se roda a análise para um agente ou para todos."""
    
    if args.model:
        # --- MODO ANÁLISE DETALHADA ---
        analyze_single_agent(args.results_dir, args.model, interactive=True)
        print("\n🎉 Análise detalhada concluída!")
    else:
        # --- MODO RELATÓRIO COMPLETO ---
        model_files = glob.glob(os.path.join(args.results_dir, '*_model.pkl'))
        if not model_files:
            print(f"Nenhum arquivo '*_model.pkl' encontrado no diretório '{args.results_dir}'.")
            return
        model_names = sorted([os.path.basename(f).replace('_model.pkl', '') for f in model_files])
        print(f"Encontrados {len(model_names)} modelos para gerar relatório: {', '.join(model_names)}")
        
        all_analyses = [analyze_single_agent(args.results_dir, name) for name in model_names]
        
        generate_readme([res for res in all_analyses if res is not None], args.results_dir)
        print("\n🎉 Processo de documentação automática concluído com sucesso!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ferramenta unificada para análise de agentes e geração de relatórios.")
    parser.add_argument("results_dir", type=str, help="Caminho para o diretório de resultados.")
    parser.add_argument("-m", "--model", type=str, help="(Opcional) Nome de um modelo específico para analisar interativamente.")
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"Erro: O diretório '{args.results_dir}' não foi encontrado.")
    else:
        main(args)