# analyzer.py
# -*- coding: utf-8 -*-
"""
Script Unificado para Análise de Agentes de RL e Geração de Relatórios (v6 - Gráfico de Treinamento Reintegrado)

Melhorias:
- Reintroduz a geração do gráfico comparativo de desempenho durante o treinamento (2x2).
- Mantém a geração de dashboards de cenários individuais por agente.
- Código organizado para gerar primeiro a análise geral e depois os detalhes.
"""

import os
import json
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple

# --- Bloco de Importação (sem alterações) ---
try:
    from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig
    from ac_qlearning_agent import ACQLearningAgent, QLearningConfig
    from ac_dqn_agent import ACDQNAgent, DQNConfig
    ENV_CLASSES_AVAILABLE = True
    print("[analyzer.py] Módulos de ambiente/agente importados com sucesso.")
except ImportError as e:
    print(f"[analyzer.py] AVISO: Não foi possível importar os módulos do agente/ambiente: {e}.")
    ENV_CLASSES_AVAILABLE = False


# --- NOVA FUNÇÃO ADICIONADA ---
def plot_comparative_training_history(results_data: Dict[str, Dict], save_dir: str):
    """
    Gera um gráfico 2x2 comparando o histórico de treinamento de todos os agentes.
    """
    print("🎨 Gerando gráfico comparativo de desempenho do treinamento...")
    sns.set_style("darkgrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Desempenho Durante o Treinamento', fontsize=20, weight='bold')
    axes = axes.flatten()
    
    palette = sns.color_palette("viridis", len(results_data))
    
    metrics_config = {
        'episode_rewards': {'ax_idx': 0, 'title': 'Recompensa Média (Média Móvel)', 'ylabel': 'Recompensa Média'},
        'comfort_percentages': {'ax_idx': 1, 'title': 'Conforto Térmico Médio (%) (Média Móvel)', 'ylabel': 'Conforto Térmico Médio'},
        'energy_consumptions': {'ax_idx': 2, 'title': 'Consumo de Energia (kW) (Média Móvel)', 'ylabel': 'Consumo de Energia'},
        'epsilon_values': {'ax_idx': 3, 'title': 'Decaimento do Epsilon (Dados Brutos)', 'ylabel': 'Decaimento do Epsilon'}
    }

    for i, (model_name, data) in enumerate(results_data.items()):
        history = data.get('training_history', {})
        for metric, config in metrics_config.items():
            ax = axes[config['ax_idx']]
            if metric in history:
                values = history[metric]
                window = 100 if 'rewards' in metric or 'comfort' in metric or 'energy' in metric else 1
                
                if len(values) >= window and window > 1:
                    moving_avg = np.convolve(values, np.ones(window) / window, mode='valid')
                    ax.plot(range(window - 1, len(values)), moving_avg, label=model_name, color=palette[i], lw=2)
                else:
                    ax.plot(values, label=model_name, color=palette[i], lw=2)

    for metric, config in metrics_config.items():
        ax = axes[config['ax_idx']]
        ax.set_title(config['title'], fontsize=14)
        ax.set_xlabel('Episódios', fontsize=12)
        ax.set_ylabel(config['ylabel'], fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, 'analysis_training_performance.png')
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  -> Gráfico de treinamento salvo em: {os.path.basename(save_path)}")


# --- Funções de análise de cenário (sem alterações) ---
def load_agent_from_results(results_dir: str, model_name: str) -> Optional[Tuple[Any, ClassroomConfig, Dict]]:
    if not ENV_CLASSES_AVAILABLE:
        print("[analyzer.py] A função load_agent_from_results foi pulada porque os módulos não estão disponíveis.")
        return None
    
    config_path = os.path.join(results_dir, f"{model_name}_results.json")
    if not os.path.exists(config_path):
        print(f"  -> [DEBUG] Arquivo de config '{config_path}' não encontrado.")
        return None

    with open(config_path, 'r') as f: results_data = json.load(f)
    
    agent_type = results_data.get('agent_type', 'q_learning')
    model_ext = '.pth' if agent_type == 'dqn' else '.pkl'
    model_path = os.path.join(results_dir, f"{model_name}_model{model_ext}")
    
    print(f"  -> [DEBUG] Procurando por arquivo de modelo: '{model_path}'")
    
    if not os.path.exists(model_path):
        print(f"  -> [DEBUG] FALHA: Arquivo de modelo não encontrado.")
        return None
        
    print(f"  -> [DEBUG] SUCESSO: Arquivo de modelo encontrado. Tentando carregar...")
    try:
        env_config = ClassroomConfig(**results_data['config']['env'])
        if agent_type == 'q_learning':
            agent = ACQLearningAgent(0, 0); agent.load_model(model_path)
        else:
            agent_config = DQNConfig(**results_data['config']['agent'])
            agent = ACDQNAgent(agent_config); agent.load_model(model_path)
        
        print(f"  -> Agente '{model_name}' ({agent_type}) carregado com sucesso.")
        return agent, env_config, results_data
    except Exception as e:
        print(f"  -> ERRO ao carregar o agente '{model_name}': {e}")
        return None

def run_simulation(agent: Any, env: ClassroomACEnvironment, scenario: Dict) -> pd.DataFrame:
    # ... (código inalterado) ...
    agent_type = 'dqn' if isinstance(agent, ACDQNAgent) else 'q_learning'
    state_disc, info = env.reset(options=scenario)
    state_cont = info['continuous_state']
    history = []
    for step in range(120):
        state = state_cont if agent_type == 'dqn' else state_disc
        action = agent.choose_action(state, training=False)
        state_disc, reward, done, _, info = env.step(action)
        state_cont = info['continuous_state']
        history.append({'step': step, 'temperature': info['temperature'], 'ac_action': action, 'comfort': 1 if info['comfort_level'] == 'COMFORTABLE' else 0, 'energy': info['energy_consumption'], 'reward': reward})
        if done: break
    return pd.DataFrame(history)

def plot_scenario_dashboard(model_name: str, all_results: Dict[str, pd.DataFrame], env: ClassroomACEnvironment, save_dir: str) -> List[str]:
    # ... (código inalterado) ...
    print(f"🎨 Gerando dashboard consolidado para '{model_name}'...")
    sns.set_style("whitegrid")
    num_scenarios = len(all_results)
    ncols = 2
    nrows = (num_scenarios + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15 * ncols, 6 * nrows), squeeze=False)
    fig.suptitle(f'Dashboard de Comportamento do Agente: {model_name.title()}', fontsize=24, weight='bold')
    axes = axes.flatten()
    for i, (scenario_name, df) in enumerate(all_results.items()):
        ax1 = axes[i]
        ax2 = ax1.twinx()
        ax1.plot(df['step'], df['temperature'], color='royalblue', lw=2, label='Temperatura (°C)')
        ax2.step(df['step'], df['ac_action'], where='post', color='crimson', alpha=0.7, lw=1.5, label='Ação do AC')
        ax1.axhspan(env.config.temp_comfort_min, env.config.temp_comfort_max, color='green', alpha=0.15, label='Faixa de Conforto')
        ax1.set_title(scenario_name, fontsize=14)
        ax1.set_ylabel('Temperatura (°C)', color='royalblue')
        ax2.set_ylabel('Ação do AC', color='crimson')
        ax2.set_yticks([0, 1, 2, 3]); ax2.set_yticklabels(['OFF', 'LOW', 'MED', 'HIGH'])
        ax1.grid(True, axis='y', linestyle='--')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    for j in range(num_scenarios, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(save_dir, f'dashboard_scenarios_{model_name.replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  -> Dashboard salvo em: {os.path.basename(save_path)}")
    return [save_path]

def generate_readme(model_analyses: List[Dict], results_dir: str):
    # ... (código inalterado) ...
    print("📝 Gerando relatório README.md...")
    readme = f"# Relatório de Análise de Agentes - {os.path.basename(results_dir)}\n\n"
    training_plot_path = "analysis_training_performance.png"
    if os.path.exists(os.path.join(results_dir, training_plot_path)):
        readme += "## 📈 Desempenho Geral Durante o Treinamento\n\n"
        readme += f"![Desempenho no Treinamento]({training_plot_path})\n\n"
    for result in model_analyses:
        readme += f"---\n\n## 🔎 Agente: `{result['name']}`\n\n"
        readme += "<details>\n<summary><strong>Clique para ver Parâmetros de Configuração</strong></summary>\n\n"
        readme += "#### Parâmetros do Agente\n" + result['agent_params_md'] + "\n\n"
        readme += "#### Parâmetros do Ambiente\n" + result['env_params_md'] + "\n\n</details>\n\n"
        readme += "### Dashboard de Comportamento em Cenários\n\n"
        for path in result['plot_paths']:
            readme += f"![Dashboard de Análise]({os.path.basename(path)})\n"
    readme_path = os.path.join(results_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f: f.write(readme)
    print(f"✅ Relatório completo salvo em: {readme_path}")

def main(args: argparse.Namespace):
    """Função principal que orquestra a análise completa."""
    if not os.path.isdir(args.results_dir):
        print(f"ERRO: O diretório '{args.results_dir}' não foi encontrado."); return
    try:
        with open('scenarios.json', 'r') as f: scenarios = json.load(f)
        print(f"Cenários de teste carregados de 'scenarios.json'.")
    except FileNotFoundError:
        print("ERRO: Arquivo 'scenarios.json' não encontrado."); return

    all_json_files = sorted(glob.glob(os.path.join(args.results_dir, '*_results.json')))
    if not all_json_files:
        print("Nenhum arquivo de resultado encontrado."); return
        
    # Carrega todos os dados primeiro
    results_data = {
        os.path.basename(f).replace('_results.json', '').replace('_', ' ').title(): json.load(open(f, 'r'))
        for f in all_json_files
    }
    
    # --- ADIÇÃO: Chamar a nova função de plotagem do treinamento ---
    plot_comparative_training_history(results_data, args.results_dir)

    # Lógica para filtrar quais modelos analisar (se especificado)
    models_to_analyze = results_data.keys()
    if args.models:
        models_to_analyze = [name.strip() for name in args.models.split(',')]
    
    model_analyses = []
    for model_name in models_to_analyze:
        if model_name not in results_data:
            print(f"AVISO: Modelo '{model_name}' não encontrado nos resultados. Pulando.")
            continue
            
        print("\n" + "="*60 + f"\nAnalisando Cenários para: {model_name}\n" + "="*60)
        load_result = load_agent_from_results(args.results_dir, model_name.lower().replace(' ', '_'))
        if load_result is None: continue
        agent, env_config, agent_results_data = load_result
        
        env = ClassroomACEnvironment(env_config)
        scenario_results = {s['name']: run_simulation(agent, env, s) for s in scenarios}
        
        agent_params_md = pd.DataFrame.from_dict(agent_results_data['config']['agent'], orient='index', columns=['Valor']).to_markdown()
        env_params_md = pd.DataFrame.from_dict(agent_results_data['config']['env'], orient='index', columns=['Valor']).to_markdown()
        
        plot_paths = plot_scenario_dashboard(model_name, scenario_results, env, args.results_dir)
        
        model_analyses.append({
            'name': model_name, 'agent_params_md': agent_params_md,
            'env_params_md': env_params_md, 'plot_paths': plot_paths
        })

    if model_analyses and not args.models:
        generate_readme(model_analyses, args.results_dir)
        print("\n🎉 Processo de documentação automática concluído com sucesso!")
    elif model_analyses:
        print(f"\n🎉 Análise para os modelos especificados concluída!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ferramenta de Análise e Geração de Relatórios para Agentes de RL.")
    parser.add_argument("results_dir", type=str, help="Caminho para o diretório de resultados.")
    parser.add_argument("-m", "--models", type=str, help="(Opcional) Nomes de modelos específicos para analisar, separados por vírgula.")
    main(parser.parse_args())