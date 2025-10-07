# analyzer.py
# -*- coding: utf-8 -*-
"""
Script Unificado para Análise de Agentes de RL e Geração de Relatórios (v5 - Dashboards Consolidados)

Melhorias:
- Gera um único arquivo de imagem (dashboard) por agente, contendo todos os cenários.
- Código de plotagem refatorado para maior clareza e eficiência.
"""

import os
import json
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dataclasses
from typing import Dict, Any, List, Optional, Tuple

try:
    from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig
    from ac_qlearning_agent import ACQLearningAgent, QLearningConfig
    from ac_dqn_agent import ACDQNAgent, DQNConfig
    ENV_CLASSES_AVAILABLE = True
except ImportError as e:
    print(f"AVISO: Não foi possível importar os módulos do agente/ambiente: {e}. A simulação será pulada.")
    ENV_CLASSES_AVAILABLE = False

def load_agent_from_results(results_dir: str, model_name: str) -> Optional[Tuple[Any, ClassroomConfig, Dict]]:
    """Carrega dinamicamente um agente (QL ou DQN) e suas configurações."""
    # (Esta função permanece a mesma da versão anterior)
    if not ENV_CLASSES_AVAILABLE: return None
    config_path = os.path.join(results_dir, f"{model_name}_results.json")
    if not os.path.exists(config_path):
        print(f"  -> Arquivo de resultados '{config_path}' não encontrado.")
        return None
    with open(config_path, 'r') as f: results_data = json.load(f)
    agent_type = results_data.get('agent_type', 'q_learning')
    model_ext = '.pth' if agent_type == 'dqn' else '.pkl'
    model_path = os.path.join(results_dir, f"{model_name}_model{model_ext}")
    if not os.path.exists(model_path):
        print(f"  -> Arquivo de modelo '{model_path}' não encontrado.")
        return None
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
    """Executa uma simulação para um único cenário."""
    # (Esta função permanece a mesma da versão anterior)
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

# --- MELHORIA: Função refatorada para gerar um único dashboard por agente ---
def plot_scenario_dashboard(model_name: str, all_results: Dict[str, pd.DataFrame], env: ClassroomACEnvironment, save_dir: str) -> List[str]:
    """Gera uma única imagem (dashboard) com todos os cenários para um agente."""
    print(f"🎨 Gerando dashboard consolidado para '{model_name}'...")
    sns.set_style("whitegrid")
    
    num_scenarios = len(all_results)
    # Define o layout do grid (ex: para 8 cenários, cria 4 linhas e 2 colunas)
    ncols = 2
    nrows = (num_scenarios + ncols - 1) // ncols
    
    # Cria a figura e os subplots uma única vez
    fig, axes = plt.subplots(nrows, ncols, figsize=(15 * ncols, 6 * nrows), squeeze=False)
    fig.suptitle(f'Dashboard de Comportamento do Agente: {model_name.title()}', fontsize=24, weight='bold')
    axes = axes.flatten() # Transforma a matriz de eixos em um array 1D para fácil iteração

    for i, (scenario_name, df) in enumerate(all_results.items()):
        ax1 = axes[i]
        
        # Painel de Controle (Temperatura vs. Ação)
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

    # Esconde eixos não utilizados se o número de cenários não preencher o grid perfeitamente
    for j in range(num_scenarios, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Salva a figura inteira como um único arquivo
    save_path = os.path.join(save_dir, f'dashboard_scenarios_{model_name.replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=150) # dpi menor para arquivos grandes
    plt.close(fig)
    
    print(f"  -> Dashboard salvo em: {os.path.basename(save_path)}")
    return [save_path] # Retorna uma lista contendo o caminho do único arquivo salvo

def generate_readme(model_analyses: List[Dict], results_dir: str):
    """Gera um arquivo README.md compilando todas as análises."""
    print("📝 Gerando relatório README.md...")
    readme = f"# Relatório de Análise de Agentes - {os.path.basename(results_dir)}\n\n"
    
    # Adiciona o gráfico de performance do treinamento no topo
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
        # A lista 'plot_paths' agora contém apenas um caminho, para o dashboard
        for path in result['plot_paths']:
            readme += f"![Dashboard de Análise]({os.path.basename(path)})\n"
    
    readme_path = os.path.join(results_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme)
    print(f"✅ Relatório completo salvo em: {readme_path}")

def main(args: argparse.Namespace):
    """Função principal que orquestra a análise completa."""
    if not os.path.isdir(args.results_dir):
        print(f"ERRO: O diretório '{args.results_dir}' não foi encontrado.")
        return
        
    try:
        with open('scenarios.json', 'r') as f:
            scenarios = json.load(f)
        print(f"Cenários de teste carregados de 'scenarios.json'.")
    except FileNotFoundError:
        print("ERRO: Arquivo 'scenarios.json' não encontrado.")
        return

    # --- MELHORIA: Filtra quais modelos analisar se o argumento --model for usado ---
    all_json_files = sorted(glob.glob(os.path.join(args.results_dir, '*_results.json')))
    files_to_process = []
    if args.models:
        model_names_to_find = [name.strip() for name in args.models.split(',')]
        for name in model_names_to_find:
            path = os.path.join(args.results_dir, f"{name.replace(' ', '_').lower()}_results.json")
            if os.path.exists(path):
                files_to_process.append(path)
            else:
                print(f"AVISO: Não foi encontrado um arquivo de resultado para o modelo '{name}'")
    else:
        files_to_process = all_json_files
        
    if not files_to_process:
        print("Nenhum modelo para analisar.")
        return

    model_analyses = []
    for json_path in files_to_process:
        model_name = os.path.basename(json_path).replace('_results.json', '').replace('_', ' ').title()
        print("\n" + "="*60 + f"\nAnalisando: {model_name}\n" + "="*60)
        
        load_result = load_agent_from_results(args.results_dir, os.path.basename(json_path).replace('_results.json', ''))
        if load_result is None: continue
        agent, env_config, results_data = load_result
        
        env = ClassroomACEnvironment(env_config)
        scenario_results = {s['name']: run_simulation(agent, env, s) for s in scenarios}
        
        agent_params_md = pd.DataFrame.from_dict(results_data['config']['agent'], orient='index', columns=['Valor']).to_markdown()
        env_params_md = pd.DataFrame.from_dict(results_data['config']['env'], orient='index', columns=['Valor']).to_markdown()
        
        # Chama a nova função de plotagem de dashboard
        plot_paths = plot_scenario_dashboard(model_name, scenario_results, env, args.results_dir)
        
        model_analyses.append({
            'name': model_name, 'agent_params_md': agent_params_md,
            'env_params_md': env_params_md, 'plot_paths': plot_paths
        })

    if model_analyses and not args.models: # Só gera o README se estiver analisando todos
        generate_readme(model_analyses, args.results_dir)
        print("\n🎉 Processo de documentação automática concluído com sucesso!")
    elif model_analyses:
        print(f"\n🎉 Análise para os modelos especificados concluída!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ferramenta de Análise e Geração de Relatórios para Agentes de RL.")
    parser.add_argument("results_dir", type=str, help="Caminho para o diretório de resultados.")
    
    # --- CORREÇÃO: Trocando o hífen por um underscore ---
    parser.add_argument(
        "-m", "--models", type=str, 
        help="(Opcional) Nomes de modelos específicos para analisar, separados por vírgula (ex: 'DQN Baseline,DQN Foco Alto'). Se não for fornecido, analisa todos."
    )
    main(parser.parse_args())