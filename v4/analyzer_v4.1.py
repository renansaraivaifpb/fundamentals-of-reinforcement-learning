# analyzer_v4.1.py
# -*- coding: utf-8 -*-
"""
Versão 4.1: Ferramenta de Análise Comportamental e Geração de Placar Final.
Gera um dashboard consolidado (todos os cenários) para cada agente.
"""
import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import argparse

# Desativa avisos comuns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from classroom_ac_env_v4 import ClassroomACEnv, ClassroomConfig, ACState
except ImportError:
    print("ERRO: Certifique-se de que 'classroom_ac_env_v4.py' está na mesma pasta.")
    exit()

def run_simulation(model: DQN, scenario: dict, env: ClassroomACEnv) -> pd.DataFrame:
    """
    Roda uma simulação de 40 passos (4h) e retorna o DataFrame.
    Esta função NÃO plota, apenas simula.
    """
    model_name = model.__class__.__name__ # Pega o nome do modelo
    print(f"  -> Simulando cenário: '{scenario['name']}'...")
    
    obs, info = env.reset(options=scenario)
    history = [info] # Armazena o estado inicial (step 0)
    temp_anterior = info['temperature']
    
    action_map = {state.value: state.name for state in ACState}
    
    simulation_steps = 40
    
    for step in range(simulation_steps): 
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        
        # Bloco de Print Detalhado (Opcional, descomente o 'if' para ativar)
        # if True:
        #     current_hour = info.get('hour', 0)
        #     action_chosen = str(info['ac_state'])
        #     # ... (resto da lógica de print)
        #     print(log_output)
        
        temp_anterior = info['temperature']
        info.update({'step': step + 1, 'action': int(action)})
        history.append(info)

        if terminated or truncated:
            break
    
    return pd.DataFrame(history)

def plot_full_dashboard(model_name: str, all_scenario_data: dict, env: ClassroomACEnv, save_dir: str) -> list:
    """
    Plota um ÚNICO gráfico com TODOS os cenários de um agente em um grid
    e salva em um arquivo.
    """
    print(f"🎨 Gerando dashboard consolidado para o agente: {model_name}...")
    
    env_config = env.config
    num_scenarios = len(all_scenario_data)
    ncols = 2
    nrows = (num_scenarios + ncols - 1) // ncols # 4 linhas para 8 cenários
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 24), squeeze=False, 
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.15})
    fig.suptitle(f'Dashboard de Comportamento do Agente: {model_name}\n(Análise das Primeiras 4 Horas)', fontsize=20, weight='bold')
    axes = axes.flatten()
    
    final_summary_data = []

    for i, (scenario_name, df) in enumerate(all_scenario_data.items()):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        # Calcula métricas
        is_in_comfort = (df['temperature'] >= env_config.temp_comfort_min) & (df['temperature'] <= env_config.temp_comfort_max)
        comfort_percentage = is_in_comfort.mean() * 100
        df['energy_consumption'] = df['action'].apply(
            lambda a: 0 if pd.isna(a) else env_config.ac_energy_consumption.get(ACState(int(a)), 0)
        )
        total_energy_kwh = (df['energy_consumption'] * env.dt).sum()
        
        final_summary_data.append({
            'agent': model_name,
            'scenario': scenario_name,
            'comfort_pct': comfort_percentage,
            'total_energy': total_energy_kwh
        })

        # Título do subplot
        title = (f"{scenario_name}\n"
                 f"(Conforto: {comfort_percentage:.1f}% | Energia: {total_energy_kwh:.2f} kWh)")
        ax1.set_title(title, fontsize=12)
        
        # Plotagem dos dados (índice vai de 0 a 40)
        ax1.plot(df.index, df['temperature'], color='royalblue', lw=2.0, label='Temperatura (°C)')
        ax2.step(df.index, df['action'], where='post', color='crimson', alpha=0.7, lw=1.5, label='Ação do AC')
        ax1.axhspan(env_config.temp_comfort_min, env_config.temp_comfort_max, color='green', alpha=0.15, label='Faixa de Conforto')
        
        # Formata o eixo X com as horas
        start_hour = int(df['hour'].iloc[0])
        tick_positions = np.arange(0, 41, 10)
        tick_labels = [f"{int((start_hour + (step * env.dt)) % 24)}h" for step in tick_positions]
        
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels)
        ax1.set_xlabel(f"Hora do Dia (Iniciando às {start_hour}h)", fontsize=10)
        
        ax1.set_ylabel("Temperatura (°C)", color='royalblue')
        ax2.set_ylabel("Ação do AC", color='crimson')
        ax2.set_yticks([0, 1, 2, 3]); ax2.set_yticklabels(['OFF', 'LOW', 'MED', 'HIGH'])
        
        lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=8)
        ax1.grid(True)

    # Esconde os eixos não utilizados
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Salva o dashboard completo
    save_path = os.path.join(save_dir, f'dashboard_completo_{model_name.replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=150)
    print(f"  -> Dashboard consolidado salvo em: {save_path}")
    plt.close(fig) # Fecha a figura para economizar memória
    
    return final_summary_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ferramenta de análise V4.1.")
    parser.add_argument("-m", "--models", type=str, help="(Opcional) Nomes de modelos para analisar, separados por vírgula.")
    parser.add_argument("--models_dir", type=str, default="models_v4_grid_search", help="Diretório dos modelos.")
    parser.add_argument("--scenarios", type=str, default="scenarios.json", help="Arquivo JSON com os cenários de teste.")
    parser.add_argument("--save_dir", type=str, default="plots_analysis_v4_1", help="Pasta para salvar os dashboards.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        with open(args.scenarios, 'r', encoding='utf-8') as f:
            scenarios_to_test = json.load(f)
        print(f"{len(scenarios_to_test)} cenários carregados de '{args.scenarios}'.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo de cenários '{args.scenarios}' não encontrado."); exit()

    model_paths = []
    if args.models:
        model_files = [m.strip() for m in args.models.split(',')]
        for model_file in model_files:
            path = model_file if os.path.isfile(model_file) else os.path.join(args.models_dir, f"{model_file}.zip")
            if os.path.exists(path): model_paths.append(path)
            else: print(f"AVISO: O arquivo de modelo não encontrado: {path}")
    else:
        model_paths = glob.glob(os.path.join(args.models_dir, '*.zip'))
        if not model_paths: print(f"ERRO: Nenhum modelo (.zip) encontrado na pasta '{args.models_dir}'."); exit()
    
    if not model_paths: print("Nenhum modelo válido para analisar."); exit()

    print(f"\nEncontrados {len(model_paths)} modelos para analisar.")

    final_summary_data_all_agents = []
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('.zip', '')
        print("\n" + "#"*80 + f"\nAnalisando modelo: {model_name}\n" + "#"*80)
        
        config_path = model_path.replace('.zip', '_config.json')
        env_params = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f: 
                data = json.load(f)
                if 'params' in data and 'env_params' in data['params']:
                    env_params = data['params']['env_params']
                else:
                    env_params = data.get('env_params', {})
            print(f"Configuração de treinamento carregada para '{model_name}'.")
        else:
            print(f"Aviso: Nenhuma configuração encontrada para '{model_name}'.")
        
        config = ClassroomConfig(**env_params)
        env = ClassroomACEnv(config=config)
        
        # Carrega o modelo UMA VEZ por agente
        try:
            model = DQN.load(model_path, env=env)
        except Exception as e:
            print(f"Erro ao carregar o modelo '{model_path}': {e}"); continue
        
        # Coleta dados de todos os cenários para este modelo
        scenario_data_for_model = {}
        for scenario in scenarios_to_test:
            # Passa o modelo JÁ CARREGADO para a simulação
            df_result = run_simulation(model, scenario, env) 
            if not df_result.empty:
                scenario_data_for_model[scenario['name']] = df_result
        
        # Plota o dashboard consolidado para o modelo
        if scenario_data_for_model:
            summary_data = plot_full_dashboard(model_name, scenario_data_for_model, env, args.save_dir)
            final_summary_data_all_agents.extend(summary_data)
    
    # Placar Final
    if len(final_summary_data_all_agents) > 0 and not args.models:
        df_final = pd.DataFrame(final_summary_data_all_agents)
        print("\n\n" + "="*80)
        print(f"📊 PLACAR FINAL - DESEMPENHO MÉDIO (Primeiras 4 Horas) 📊")
        print("="*80)
        leaderboard = df_final.groupby('agent').agg(
            comfort_pct_medio=('comfort_pct', 'mean'),
            total_energy_media=('total_energy', 'mean')
        ).round(2).sort_values('comfort_pct_medio', ascending=False)
        print(leaderboard.to_markdown())