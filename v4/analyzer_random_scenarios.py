# analyzer_random_scenarios.py
# -*- coding: utf-8 -*-
"""
Versão 4: Ferramenta de Análise com Cenários Aleatórios.
Testa os agentes em N cenários gerados aleatoriamente.
"""
import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import argparse
import random

# Desativa avisos comuns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from classroom_ac_env_v4 import ClassroomACEnv, ClassroomConfig, ACState
except ImportError:
    print("ERRO: Certifique-se de que 'classroom_ac_env_v4.py' está na mesma pasta.")
    exit()

def generate_random_scenario(scenario_index: int) -> dict:
    """Gera um único cenário com condições iniciais aleatórias."""
    start_temp = round(random.uniform(18.0, 32.0), 1)
    occupancy = random.randint(0, 30)
    hour = random.randint(0, 23) # Começa em qualquer hora do dia
    
    return {
        "name": f"Aleatório {scenario_index}: {start_temp}°C, {occupancy} Pessoas, {hour}h",
        "start_temp": start_temp,
        "occupancy": occupancy,
        "hour": hour
    }

def run_and_analyze_simulation(model_path: str, scenario: dict, env: ClassroomACEnv) -> tuple:
    """
    Roda uma simulação, plota o gráfico focado em 7h-22h
    e retorna as métricas de desempenho.
    """
    model_name = os.path.basename(model_path).replace('.zip', '')
    print(f"\n--- Analisando o cenário: '{scenario['name']}' ---")
    
    try:
        model = DQN.load(model_path, env=env)
    except Exception as e:
        print(f"Erro ao carregar o modelo '{model_path}': {e}"); return (None, None)

    obs, info = env.reset(options=scenario)
    history = [info]
    
    # Simula por 240 passos (24h)
    for step in range(240): 
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        info.update({'step': step + 1, 'action': int(action)})
        history.append(info)
        if terminated or truncated:
            break
    
    df = pd.DataFrame(history)
    
    # --- ANÁLISE E PLOTAGEM ---
    
    # Filtra o DataFrame para o período de 7h às 22h
    df_filtered = df[(df['hour'] >= 7) & (df['hour'] <= 22)].copy()
    
    if df_filtered.empty:
        print(f"  -> Nenhum dado encontrado no período de análise (7h-22h).")
        plt.close('all')
        return (None, None)

    env_config = env.config
    
    is_in_comfort = (df_filtered['temperature'] >= env_config.temp_comfort_min) & (df_filtered['temperature'] <= env_config.temp_comfort_max)
    comfort_percentage = is_in_comfort.mean() * 100
    
    df_filtered['energy_consumption'] = df_filtered['action'].apply(
        lambda a: 0 if pd.isna(a) else env_config.ac_energy_consumption.get(ACState(int(a)), 0)
    )
    total_energy_kwh = (df_filtered['energy_consumption'] * env.dt).sum()

    # Plotagem
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax2 = ax1.twinx()

    title = (f"Agente: {model_name} | Cenário: {scenario['name']}\n"
             f"Análise do Período (7h às 22h)\n"
             f"(Tempo em Conforto: {comfort_percentage:.1f}% | Energia Total: {total_energy_kwh:.2f} kWh)")
    ax1.set_title(title, fontsize=16)
    
    df_plot = df_filtered.reset_index(drop=True)
    
    ax1.plot(df_plot.index, df_plot['temperature'], color='royalblue', lw=2.5, label='Temperatura (°C)')
    ax2.step(df_plot.index, df_plot['action'], where='post', color='crimson', alpha=0.7, lw=2, label='Ação do AC')
    ax1.axhspan(env_config.temp_comfort_min, env_config.temp_comfort_max, color='green', alpha=0.15, label='Faixa de Conforto')
    
    # Lógica de rótulos de hora (ticks)
    num_ticks = min(10, len(df_plot))
    tick_indices = np.linspace(0, len(df_plot) - 1, num_ticks, dtype=int)
    tick_labels = [f"{int(h)}h" for h in df_plot['hour'].iloc[tick_indices]]
    
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(tick_labels)
    ax1.set_xlabel(f"Hora do Dia (Período de Análise: 7h-22h)", fontsize=12)
    
    ax2.set_ylabel("Ação do AC", color='crimson')
    ax2.set_yticks([0, 1, 2, 3]); ax2.set_yticklabels(['OFF', 'LOW', 'MED', 'HIGH'])
    
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.grid(True)
    plt.show()

    return comfort_percentage, total_energy_kwh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ferramenta de análise V4.1 com cenários aleatórios.")
    parser.add_argument("-m", "--models", type=str, help="Nomes de modelos para analisar, separados por vírgula.")
    parser.add_argument("--models_dir", type=str, default="models_v4_grid_search", help="Diretório dos modelos.")
    parser.add_argument("-n", "--num_scenarios", type=int, default=5, help="Número de cenários aleatórios para gerar.")
    args = parser.parse_args()

    # --- MELHORIA: GERA CENÁRIOS ALEATÓRIOS ---
    print(f"Gerando {args.num_scenarios} cenários de teste aleatórios...")
    scenarios_to_test = [generate_random_scenario(i+1) for i in range(args.num_scenarios)]

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

    final_summary_data = []
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
        
        for scenario in scenarios_to_test:
            metrics = run_and_analyze_simulation(model_path, scenario, env)
            
            if metrics is not None and metrics[0] is not None:
                comfort_pct, total_energy = metrics
                
                final_summary_data.append({
                    'agent': model_name,
                    'scenario': scenario['name'],
                    'comfort_pct': comfort_pct,
                    'total_energy': total_energy
                })
    
    if len(final_summary_data) > 0 and not args.models:
        df_final = pd.DataFrame(final_summary_data)
        print("\n\n" + "="*80)
        print(f"📊 PLACAR FINAL - DESEMPENHO MÉDIO (Cenários Aleatórios, 7h-22h) 📊")
        print("="*80)
        leaderboard = df_final.groupby('agent').agg(
            comfort_pct_medio=('comfort_pct', 'mean'),
            total_energy_media=('total_energy', 'mean')
        ).round(2).sort_values('comfort_pct_medio', ascending=False)
        print(leaderboard.to_markdown())