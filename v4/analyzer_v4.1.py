# analyzer_v4.1.py
# -*- coding: utf-8 -*-
"""
Versão 4.1: Ferramenta de Análise Comportamental e Geração de Placar Final.
Análise focada no período das 7h às 22h.
"""
import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import argparse

try:
    from classroom_ac_env_v4 import ClassroomACEnv, ClassroomConfig, ACState
except ImportError:
    print("ERRO: Certifique-se de que 'classroom_ac_env_v4.py' está na mesma pasta.")
    exit()

def run_sb3_simulation(model_path: str, scenario: dict, env: ClassroomACEnv) -> pd.DataFrame:
    """
    Roda uma simulação completa de 24h para um modelo e cenário, e retorna o histórico.
    """
    model_name = os.path.basename(model_path).replace('.zip', '')
    print(f"\n--- Simulando cenário: '{scenario['name']}' para o agente: {model_name} ---")
    
    try:
        model = DQN.load(model_path, env=env)
    except Exception as e:
        print(f"Erro ao carregar o modelo '{model_path}': {e}"); return pd.DataFrame()

    obs, info = env.reset(options=scenario)
    history = [info]
    
    # Roda por 240 passos para simular um ciclo diário completo
    for step in range(239):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        action_map = {0: 'OFF', 1: 'LOW', 2: 'MEDIUM', 3: 'HIGH'}
        info.update({
            'step': step + 1, 
            'action': int(action)
        })
        history.append(info)
        print(f"\n--- Passo {step + 1} ---")
        print(f"Temperatura Externa: {info['debug_outdoor_temp']:.2f}°C")
        print(f"Temperatura Interna: {info['temperature']:.2f}°C")
        print(f"Ganho de Calor Total: {info['debug_total_heat_gain']:.2f} kW")
        print(f"Efeito de Resfriamento: {info['debug_cooling_effect']:.2f} kW")
        print(f"Calor Líquido (Net Heat): {info['debug_net_heat']:.2f} kW")
        print(f"Variação de Temperatura: {info['debug_temp_change']:.2f}°C")
        print(f"Nível de Ruído: {info['debug_noise']:.2f} dB")
        print(f"Ação do AC: {action_map.get(info['action'], 'UNKNOWN')}")

        if terminated or truncated:
            break
            
    return pd.DataFrame(history)

def plot_and_analyze_simulation(df: pd.DataFrame, model_name: str, scenario_name: str, env: ClassroomACEnv) -> tuple:
    """
    Filtra os dados pelo horário de interesse (7h-22h), plota o gráfico e retorna as métricas.
    """
    # Filtra o DataFrame para o período de 7h às 22h
    df_filtered = df[(df['hour'] >= 7) & (df['hour'] <= 22)].copy()
    
    if df_filtered.empty:
        print(f"  -> Nenhum dado encontrado no período de análise (7h-22h) para o cenário '{scenario_name}'.")
        return 0, 0

    env_config = env.config
    
    # Calcula métricas com base nos dados filtrados
    is_in_comfort = (df_filtered['temperature'] >= env_config.temp_comfort_min) & (df_filtered['temperature'] <= env_config.temp_comfort_max)
    comfort_percentage = is_in_comfort.mean() * 100
    
    df_filtered['energy_consumption'] = df_filtered['action'].apply(
        lambda a: 0 if pd.isna(a) else env_config.ac_energy_consumption.get(ACState(int(a)), 0)
    )
    total_energy_kwh = (df_filtered['energy_consumption'] * env.dt).sum()

    # Plotagem
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax2 = ax1.twinx()

    title = (f"Agente: {model_name} | Cenário: {scenario_name}\n"
             f"Análise do Período (7h às 22h)\n"
             f"(Tempo em Conforto: {comfort_percentage:.1f}% | Energia Total: {total_energy_kwh:.2f} kWh)")
    ax1.set_title(title, fontsize=16)
    
    ax1.plot(df_filtered['step'], df_filtered['temperature'], color='royalblue', lw=2.5, label='Temperatura (°C)')
    ax2.step(df_filtered['step'], df_filtered['action'], where='post', color='crimson', alpha=0.7, lw=2, label='Ação do AC')
    ax1.axhspan(env_config.temp_comfort_min, env_config.temp_comfort_max, color='green', alpha=0.15, label='Faixa de Conforto')
    
    ax1.set_xlabel("Passos de Tempo (6 min)"); ax1.set_ylabel("Temperatura (°C)", color='royalblue')
    ax2.set_ylabel("Ação do AC", color='crimson')
    ax2.set_yticks([0, 1, 2, 3]); ax2.set_yticklabels(['OFF', 'LOW', 'MED', 'HIGH'])
    
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.grid(True)
    plt.show()

    return comfort_percentage, total_energy_kwh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ferramenta de análise V4.1.")
    parser.add_argument("-m", "--models", type=str, help="(Opcional) Nomes de modelos para analisar, separados por vírgula.")
    parser.add_argument("--models_dir", type=str, default="models_v4", help="Diretório dos modelos.")
    parser.add_argument("--scenarios", type=str, default="scenarios.json", help="Arquivo JSON com os cenários de teste.")
    args = parser.parse_args()

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

    final_summary_data = []
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('.zip', '')
        print("\n" + "#"*80 + f"\nAnalisando modelo: {model_name}\n" + "#"*80)
        
        config_path = model_path.replace('.zip', '_config.json')
        env_params = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f: data = json.load(f); env_params = data.get('env_params', {})
            print(f"Configuração de treinamento carregada para '{model_name}'.")
        else:
            print(f"Aviso: Nenhuma configuração encontrada para '{model_name}'.")
        
        config = ClassroomConfig(**env_params)
        env = ClassroomACEnv(config=config)
        
        for scenario in scenarios_to_test:
            df_full_day = run_sb3_simulation(model_path, scenario, env)
            
            if not df_full_day.empty:
                comfort_pct, total_energy = plot_and_analyze_simulation(df_full_day, model_name, scenario['name'], env)
                
                final_summary_data.append({
                    'agent': model_name,
                    'scenario': scenario['name'],
                    'comfort_pct': comfort_pct,
                    'total_energy': total_energy
                })
    
    if len(final_summary_data) > 0 and not args.models:
        df_final = pd.DataFrame(final_summary_data)
        print("\n\n" + "="*80)
        print("📊 PLACAR FINAL - DESEMPENHO MÉDIO (7h às 22h) 📊")
        print("="*80)
        leaderboard = df_final.groupby('agent').agg(
            comfort_pct_medio=('comfort_pct', 'mean'),
            total_energy_media=('total_energy', 'mean')
        ).round(2).sort_values('comfort_pct_medio', ascending=False)
        print(leaderboard.to_markdown())