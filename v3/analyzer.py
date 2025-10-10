# analyzer_v3.py
# -*- coding: utf-8 -*-
"""
Script de Análise (Versão 3.3) para modelos Stable-Baselines3.

Melhorias:
- Adicionado argumento de linha de comando '-m' para analisar um único modelo.
- Organização aprimorada do código.
"""
import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import argparse # Importa a biblioteca para argumentos de linha de comando

try:
    from classroom_ac_env_v3 import ClassroomACEnv, ACState
except ImportError:
    print("ERRO: Certifique-se de que 'classroom_ac_env_v3.py' está na mesma pasta.")
    exit()

def run_sb3_simulation(model_path: str, scenario: dict):
    """Carrega um modelo SB3 e roda uma simulação, exibindo detalhes da física."""
    model_name = os.path.basename(model_path).replace('.zip', '')
    print(f"\n--- Analisando o cenário: '{scenario['name']}' para o agente: {model_name} ---")
    
    try:
        model = DQN.load(model_path)
    except Exception as e:
        print(f"Erro ao carregar o modelo '{model_path}': {e}"); return

    # NOTA: O env é instanciado com a config padrão aqui.
    # Para uma análise 100% precisa, o ideal seria carregar a config exata usada no treino.
    # Por simplicidade, estamos usando a config padrão para a análise de comportamento.
    env = ClassroomACEnv()
    obs, info = env.reset(options=scenario)
    
    history = []
    temp_anterior = info['temperature']
    
    print("\nIniciando simulação passo a passo com depuração da física:")
    for step in range(120):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        
        log_output = (
            f"[Passo {step:3d}] Ação: {str(info['ac_state']):<6} | "
            f"Net Heat: {info.get('debug_net_heat', 0):7.3f} | "
            f"Temp: {temp_anterior:.2f}°C + ({info.get('debug_temp_change', 0):.3f})°C => {info['temperature']:.2f}°C"
        )
        print(log_output)
        
        temp_anterior = info['temperature']
        
        step_data = info
        step_data['step'] = step
        step_data['action'] = int(action)
        step_data['energy_consumption'] = env.config.ac_energy_consumption.get(ACState(int(action)), 0)
        history.append(step_data)
        
        if terminated or truncated:
            break
            
    df = pd.DataFrame(history)
    
    # Plotagem
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax2 = ax1.twinx()
    
    ax1.plot(df['step'], df['temperature'], color='royalblue', lw=2.5, label='Temperatura (°C)')
    ax2.step(df['step'], df['action'], where='post', color='crimson', alpha=0.7, lw=2, label='Ação do AC')
    comfort_min = env.config.temp_comfort_min; comfort_max = env.config.temp_comfort_max
    ax1.axhspan(comfort_min, comfort_max, color='green', alpha=0.15, label='Faixa de Conforto')
    
    is_in_comfort = (df['temperature'] >= comfort_min) & (df['temperature'] <= comfort_max)
    comfort_percentage = is_in_comfort.mean() * 100
    total_energy_kwh = (df['energy_consumption'] * env.dt).sum()
    
    title = (f"Agente: {model_name}\nCenário: {scenario['name']}\n"
             f"(Tempo em Conforto: {comfort_percentage:.1f}% | Energia Total: {total_energy_kwh:.2f} kWh)")
    ax1.set_title(title, fontsize=16)
    
    ax1.set_xlabel("Passos de Tempo (6 min)"); ax1.set_ylabel("Temperatura (°C)", color='royalblue')
    ax2.set_ylabel("Ação do AC", color='crimson')
    ax2.set_yticks([0, 1, 2, 3]); ax2.set_yticklabels(['OFF', 'LOW', 'MED', 'HIGH'])
    
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.grid(True); plt.show()

if __name__ == '__main__':
    # --- MELHORIA: Adicionando argumentos de linha de comando ---
    parser = argparse.ArgumentParser(description="Ferramenta de análise para modelos de RL.")
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        help="(Opcional) Caminho para um arquivo de modelo .zip específico para analisar."
    )
    parser.add_argument(
        "--models_dir", 
        type=str, 
        default="models", 
        help="Diretório onde os modelos em massa estão salvos (padrão: 'models')."
    )
    parser.add_argument(
        "--scenarios", 
        type=str, 
        default="scenarios.json", 
        help="Arquivo JSON com os cenários de teste (padrão: 'scenarios.json')."
    )
    args = parser.parse_args()

    # Carrega os cenários
    try:
        with open(args.scenarios, 'r', encoding='utf-8') as f:
            scenarios_to_test = json.load(f)
        print(f"{len(scenarios_to_test)} cenários carregados de '{args.scenarios}'.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo de cenários '{args.scenarios}' não encontrado."); exit()

    # --- MELHORIA: Lógica condicional para análise ---
    model_paths = []
    if args.model:
        # Se um modelo específico foi fornecido
        if os.path.exists(args.model):
            model_paths.append(args.model)
        else:
            print(f"ERRO: O arquivo de modelo especificado não foi encontrado: {args.model}"); exit()
    else:
        # Se nenhum modelo foi especificado, procura todos na pasta padrão
        model_paths = glob.glob(os.path.join(args.models_dir, '*.zip'))
        if not model_paths:
            print(f"ERRO: Nenhum modelo (.zip) encontrado na pasta '{args.models_dir}'."); exit()
    
    print(f"\nEncontrados {len(model_paths)} modelos para analisar.")

    # Loop para analisar cada modelo em todos os cenários
    for model_file in model_paths:
        print("\n" + "#"*80)
        print(f"Analisando modelo: {os.path.basename(model_file)}")
        print("#"*80)
        for scenario in scenarios_to_test:
            run_sb3_simulation(model_file, scenario)