# analyzer_comparison_v4.py
# -*- coding: utf-8 -*-
"""
Versão 4 Final: Análise comparativa em lote de agentes de RL treinados
contra um baseline de termostato inteligente.
"""
import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import argparse

try:
    # Importa a versão correta do ambiente
    from classroom_ac_env_v4 import ClassroomACEnv, ClassroomConfig, ACState
except ImportError:
    print("ERRO: Certifique-se de que 'classroom_ac_env_v4.py' está na mesma pasta.")
    exit()

# --- 1. AGENTE DE REGRAS APRIMORADO (TERMOSTATO INTELIGENTE) ---
class ThermostatAgent:
    """
    Este agente age como um termostato com uma "zona morta" (deadband).
    Ele tenta manter a temperatura dentro da faixa de conforto [t_min, t_max].
    """
    def __init__(self, config: ClassroomConfig):
        self.config = config
        print(f"Termostato criado com faixa de conforto de [{config.temp_comfort_min}°C - {config.temp_comfort_max}°C].")

    def predict(self, obs: dict, deterministic: bool = True) -> tuple:
        """
        Escolhe uma ação com base na posição da temperatura atual em relação à faixa.
        """
        current_temp = obs['temperature']
        t_min = self.config.temp_comfort_min
        t_max = self.config.temp_comfort_max
        
        # Lógica do termostato
        if current_temp > t_max + 2:  # Se estiver mais de 2°C acima do máximo
            action = ACState.HIGH.value
        elif current_temp > t_max:      # Se estiver acima do máximo
            action = ACState.MEDIUM.value
        elif current_temp < t_min:      # Se estiver abaixo do mínimo (ou confortável)
            action = ACState.OFF.value
        else: # Se estiver dentro da faixa de conforto [t_min, t_max]
            action = ACState.OFF.value # Zona morta: não faz nada
            
        return action, None

# --- 2. FUNÇÃO DE SIMULAÇÃO GENÉRICA ---
def run_simulation_for_agent(agent, scenario: dict, env: ClassroomACEnv) -> pd.DataFrame:
    """Roda uma simulação para um agente específico (RL ou de regras)."""
    obs_vec, info = env.reset(options=scenario)
    history = [info]
    
    for step in range(240): # Simula por 24h
        obs_for_predict = info if isinstance(agent, ThermostatAgent) else obs_vec
        action, _ = agent.predict(obs_for_predict, deterministic=True)
        obs_vec, _, terminated, truncated, info = env.step(int(action))
        info.update({'step': step + 1, 'action': int(action)})
        history.append(info)
        if terminated or truncated: break
            
    return pd.DataFrame(history)

# --- 3. FUNÇÃO DE PLOTAGEM E CÁLCULO DE MÉTRICAS ---
def plot_and_get_metrics(df: pd.DataFrame, env: ClassroomACEnv) -> tuple:
    """Calcula as métricas de desempenho para um DataFrame de simulação."""
    
    # Agora acessamos a configuração através do ambiente
    env_config = env.config
    
    is_in_comfort = (df['temperature'] >= env_config.temp_comfort_min) & (df['temperature'] <= env_config.temp_comfort_max)
    comfort_percentage = is_in_comfort.mean() * 100
    
    df['energy_consumption'] = df['action'].apply(
        lambda a: 0 if pd.isna(a) else env_config.ac_energy_consumption.get(ACState(int(a)), 0)
    )
    # --- CORREÇÃO: Usando env.dt em vez de env_config.dt ---
    total_energy_kwh = (df['energy_consumption'] * env.dt).sum()

    return comfort_percentage, total_energy_kwh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ferramenta de análise comparativa em lote (V4).")
    parser.add_argument("--models_dir", type=str, default="models_v4_grid_search", help="Diretório dos modelos.")
    parser.add_argument("--scenarios", type=str, default="scenarios.json", help="Arquivo JSON de cenários.")
    args = parser.parse_args()

    try:
        with open(args.scenarios, 'r', encoding='utf-8') as f:
            scenarios_to_test = json.load(f)
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{args.scenarios}' não encontrado."); exit()

    model_paths = sorted(glob.glob(os.path.join(args.models_dir, '*.zip')))
    if not model_paths:
        print(f"ERRO: Nenhum modelo (.zip) encontrado na pasta '{args.models_dir}'."); exit()

    print(f"Encontrados {len(model_paths)} modelos para analisar.")

    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('.zip', '')
        print("\n" + "#"*80 + f"\nAnalisando Modelo: {model_name}\n" + "#"*80)
        
        config_path = model_path.replace('.zip', '_config.json')
        env_params = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                full_config = json.load(f)
                env_params = full_config.get('params', {}).get('env_params', {})
        
        config = ClassroomConfig(**env_params)
        env = ClassroomACEnv(config=config)

        rl_agent = DQN.load(model_path, env=env)
        rule_agent = ThermostatAgent(config=config)

        for scenario in scenarios_to_test:
            print(f"\n--- Executando cenário: {scenario['name']} ---")
            df_rl = run_simulation_for_agent(rl_agent, scenario, env)
            df_rules = run_simulation_for_agent(rule_agent, scenario, env)

            # --- CORREÇÃO: Passando 'env' para a função de métricas ---
            comfort_rl, energy_rl = plot_and_get_metrics(df_rl, env)
            comfort_rules, energy_rules = plot_and_get_metrics(df_rules, env)

            # O resto da lógica de plotagem continua a mesma...
            fig, ax1 = plt.subplots(figsize=(18, 8))
            ax2 = ax1.twinx()
            
            ax1.plot(df_rl['step'], df_rl['temperature'], color='royalblue', lw=3, label=f'Temperatura (RL) - Conforto: {comfort_rl:.1f}%, Energia: {energy_rl:.2f} kWh')
            ax1.plot(df_rules['step'], df_rules['temperature'], color='cyan', lw=2, linestyle='--', label=f'Temperatura (Termostato) - Conforto: {comfort_rules:.1f}%, Energia: {energy_rules:.2f} kWh')
            ax2.step(df_rl['step'], df_rl['action'], where='post', color='crimson', alpha=0.8, lw=2, label='Ação (RL)')
            ax2.step(df_rules['step'], df_rules['action'] - 0.1, where='post', color='orange', alpha=0.7, lw=2, linestyle=':', label='Ação (Termostato)')
            
            ax1.axhspan(env.config.temp_comfort_min, env.config.temp_comfort_max, color='green', alpha=0.1, label='Faixa de Conforto')
            
            ax1.set_title(f"Modelo: {model_name}\nCenário: {scenario['name']}", fontsize=16)
            ax1.set_xlabel("Passos de Tempo (6 min)"); ax1.set_ylabel("Temperatura (°C)")
            ax2.set_ylabel("Ação do Agente")
            ax2.set_yticks([0, 1, 2, 3]); ax2.set_yticklabels(['OFF', 'LOW', 'MED', 'HIGH'])
            
            # Combina legendas
            lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper right')
            
            plt.grid(True); plt.show()