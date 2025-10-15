# main_train_v4_grid_search.py
# -*- coding: utf-8 -*-
"""
Versão 4.2: Varredura de Hiperparâmetros em Larga Escala.
Este script executa um grid search completo, testando múltiplas configurações de
ambiente (hardware do AC) contra múltiplas "personalidades" de agente.
"""
import os
import json
import time
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# Certifique-se de que a versão mais recente do seu ambiente está na pasta
from classroom_ac_env_v4 import ClassroomACEnv, ClassroomConfig, ActionRepeatWrapper

if __name__ == '__main__':
    # --- 1. GRADE DE TESTES DE HARDWARE (CONFIGURAÇÕES DO AR-CONDICIONADO) ---
    ac_setups = {
        "AC_Eco": {
            'ac_cooling_power': {'OFF': 0.0, 'LOW': 2.0, 'MEDIUM': 4.0, 'HIGH': 6.0}
        },
        "AC_Padrao": {
            'ac_cooling_power': {'OFF': 0.0, 'LOW': 4.0, 'MEDIUM': 8.0, 'HIGH': 12.0}
        },
        "AC_Super": {
            'ac_cooling_power': {'OFF': 0.0, 'LOW': 8.0, 'MEDIUM': 16.0, 'HIGH': 24.0}
        }
    }

    # --- 2. GRADE DE TESTES DE "PERSONALIDADES" DO AGENTE ---
    agent_personalities = {
        "Agente_Equilibrado": {
            "env_params": {"thermal_mass": 25.0, "action_change_penalty": -10.0, "comfort_bonus": 15.0, "comfort_sensitivity": 0.6},
            "agent_params": {"learning_rate": 5e-5, "batch_size": 64},
            "action_repeat": 2,
        },
        "Agente_Paciente_Economico": {
            "env_params": {"thermal_mass": 80.0, "action_change_penalty": -25.0, "energy_penalty_factor": 0.3},
            "agent_params": {"learning_rate": 1e-5, "batch_size": 128},
            "action_repeat": 4,
        },
        "Agente_Focado_em_Conforto": {
            "env_params": {"thermal_mass": 15.0, "action_change_penalty": -5.0, "comfort_bonus": 50.0, "energy_penalty_factor": 0.05},
            "agent_params": {"learning_rate": 1e-4, "batch_size": 32},
            "action_repeat": 1,
        },
        "Agente_Estavel": {
            "env_params": {"thermal_mass": 40.0, "action_change_penalty": -50.0, "comfort_bonus": 20.0},
            "agent_params": {"learning_rate": 3e-5},
            "action_repeat": 3,
        },
        # --- NOVA PERSONALIDADE ADICIONADA AQUI ---
        "Agente_Smart_Cold": {
            "env_params": {
                "thermal_mass": 25.0, 
                "action_change_penalty": -15.0, 
                "comfort_bonus": 20.0, 
                "comfort_sensitivity": 0.7,
                "cold_action_penalty": -50.0 # A penalidade por resfriar no frio
            },
            "agent_params": {"learning_rate": 5e-5},
            "action_repeat": 2,
        }
    }
    
    # --- Cria pastas para organização ---
    models_dir = "models_v4_grid_search"
    logs_dir = "sb3_logs_v4_grid_search"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # --- 3. LOOPS ANINHADOS PARA CRUZAR OS EXPERIMENTOS ---
    total_experiments = len(ac_setups) * len(agent_personalities)
    current_experiment = 0

    for ac_name, ac_params in ac_setups.items():
        for agent_name, agent_params in agent_personalities.items():
            current_experiment += 1
            exp_name = f"{ac_name}_{agent_name}"
            
            print("\n" + "="*80)
            print(f"INICIANDO EXPERIMENTO [{current_experiment}/{total_experiments}]: {exp_name}")
            print("="*80)
            start_time = time.time()

            # Combina os parâmetros do ambiente
            # Começa com os parâmetros da "personalidade"
            env_params = agent_params.get('env_params', {}).copy()
            # Adiciona os parâmetros do "hardware"
            env_params.update(ac_params)

            config = ClassroomConfig(**env_params)
            env = ClassroomACEnv(config=config)
            env = ActionRepeatWrapper(env, repeat=agent_params.get("action_repeat", 1))
            env = Monitor(env)

            # Define caminhos únicos
            model_path = os.path.join(models_dir, exp_name)
            tensorboard_log_path = os.path.join(logs_dir, exp_name)
            config_path = os.path.join(models_dir, f"{exp_name}_config.json")

            base_agent_params = {
            "policy": "MlpPolicy",
            "env": env,
            "verbose": 0,
            "tensorboard_log": tensorboard_log_path
            }
            all_agent_params = {**base_agent_params, **agent_params.get('agent_params', {})}
            # Define o modelo DQN com os hiperparâmetros do agente
            model = DQN(**all_agent_params)

            # Treina o modelo
            # Reduzido para testes mais rápidos, aumente para um treino final.
            total_training_steps = 550000 
            model.learn(total_timesteps=total_training_steps, log_interval=None) # Desativa o log_interval do learn para um output mais limpo

            # Salva o modelo e a configuração usada
            model.save(model_path)
            full_exp_config = {
                "ac_setup_name": ac_name,
                "agent_personality_name": agent_name,
                "params": {
                    "env_params": env_params,
                    "agent_params": agent_params.get('agent_params', {}),
                    "action_repeat": agent_params.get("action_repeat", 1)
                }
            }
            with open(config_path, 'w') as f:
                json.dump(full_exp_config, f, indent=4)
            
            end_time = time.time()
            print(f"-> Experimento '{exp_name}' concluído em {(end_time - start_time)/60:.2f} minutos.")
            print(f"-> Modelo salvo em: {model_path}.zip")

    print("\n\n" + "*"*80 + "\nVARREDURA COMPLETA DE PARÂMETROS CONCLUÍDA!\n" + "*"*80)