# main_train_v3.py
# -*- coding: utf-8 -*-
"""
Script de Treinamento (Versão 3.2) - Orquestrador de Experimentos com Stable-Baselines3.
"""
import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import time

# Importa o ambiente compatível com Gymnasium
from classroom_ac_env_v3 import ClassroomACEnv, ClassroomConfig

if __name__ == '__main__':
    # --- DEFINIÇÃO DOS EXPERIMENTOS ---
    experiments = {
        "DQN_Baseline_Asymmetric": {
            "reward_structure": {
                'COMFORTABLE': 5.0, 'WARM': -5.0, 'VERY_HOT': -15.0,
                'COLD': 0.0, 'VERY_COLD': -1.0
            }, # Use uma recompensa agressiva
        "action_change_penalty": -10.0 
        },
        "DQN_Aggressive_Bonus": {
            "reward_structure": {
                'COMFORTABLE': 15.0, 'WARM': -5.0, 'VERY_HOT': -15.0,
                'COLD': 0.0, 'VERY_COLD': -1.0
            }, # Use uma recompensa agressiva
        "action_change_penalty": -10.0 
        },
        "DQN_Aggressive_Penalty": {
            "reward_structure": {
                'COMFORTABLE': 5.0, 'WARM': -15.0, 'VERY_HOT': -40.0,
                'COLD': 0.0, 'VERY_COLD': -1.0
            }, # Use uma recompensa agressiva
        "action_change_penalty": -10.0 
        },
        "DQN_Combined_Aggression": {
            "reward_structure": {
                'COMFORTABLE': 15.0, 'WARM': -15.0, 'VERY_HOT': -40.0,
                'COLD': 0.0, 'VERY_COLD': -1.0
            }, # Use uma recompensa agressiva
        "action_change_penalty": -10.0 
        }
    }
    
    # --- ADIÇÃO CRÍTICA: Definindo a física do AC potente ---
    powerful_ac_config = {
        'OFF': 0.0, 'LOW': 8.0, 'MEDIUM': 16.0, 'HIGH': 24.0
    }

    # --- CRIA PASTAS PARA ORGANIZAÇÃO ---
    models_dir = "models"
    logs_dir = "sb3_logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # --- LOOP PRINCIPAL DE TREINAMENTO ---
    for exp_name, exp_config in experiments.items():
        print("\n" + "="*80)
        print(f"INICIANDO EXPERIMENTO: {exp_name}")
        print("="*80)
        
        start_time = time.time()

        # 1. Configura o ambiente para este experimento
        # --- CORREÇÃO: Passando a configuração do AC potente ---
        # Experimento para forçar uma política estável
        config_estavel = ClassroomConfig(
            ac_cooling_power=powerful_ac_config,
            comfort_bonus=10.0,             # Recompensa alta por conforto
            comfort_sensitivity=0.8,        # Penalidade quadrática mais severa
            action_change_penalty=-30.0     # Penalidade de mudança 5x maior!
        )
        
        env = ClassroomACEnv(config=config_estavel)
        env = Monitor(env)

        # 2. Define caminhos únicos para salvar o modelo e os logs
        model_path = os.path.join(models_dir, exp_name)
        tensorboard_log_path = os.path.join(logs_dir, exp_name)

        # 3. Define o modelo DQN
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=5e-5,
            buffer_size=150000,
            learning_starts=1000,
            batch_size=64,
            verbose=1,
            tensorboard_log=tensorboard_log_path
        )

        # 4. Treina o modelo
        total_training_steps = 200000
        model.learn(total_timesteps=total_training_steps, log_interval=10)

        # 5. Salva o modelo treinado
        model.save(model_path)
        
        end_time = time.time()
        print(f"Experimento '{exp_name}' concluído em {(end_time - start_time)/60:.2f} minutos.")
        print(f"Modelo salvo em: {model_path}.zip")

    print("\n\n" + "*"*80)
    print("TODOS OS EXPERIMENTOS FORAM CONCLUÍDOS!")
    print("*"*80)