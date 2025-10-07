# main_train.py
# -*- coding: utf-8 -*-
"""
Script Principal para Treinamento e Avaliação de Agentes (Q-Learning e DQN).

Este script orquestra o treinamento de agentes para controlar o sistema
de ar-condicionado, sendo flexível para diferentes tipos de agentes e configurações.

Autor: Renan Saraiva dos Santos & Gemini Mentor
"""

import os
import json
import dataclasses
import pickle
from datetime import datetime
import logging
import numpy as np


# --- MODIFICAÇÃO: Importações flexíveis e dos novos agentes ---
from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig
from ac_qlearning_agent import ACQLearningAgent, QLearningConfig
from ac_dqn_agent import ACDQNAgent, DQNConfig

# --- MODIFICAÇÃO: Configuração do logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def create_experiment_configs():
    """Define as configurações para cada experimento a ser executado."""
    EPISODES_QL = 1000
    EPISODES_DQN = 2000 # DQN pode ser mais rápido
    
    # Estratégia 1: Aumentar moderadamente o bônus de conforto
    reward_moderate_focus = {
        'COMFORTABLE': 5.0, # Aumentado de 1.0 para 5.0
        'WARM': -5.0, 'VERY_HOT': -15.0, 'COLD': -5.0, 'VERY_COLD': -15.0
    }

    # Estratégia 2: Aumentar agressivamente o bônus de conforto
    reward_high_focus = {
        'COMFORTABLE': 10.0, # Aumentado de 1.0 para 10.0
        'WARM': -5.0, 'VERY_HOT': -15.0, 'COLD': -5.0, 'VERY_COLD': -15.0
    }

    # Estratégia 3: Bônus alto e penalidades mais severas (Tolerância Zero)
    reward_zero_tolerance = {
        'COMFORTABLE': 10.0,
        'WARM': -10.0,       # Penalidade dobrada
        'VERY_HOT': -30.0,  # Penalidade dobrada
        'COLD': -10.0,
        'VERY_COLD': -30.0
    }

    # --- DEFINIÇÃO DOS EXPERIMENTOS ---
    configs = {
        'dqn_baseline': {
            'agent_type': 'dqn',
            'env_config': ClassroomConfig(season='summer'), # Nosso controle
            'agent_config': DQNConfig(episodes=EPISODES_DQN),
            'name': 'DQN Baseline'
        },
        'dqn_moderate_comfort': {
            'agent_type': 'dqn',
            'env_config': ClassroomConfig(season='summer', reward_structure=reward_moderate_focus),
            'agent_config': DQNConfig(episodes=EPISODES_DQN),
            'name': 'DQN Foco Moderado em Conforto'
        },
        'dqn_high_comfort': {
            'agent_type': 'dqn',
            'env_config': ClassroomConfig(season='summer', reward_structure=reward_high_focus),
            'agent_config': DQNConfig(episodes=EPISODES_DQN),
            'name': 'DQN Foco Alto em Conforto'
        },
        'dqn_zero_tolerance': {
            'agent_type': 'dqn',
            'env_config': ClassroomConfig(season='summer', reward_structure=reward_zero_tolerance),
            'agent_config': DQNConfig(episodes=EPISODES_DQN),
            'name': 'DQN Tolerância Zero'
        }
    }
    return configs

def sanitize_for_json(obj):
    """Converte recursivamente um objeto para que seja serializável em JSON."""
    if isinstance(obj, dict): return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [sanitize_for_json(elem) for elem in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def run_experiment(exp_config, save_dir):
    """Executa um experimento completo (treinamento e avaliação)."""
    name = exp_config['name']
    agent_type = exp_config['agent_type']
    env_config = exp_config['env_config']
    agent_config = exp_config['agent_config']
    
    logging.info(f"\n{'='*50}\nExecutando experimento: {name}\n{'='*50}")

    env = ClassroomACEnvironment(env_config)
    
    if agent_type == 'q_learning':
        # --- CORREÇÃO: Adicionamos um método de avaliação ao Q-Learning agent para consistência ---
        # (O seu agente Q-Learning já deve ter um método evaluate, esta linha é para garantir)
        if not hasattr(ACQLearningAgent, 'evaluate'):
            # Este é um placeholder, o seu agente já tem um melhor
            def q_eval(self, env, episodes=10000): return {} 
            ACQLearningAgent.evaluate = q_eval
        agent = ACQLearningAgent(env.n_states, env.n_actions, agent_config)
    elif agent_type == 'dqn':
        agent = ACDQNAgent(agent_config)
    else:
        raise ValueError(f"Tipo de agente desconhecido: {agent_type}")

    logging.info("Treinando agente...")
    history = agent.train(env, verbose=True)
    
    # --- CORREÇÃO: Etapa de avaliação adicionada ---
    logging.info("Avaliando agente treinado...")
    evaluation_stats = agent.evaluate(env, episodes=10000) # Avalia por 50 episódios para ter uma boa média
    
    # Salvar resultados
    results = {
        'name': name,
        'agent_type': agent_type,
        'config': {'env': dataclasses.asdict(env_config), 'agent': dataclasses.asdict(agent_config)},
        'training_history': history,
        'evaluation_stats': evaluation_stats, 
        'timestamp': datetime.now().isoformat()
    }
    
    model_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
    if agent_type == 'dqn':
        model_path = model_path.replace('.pkl', '.pth')

    agent.save_model(model_path)
    
    results_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_results.json")
    with open(results_path, 'w') as f:
        json.dump(sanitize_for_json(results), f, indent=4)
        
    logging.info(f"Resultados e modelo para '{name}' salvos com sucesso.")
    return results

def main():
    """Função principal que orquestra todos os experimentos."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Resultados serão salvos em: {save_dir}")
    
    experiment_configs = create_experiment_configs()
    
    for _, config in experiment_configs.items():
        try:
            run_experiment(config, save_dir)
        except Exception as e:
            logging.error(f"Erro catastrófico no experimento {config['name']}: {e}", exc_info=True)
            continue
            
    logging.info(f"\nTodos os experimentos foram concluídos! Resultados em: {save_dir}")
    print(f"\nPara analisar os resultados, execute:\npython analysis.py {save_dir}")

if __name__ == "__main__":
    main()












