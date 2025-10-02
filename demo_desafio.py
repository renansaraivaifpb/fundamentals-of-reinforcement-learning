# -*- coding: utf-8 -*-
"""
Script de Demonstração Desafiador para o Sistema de Controle de AC.

Este script executa uma simulação comparativa entre os agentes em um
cenário mais difícil:
- A sala esquenta mais rápido (menor inércia térmica).
- A avaliação começa com a sala já quente (27°C).

Isso força os agentes a demonstrarem ativamente suas políticas de
resfriamento.

Autor: Renan Saraiva dos Santos 
"""

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Importa as classes necessárias dos seus outros arquivos
from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig, ACState
from ac_qlearning_agent import ACQLearningAgent, QLearningConfig

def plot_temperature_comparison(results: dict, config: ClassroomConfig):
    """
    Plota as curvas de temperatura de múltiplos agentes em um único gráfico.
    """
    plt.figure(figsize=(12, 7))
    
    # Plota as linhas da faixa de conforto
    plt.axhline(y=config.temp_comfort_min, color='green', linestyle='--', alpha=0.8, label='Conforto Min')
    plt.axhline(y=config.temp_comfort_max, color='green', linestyle='--', alpha=0.8, label='Conforto Max')

    # Plota a curva de temperatura para cada agente
    for name, temp_history in results.items():
        plt.plot(temp_history, label=name, lw=2, alpha=0.9)

    # Configurações do gráfico
    plt.title('Comparação do Controle de Temperatura (Cenário Desafiador)', fontsize=16)
    plt.xlabel('Passos de Tempo (intervalos de 6 min)', fontsize=12)
    plt.ylabel('Temperatura da Sala (°C)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.ylim(config.temp_comfort_min - 2, config.temp_very_hot - 2)
    
    # Salva e exibe o gráfico
    plt.savefig('temperature_comparison_challenging.png', dpi=300)
    plt.show()

def main():
    """Função principal que executa a simulação e comparação."""
    print("="*60)
    print("INICIANDO COMPARAÇÃO EM CENÁRIO DESAFIADOR")
    print("="*60)
    
    # --- 1. CONFIGURAÇÃO DO AMBIENTE DESAFIADOR ---
    # Criamos uma nova configuração que faz a sala esquentar mais rápido
    challenging_env_config = ClassroomConfig(
        thermal_mass=400.0,         # Reduzido de 1000 (menos inércia térmica)
        heat_gain_per_person=0.12,  # Aumentado de 0.1 (pessoas geram mais calor)
    )
    print("Ambiente configurado com menor inércia térmica e maior ganho de calor.")
    
    # Define as configurações dos agentes a serem comparados
    agent_configs = {
        "Conservador": QLearningConfig(alpha=0.05, gamma=0.9, epsilon=0.1, episodes=300),
        "Agressivo": QLearningConfig(alpha=0.2, gamma=0.99, epsilon=0.3, episodes=300),
        "Equilibrado": QLearningConfig(alpha=0.1, gamma=0.95, epsilon=0.2, episodes=300)
    }
    
    temperature_results = {}
    
    for name, agent_config in agent_configs.items():
        print(f"\n--- Simulando Agente: {name} ---")
        
        # 2. Cria ambiente e agente com a configuração desafiadora
        env = ClassroomACEnvironment(challenging_env_config)
        agent = ACQLearningAgent(env.n_states, env.n_actions, agent_config)
        
        # 3. Treina o agente
        print(f"Treinando agente '{name}'...")
        agent.train(env, verbose=False)
        
        # 4. Executa um episódio de avaliação para registrar a performance
        print("Executando episódio de avaliação em cenário quente...")
        temp_history = []
        state = env.reset()
        
        # --- 5. PONTO DE PARTIDA DESAFIADOR ---
        # Forçamos o início da avaliação com a sala quente e com ocupação
        env.current_temp = 27.0
        env.occupancy = 25
        state = env._discretize_state() # Atualiza o estado discreto
        
        temp_history.append(env.current_temp)

        # Simula um dia completo (240 passos)
        for _ in tqdm(range(239)):
            action = agent.choose_action(state, training=False)
            next_state, _, done, info = env.step(action)
            
            temp_history.append(info['temperature'])
            state = next_state
            
            if done:
                break
        
        temperature_results[name] = temp_history
    
    # 6. Plota os resultados comparativos
    print("\n--- Gerando Gráfico Comparativo ---")
    plot_temperature_comparison(temperature_results, challenging_env_config)
    print("Gráfico 'temperature_comparison_challenging.png' salvo com sucesso!")

if __name__ == "__main__":
    main()