# -*- coding: utf-8 -*-
"""
Demonstração do Sistema de Controle de Ar-Condicionado com RL

Este script demonstra o funcionamento básico do sistema de controle
inteligente de ar-condicionado para salas de aula.

Autor: Renan (com assistência de IA)
"""

import numpy as np
import matplotlib.pyplot as plt
from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig, ACState, ComfortLevel
from ac_qlearning_agent import ACQLearningAgent, QLearningConfig

def demo_basic_training():
    """Demonstração básica de treinamento"""
    print("="*60)
    print("DEMONSTRAÇÃO: Sistema de Controle de Ar-Condicionado com RL")
    print("="*60)
    
    # 1. Cria ambiente
    print("\n1. Criando ambiente de sala de aula...")
    config = ClassroomConfig()
    env = ClassroomACEnvironment(config)
    print(f"   - Dimensões: {config.length}m × {config.width}m × {config.height}m")
    print(f"   - Capacidade: {config.max_occupancy} pessoas")
    print(f"   - Faixa de conforto: {config.temp_comfort_min}-{config.temp_comfort_max}°C")
    print(f"   - Estados possíveis: {env.n_states}")
    print(f"   - Ações possíveis: {env.n_actions}")
    
    # 2. Cria agente
    print("\n2. Criando agente Q-Learning...")
    agent_config = QLearningConfig(
        alpha=0.1,
        gamma=0.95,
        epsilon=0.2,
        episodes=200  # Reduzido para demonstração
    )
    agent = ACQLearningAgent(env.n_states, env.n_actions, agent_config)
    print(f"   - Taxa de aprendizado: {agent_config.alpha}")
    print(f"   - Fator de desconto: {agent_config.gamma}")
    print(f"   - Taxa de exploração: {agent_config.epsilon}")
    
    # 3. Treina agente
    print("\n3. Treinando agente...")
    print("   (Isso pode levar alguns minutos...)")
    history = agent.train(env, verbose=True)
    
    # 4. Avalia política
    print("\n4. Avaliando política aprendida...")
    eval_stats = agent.evaluate(env, episodes=3, render=False)
    
    # 5. Mostra resultados
    print("\n5. Resultados do treinamento:")
    print(f"   - Recompensa média: {eval_stats['avg_reward']:.2f}")
    print(f"   - Conforto médio: {eval_stats['avg_comfort']:.1f}%")
    print(f"   - Consumo energético: {eval_stats['avg_energy']:.2f} kW")
    print(f"   - Uso do AC: {eval_stats['avg_ac_usage']:.1f}%")
    print(f"   - Temperatura média: {eval_stats['avg_temperature']:.1f}°C")
    
    return agent, env, history

def demo_policy_analysis(agent, env):
    """Demonstração da análise da política"""
    print("\n" + "="*60)
    print("ANÁLISE DA POLÍTICA APRENDIDA")
    print("="*60)
    
    # Análise da política
    policy = agent.get_policy()
    action_names = ['OFF', 'LOW', 'MEDIUM', 'HIGH']
    
    print("\n1. Distribuição de ações na política:")
    unique, counts = np.unique(policy, return_counts=True)
    for action, count in zip(unique, counts):
        percentage = count / len(policy) * 100
        print(f"   - {action_names[action]}: {count} estados ({percentage:.1f}%)")
    
    # Testa cenários específicos
    print("\n2. Testando cenários específicos:")
    
    scenarios = [
        {"name": "Sala vazia, manhã", "temp": 20, "occ": 0, "hour": 8},
        {"name": "Sala cheia, manhã", "temp": 20, "occ": 30, "hour": 8},
        {"name": "Sala vazia, tarde quente", "temp": 30, "occ": 0, "hour": 14},
        {"name": "Sala cheia, tarde quente", "temp": 30, "occ": 30, "hour": 14},
        {"name": "Sala vazia, noite", "temp": 18, "occ": 0, "hour": 20},
    ]
    
    for scenario in scenarios:
        # Configura cenário
        env.current_temp = scenario["temp"]
        env.occupancy = scenario["occ"]
        env.hour_of_day = scenario["hour"]
        
        # Obtém ação recomendada
        state = env._discretize_state()
        action = agent.choose_action(state, training=False)
        
        print(f"   - {scenario['name']}: {action_names[action]}")

def demo_visualization(agent, env):
    """Demonstração das visualizações"""
    print("\n" + "="*60)
    print("VISUALIZAÇÕES DO SISTEMA")
    print("="*60)
    
    # 1. Progresso de treinamento
    print("\n1. Gerando gráfico de progresso de treinamento...")
    agent.plot_training_progress()
    
    # 2. Simulação em tempo real
    print("\n2. Simulando controle em tempo real...")
    state = env.reset()
    
    # Configura cenário interessante
    env.current_temp = 25.0
    env.occupancy = 20
    env.hour_of_day = 10
    
    print("   Simulando 50 passos de tempo (5 horas)...")
    print("   Tempo | Temp | Ocup | AC   | Conforto | Recompensa")
    print("   " + "-"*50)
    
    total_reward = 0
    for step in range(50):
        action = agent.choose_action(state, training=False)
        next_state, reward, done, info = env.step(action)
        
        if step % 10 == 0:  # Mostra a cada 10 passos
            print(f"   {step:4d} | {info['temperature']:4.1f} | {info['occupancy']:4d} | "
                  f"{ACState(action).name:4s} | {info['comfort_level']:8s} | {reward:8.2f}")
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    print(f"\n   Recompensa total: {total_reward:.2f}")
    
    # 3. Renderiza ambiente
    print("\n3. Renderizando visualização do ambiente...")
    env.render()

def demo_comparison():
    """Demonstração comparativa entre diferentes configurações"""
    print("\n" + "="*60)
    print("COMPARAÇÃO DE CONFIGURAÇÕES")
    print("="*60)
    
    configs = {
        "Conservador": QLearningConfig(alpha=0.05, gamma=0.9, epsilon=0.1, episodes=100),
        "Agressivo": QLearningConfig(alpha=0.2, gamma=0.99, epsilon=0.3, episodes=100),
        "Equilibrado": QLearningConfig(alpha=0.1, gamma=0.95, epsilon=0.2, episodes=100)
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\nTestando configuração: {name}")
        
        # Cria ambiente e agente
        env = ClassroomACEnvironment(ClassroomConfig())
        agent = ACQLearningAgent(env.n_states, env.n_actions, config)
        
        # Treina rapidamente
        agent.train(env, verbose=False)
        
        # Avalia
        eval_stats = agent.evaluate(env, episodes=3, render=False)
        results[name] = eval_stats
        
        print(f"   Recompensa: {eval_stats['avg_reward']:.2f}")
        print(f"   Conforto: {eval_stats['avg_comfort']:.1f}%")
        print(f"   Energia: {eval_stats['avg_energy']:.2f} kW")
    
    # Mostra comparação
    print("\nComparação final:")
    print("Configuração | Recompensa | Conforto | Energia")
    print("-" * 45)
    for name, stats in results.items():
        print(f"{name:12s} | {stats['avg_reward']:9.2f} | {stats['avg_comfort']:7.1f}% | {stats['avg_energy']:6.2f} kW")

def main():
    """Função principal da demonstração"""
    try:
        # Demonstração básica
        agent, env, history = demo_basic_training()
        
        # Análise da política
        demo_policy_analysis(agent, env)
        
        # Visualizações
        demo_visualization(agent, env)
        
        # Comparação
        demo_comparison()
        
        print("\n" + "="*60)
        print("DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*60)
        print("\nO sistema demonstrou:")
        print("✓ Treinamento de agente RL para controle de AC")
        print("✓ Aprendizado de política balanceando conforto e eficiência")
        print("✓ Análise de comportamento em diferentes cenários")
        print("✓ Visualizações e métricas de performance")
        print("\nPara experimentos mais detalhados, execute:")
        print("  python main_train.py")
        print("\nPara análise avançada, execute:")
        print("  python analysis_utils.py")
        
    except Exception as e:
        print(f"\nErro durante a demonstração: {e}")
        print("Verifique se todas as dependências estão instaladas.")

if __name__ == "__main__":
    main()