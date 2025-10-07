# -*- coding: utf-8 -*-
"""
Script para comparar o controle de temperatura de diferentes agentes de RL.

Este script treina múltiplos agentes com diferentes configurações de hiperparâmetros,
executa um episódio de avaliação para cada um, e plota suas curvas de controle
de temperatura em um único gráfico para análise comparativa.

Autor: Renan (com assistência de IA)
"""

import matplotlib.pyplot as plt
from tqdm import tqdm

# Importa as classes necessárias dos seus outros arquivos
from classroom_ac_env import ClassroomACEnvironment, ClassroomConfig
from ac_qlearning_agent import ACQLearningAgent, QLearningConfig

def plot_temperature_comparison(results: dict, config: ClassroomConfig):
    """
    Plota as curvas de temperatura de múltiplos agentes em um único gráfico.

    Args:
        results (dict): Dicionário com nomes dos agentes como chaves e
                        listas de temperaturas como valores.
        config (ClassroomConfig): A configuração do ambiente para plotar
                                  as linhas de conforto.
    """
    plt.figure(figsize=(12, 7))
    
    # Plota as linhas da faixa de conforto
    plt.axhline(y=config.temp_comfort_min, color='green', linestyle='--', alpha=0.8, label='Conforto Min')
    plt.axhline(y=config.temp_comfort_max, color='green', linestyle='--', alpha=0.8, label='Conforto Max')

    # Plota a curva de temperatura para cada agente
    for name, temp_history in results.items():
        plt.plot(temp_history, label=name, lw=2) # lw = linewidth

    # Configurações do gráfico
    plt.title('Comparação do Controle de Temperatura Entre Agentes', fontsize=16)
    plt.xlabel('Passos de Tempo (intervalos de 6 min)', fontsize=12)
    plt.ylabel('Temperatura da Sala (°C)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.ylim(config.temp_comfort_min - 2, config.temp_comfort_max + 2) # Foco na faixa de conforto
    
    # Salva e exibe o gráfico
    plt.savefig('temperature_comparison.png', dpi=300)
    plt.show()

def main():
    """Função principal que executa a simulação e comparação."""
    print("="*60)
    print("INICIANDO COMPARAÇÃO DE CONTROLE DE TEMPERATURA")
    print("="*60)
    
    # Define as configurações dos agentes a serem comparados
    configs = {
        "Conservador": QLearningConfig(alpha=0.05, gamma=0.9, epsilon=0.1, episodes=200),
        "Agressivo": QLearningConfig(alpha=0.2, gamma=0.99, epsilon=0.3, episodes=200),
        "Equilibrado": QLearningConfig(alpha=0.1, gamma=0.95, epsilon=0.2, episodes=200)
    }
    
    # Dicionário para armazenar o histórico de temperatura de cada agente
    temperature_results = {}
    
    # Cria uma configuração de ambiente padrão
    env_config = ClassroomConfig()
    
    for name, agent_config in configs.items():
        print(f"\n--- Simulando Agente: {name} ---")
        
        # 1. Cria ambiente e agente
        env = ClassroomACEnvironment(env_config)
        agent = ACQLearningAgent(env.n_states, env.n_actions, agent_config)
        
        # 2. Treina o agente (sem mostrar o output do treino para não poluir)
        print("Treinando agente...")
        agent.train(env, verbose=False)
        
        # 3. Executa um episódio de avaliação para registrar a performance
        print("Executando episódio de avaliação...")
        temp_history = []
        state = env.reset()
        done = False
        
        # Simula um dia completo (240 passos)
        for _ in tqdm(range(240)):
            # Escolhe a melhor ação (sem exploração)
            action = agent.choose_action(state, training=False)
            
            # Executa a ação no ambiente
            next_state, _, done, info = env.step(action)
            
            # Guarda a temperatura do passo atual
            temp_history.append(info['temperature'])
            state = next_state
            
            if done:
                break
        
        # Armazena o histórico de temperatura para este agente
        temperature_results[name] = temp_history
    
    # 4. Plota os resultados comparativos
    print("\n--- Gerando Gráfico Comparativo ---")
    plot_temperature_comparison(temperature_results, env_config)
    print("Gráfico 'temperature_comparison.png' salvo com sucesso!")

if __name__ == "__main__":
    main()