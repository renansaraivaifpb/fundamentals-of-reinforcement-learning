# -*- coding: utf-8 -*-
"""
Ambiente de Sala de Aula com Ar-Condicionado para Aprendizagem por Reforço

Este módulo implementa um ambiente simulado de uma sala de aula com sistema de ar-condicionado,
onde um agente de RL deve aprender a controlar a temperatura de forma a balancear:
- Conforto térmico dos usuários
- Eficiência energética

Autor: Renan (com assistência de IA)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum

class ACState(Enum):
    """Estados do ar-condicionado"""
    OFF = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class ComfortLevel(Enum):
    """Níveis de conforto térmico"""
    VERY_COLD = 0
    COLD = 1
    COMFORTABLE = 2
    WARM = 3
    VERY_HOT = 4

@dataclass
class ClassroomConfig:
    """Configuração da sala de aula"""
    # Dimensões da sala (metros)
    length: float = 8.0
    width: float = 6.0
    height: float = 3.0
    
    # Capacidade de ocupação
    max_occupancy: int = 30
    
    # Parâmetros térmicos
    thermal_mass: float = 1000.0  # Capacidade térmica da sala
    heat_transfer_coeff: float = 0.5  # Coeficiente de transferência de calor
    
    # Temperaturas ideais (Celsius)
    temp_comfort_min: float = 22.0
    temp_comfort_max: float = 26.0
    temp_very_cold: float = 18.0
    temp_very_hot: float = 30.0
    
    # Parâmetros do ar-condicionado
    ac_cooling_power: Dict[ACState, float] = None  # Potência de refrigeração por estado
    ac_energy_consumption: Dict[ACState, float] = None  # Consumo energético por estado
    
    def __post_init__(self):
        if self.ac_cooling_power is None:
            self.ac_cooling_power = {
                ACState.OFF: 0.0,
                ACState.LOW: 2.0,    # kW
                ACState.MEDIUM: 4.0,
                ACState.HIGH: 6.0
            }
        
        if self.ac_energy_consumption is None:
            self.ac_energy_consumption = {
                ACState.OFF: 0.0,
                ACState.LOW: 1.5,    # kW
                ACState.MEDIUM: 3.0,
                ACState.HIGH: 5.0
            }

class ClassroomACEnvironment:
    """
    Ambiente de sala de aula com sistema de ar-condicionado controlado por RL.
    
    Estados:
        - Temperatura atual da sala (discretizada)
        - Número de ocupantes (discretizado)
        - Estado atual do ar-condicionado
        - Hora do dia (para simular variações externas)
    
    Ações:
        - Manter estado atual do AC
        - Ligar/desligar AC
        - Aumentar/diminuir potência do AC
    """
    
    def __init__(self, config: ClassroomConfig = None):
        self.config = config or ClassroomConfig()
        
        # Estado atual
        self.current_temp = 24.0  # Temperatura inicial (Celsius)
        self.occupancy = 0
        self.ac_state = ACState.OFF
        self.time_step = 0
        self.hour_of_day = 8  # Hora inicial (8h da manhã)
        
        # Parâmetros de simulação
        self.dt = 0.1  # Passo de tempo (horas)
        self.outdoor_temp = 25.0  # Temperatura externa base
        self.heat_gain_per_person = 0.1  # kW por pessoa
        
        # Discretização para RL
        self.temp_bins = np.linspace(15, 35, 21)  # 20 bins de temperatura
        self.occupancy_bins = [0, 5, 10, 15, 20, 25, 30]  # 6 bins de ocupação
        self.time_bins = list(range(24))  # 24 horas do dia
        
        # Ações possíveis
        self.actions = list(ACState)
        self.n_actions = len(self.actions)
        
        # Estados discretos
        self.n_temp_states = len(self.temp_bins) - 1
        self.n_occupancy_states = len(self.occupancy_bins) - 1
        self.n_time_states = len(self.time_bins)
        
        # Estado total = (temp_discrete, occupancy_discrete, time_discrete)
        self.n_states = (self.n_temp_states * 
                        self.n_occupancy_states * 
                        self.n_time_states)
        
        # Histórico para análise
        self.temp_history = []
        self.energy_history = []
        self.comfort_history = []
        self.ac_state_history = []
        
    def _discretize_state(self) -> int:
        """Converte estado contínuo para estado discreto para RL"""
        # Discretiza temperatura
        temp_idx = np.digitize(self.current_temp, self.temp_bins) - 1
        temp_idx = np.clip(temp_idx, 0, self.n_temp_states - 1)
        
        # Discretiza ocupação
        occ_idx = np.digitize(self.occupancy, self.occupancy_bins) - 1
        occ_idx = np.clip(occ_idx, 0, self.n_occupancy_states - 1)
        
        # Hora do dia
        time_idx = self.hour_of_day % 24
        
        # Converte para estado único
        state = (temp_idx * self.n_occupancy_states * self.n_time_states + 
                occ_idx * self.n_time_states + 
                time_idx)
        
        return state
    
    def _get_comfort_level(self) -> ComfortLevel:
        """Determina o nível de conforto térmico baseado na temperatura"""
        if self.current_temp < self.config.temp_very_cold:
            return ComfortLevel.VERY_COLD
        elif self.current_temp < self.config.temp_comfort_min:
            return ComfortLevel.COLD
        elif self.current_temp <= self.config.temp_comfort_max:
            return ComfortLevel.COMFORTABLE
        elif self.current_temp < self.config.temp_very_hot:
            return ComfortLevel.WARM
        else:
            return ComfortLevel.VERY_HOT
    
    def _calculate_outdoor_temp(self) -> float:
        """Calcula temperatura externa baseada na hora do dia"""
        # Simula variação diária da temperatura externa
        base_temp = 20.0
        amplitude = 8.0
        phase = (self.hour_of_day - 6) * np.pi / 12  # Pico às 14h
        
        return base_temp + amplitude * np.sin(phase)
    
    def _calculate_heat_gain(self) -> float:
        """Calcula ganho de calor total na sala"""
        # Calor dos ocupantes
        people_heat = self.occupancy * self.heat_gain_per_person
        
        # Calor externo (simplificado)
        outdoor_temp = self._calculate_outdoor_temp()
        external_heat = self.config.heat_transfer_coeff * (outdoor_temp - self.current_temp)
        
        return people_heat + external_heat
    
    def _calculate_cooling_effect(self) -> float:
        """Calcula efeito de refrigeração do ar-condicionado"""
        return self.config.ac_cooling_power[self.ac_state]
    
    def _calculate_energy_consumption(self) -> float:
        """Calcula consumo energético do ar-condicionado"""
        return self.config.ac_energy_consumption[self.ac_state]
    
    def _calculate_reward(self) -> float:
        """
        Calcula recompensa baseada em conforto térmico e eficiência energética.
        
        A recompensa balanceia:
        - Conforto térmico (peso maior)
        - Eficiência energética (peso menor)
        """
        comfort_level = self._get_comfort_level()
        energy_consumption = self._calculate_energy_consumption()
        
        # Recompensa por conforto (0 a 1)
        comfort_rewards = {
            ComfortLevel.VERY_COLD: -2.0,
            ComfortLevel.COLD: -1.0,
            ComfortLevel.COMFORTABLE: 1.0,
            ComfortLevel.WARM: -1.0,
            ComfortLevel.VERY_HOT: -2.0
        }
        comfort_reward = comfort_rewards[comfort_level]
        
        # Penalidade por consumo energético (0 a -0.5)
        energy_penalty = -energy_consumption * 0.1
        
        # Recompensa total
        total_reward = comfort_reward + energy_penalty
        
        return total_reward
    
    def reset(self) -> int:
        """Reseta o ambiente para estado inicial"""
        self.current_temp = 24.0
        self.occupancy = np.random.randint(0, self.config.max_occupancy + 1)
        self.ac_state = ACState.OFF
        self.time_step = 0
        self.hour_of_day = 8
        
        # Limpa histórico
        self.temp_history = [self.current_temp]
        self.energy_history = [0.0]
        self.comfort_history = [self._get_comfort_level()]
        self.ac_state_history = [self.ac_state]
        
        return self._discretize_state()
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Executa uma ação no ambiente.
        
        Args:
            action: Ação a ser executada (índice do ACState)
            
        Returns:
            next_state: Próximo estado discreto
            reward: Recompensa obtida
            done: Se o episódio terminou
            info: Informações adicionais
        """
        # Atualiza estado do ar-condicionado
        self.ac_state = ACState(action)
        
        # Calcula mudanças na temperatura
        heat_gain = self._calculate_heat_gain()
        cooling_effect = self._calculate_cooling_effect()
        
        # Atualiza temperatura usando modelo térmico simplificado
        net_heat = heat_gain - cooling_effect
        temp_change = (net_heat / self.config.thermal_mass) * self.dt
        self.current_temp += temp_change
        
        # Limita temperatura a valores realistas
        self.current_temp = np.clip(self.current_temp, 10.0, 40.0)
        
        # Atualiza ocupação (variação aleatória)
        if np.random.random() < 0.1:  # 10% de chance de mudança
            self.occupancy = np.random.randint(0, self.config.max_occupancy + 1)
        
        # Atualiza tempo
        self.time_step += 1
        self.hour_of_day = (self.hour_of_day + 1) % 24
        
        # Calcula recompensa
        reward = self._calculate_reward()
        
        # Verifica se episódio terminou (simulação de 24 horas)
        done = self.time_step >= 240  # 24 horas com dt=0.1
        
        # Atualiza histórico
        self.temp_history.append(self.current_temp)
        self.energy_history.append(self._calculate_energy_consumption())
        self.comfort_history.append(self._get_comfort_level())
        self.ac_state_history.append(self.ac_state)
        
        # Informações adicionais
        info = {
            'temperature': self.current_temp,
            'occupancy': self.occupancy,
            'ac_state': self.ac_state.name,
            'comfort_level': self._get_comfort_level().name,
            'energy_consumption': self._calculate_energy_consumption(),
            'hour': self.hour_of_day
        }
        
        return self._discretize_state(), reward, done, info
    
    def get_state_info(self, state: int) -> Dict:
        """Converte estado discreto de volta para informações legíveis"""
        # Reverte discretização
        time_idx = state % self.n_time_states
        temp_occ_state = state // self.n_time_states
        occ_idx = temp_occ_state % self.n_occupancy_states
        temp_idx = temp_occ_state // self.n_occupancy_states
        
        temp = (self.temp_bins[temp_idx] + self.temp_bins[temp_idx + 1]) / 2
        occupancy = (self.occupancy_bins[occ_idx] + self.occupancy_bins[occ_idx + 1]) / 2
        hour = time_idx
        
        return {
            'temperature': temp,
            'occupancy': occupancy,
            'hour': hour
        }
    
    def render(self, save_path: str = None):
        """Visualiza o estado atual do ambiente"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Gráfico de temperatura ao longo do tempo
        axes[0, 0].plot(self.temp_history)
        axes[0, 0].axhline(y=self.config.temp_comfort_min, color='g', linestyle='--', alpha=0.7, label='Conforto Min')
        axes[0, 0].axhline(y=self.config.temp_comfort_max, color='g', linestyle='--', alpha=0.7, label='Conforto Max')
        axes[0, 0].set_title('Temperatura da Sala')
        axes[0, 0].set_xlabel('Passos de Tempo')
        axes[0, 0].set_ylabel('Temperatura (°C)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Consumo energético
        axes[0, 1].plot(self.energy_history)
        axes[0, 1].set_title('Consumo Energético do AC')
        axes[0, 1].set_xlabel('Passos de Tempo')
        axes[0, 1].set_ylabel('Consumo (kW)')
        axes[0, 1].grid(True)
        
        # Estado do AC
        ac_states = [state.value for state in self.ac_state_history]
        axes[1, 0].plot(ac_states)
        axes[1, 0].set_title('Estado do Ar-Condicionado')
        axes[1, 0].set_xlabel('Passos de Tempo')
        axes[1, 0].set_ylabel('Estado (0=OFF, 1=LOW, 2=MED, 3=HIGH)')
        axes[1, 0].set_ylim(-0.5, 3.5)
        axes[1, 0].grid(True)
        
        # Nível de conforto
        comfort_values = [level.value for level in self.comfort_history]
        axes[1, 1].plot(comfort_values)
        axes[1, 1].set_title('Nível de Conforto Térmico')
        axes[1, 1].set_xlabel('Passos de Tempo')
        axes[1, 1].set_ylabel('Conforto (0=Very Cold, 4=Very Hot)')
        axes[1, 1].set_ylim(-0.5, 4.5)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas do episódio"""
        return {
            'avg_temperature': np.mean(self.temp_history),
            'temp_std': np.std(self.temp_history),
            'comfort_percentage': np.mean([c == ComfortLevel.COMFORTABLE for c in self.comfort_history]) * 100,
            'total_energy_consumption': np.sum(self.energy_history),
            'avg_energy_per_hour': np.mean(self.energy_history),
            'ac_usage_percentage': np.mean([s != ACState.OFF for s in self.ac_state_history]) * 100
        }

# Exemplo de uso
if __name__ == "__main__":
    # Cria ambiente
    config = ClassroomConfig()
    env = ClassroomACEnvironment(config)
    
    # Testa ambiente
    state = env.reset()
    print(f"Estado inicial: {state}")
    print(f"Informações do estado: {env.get_state_info(state)}")
    
    # Executa algumas ações aleatórias
    for i in range(10):
        action = np.random.randint(0, env.n_actions)
        next_state, reward, done, info = env.step(action)
        print(f"Passo {i+1}: Ação={ACState(action).name}, "
              f"Temp={info['temperature']:.1f}°C, "
              f"Conforto={info['comfort_level']}, "
              f"Recompensa={reward:.2f}")
        
        if done:
            break
    
    # Mostra estatísticas
    stats = env.get_statistics()
    print(f"\nEstatísticas do episódio:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    # Renderiza visualização
    env.render()