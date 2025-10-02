# -*- coding: utf-8 -*-
"""
Ambiente de Sala de Aula com Ar-Condicionado para Aprendizagem por Reforço

Este módulo implementa um ambiente simulado de uma sala de aula com sistema de ar-condicionado,
onde um agente de RL deve aprender a controlar a temperatura de forma a balancear:
- Conforto térmico dos usuários
- Eficiência energética

Autor: Renan Saraiva dos Santos
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt

class ACState(Enum):
    """Estados do ar-condicionado"""
    OFF, LOW, MEDIUM, HIGH = 0, 1, 2, 3

class ComfortLevel(Enum):
    """Níveis de conforto térmico"""
    VERY_COLD, COLD, COMFORTABLE, WARM, VERY_HOT = 0, 1, 2, 3, 4

@dataclass
class ClassroomConfig:
    """Configuração da sala de aula"""
    # Dimensões da sala (metros)
    length: float = 8.0; width: float = 6.0; height: float = 3.0
    max_occupancy: int = 30
    thermal_mass: float = 1000.0
    heat_transfer_coeff: float = 0.5
    heat_gain_per_person: float = 0.1
    temp_comfort_min: float = 22.0; temp_comfort_max: float = 26.0
    temp_very_cold: float = 18.0; temp_very_hot: float = 30.0
    initial_temp: float = 24.0
    ac_cooling_power: Dict[ACState, float] = None
    ac_energy_consumption: Dict[ACState, float] = None

    # Usamos strings como chaves para facilitar a serialização para JSON.
    reward_structure: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # (Lógica de autocorreção dos dicionários de AC, como na versão anterior)
        def rebuild_dict(d: dict) -> dict:
            if not d or not isinstance(list(d.keys())[0], str): return d
            rebuilt = {}
            for k, v in d.items():
                try: key_enum = ACState(int(k))
                except ValueError: key_name = k.split('.')[-1]; key_enum = ACState[key_name]
                rebuilt[key_enum] = v
            return rebuilt

        if self.ac_cooling_power is None: self.ac_cooling_power = {ACState.OFF: 0.0, ACState.LOW: 2.0, ACState.MEDIUM: 4.0, ACState.HIGH: 6.0}
        else: self.ac_cooling_power = rebuild_dict(self.ac_cooling_power)
        
        if self.ac_energy_consumption is None: self.ac_energy_consumption = {ACState.OFF: 0.0, ACState.LOW: 1.5, ACState.MEDIUM: 3.0, ACState.HIGH: 5.0}
        else: self.ac_energy_consumption = rebuild_dict(self.ac_energy_consumption)

        # Define os valores padrão para a estrutura de recompensa, se não for fornecida.
        if not self.reward_structure:
            self.reward_structure = {
                'VERY_COLD': -15.0, 'COLD': -5.0,
                'COMFORTABLE': 1.0,
                'WARM': -5.0, 'VERY_HOT': -15.0
            }

        # Bloco de inicialização
        if self.ac_cooling_power is None:
            self.ac_cooling_power = {ACState.OFF: 0.0, ACState.LOW: 2.0, ACState.MEDIUM: 4.0, ACState.HIGH: 6.0}
        else:
            # Se não for None, pode ter vindo de um JSON, então tentamos reconstruí-lo
            self.ac_cooling_power = rebuild_dict(self.ac_cooling_power)
        
        if self.ac_energy_consumption is None:
            self.ac_energy_consumption = {ACState.OFF: 0.0, ACState.LOW: 1.5, ACState.MEDIUM: 3.0, ACState.HIGH: 5.0}
        else:
            # Faz o mesmo para o dicionário de consumo de energia
            self.ac_energy_consumption = rebuild_dict(self.ac_energy_consumption)

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
        self.current_temp = self.config.initial_temp
        self.occupancy, self.ac_state, self.time_step, self.hour_of_day = 0, ACState.OFF, 0, 8
        self.previous_action = ACState.OFF
        self.dt, self.temperature_noise_std = 0.1, 0.1
        self.temp_bins = np.linspace(15, 35, 21) 
        self.occupancy_bins, self.time_bins = [0, 5, 10, 15, 20, 25, 30], list(range(24))
        self.actions, self.n_actions = list(ACState), len(ACState)
        
        self.n_temp_states = len(self.temp_bins) - 1
        self.n_occupancy_states = len(self.occupancy_bins) - 1
        
        # --- INÍCIO DA CORREÇÃO ---
        # A atribuição em uma única linha foi separada para corrigir o AttributeError.
        # Primeiro, definimos self.n_time_states.
        self.n_time_states = len(self.time_bins)
        # Em seguida, usamos o valor recém-definido para calcular o total de estados.
        self.n_states = (self.n_temp_states * self.n_occupancy_states * self.n_time_states)
        # --- FIM DA CORREÇÃO ---
        
        self.temp_history, self.energy_history, self.comfort_history, self.ac_state_history = [], [], [], []
        
        
    def _discretize_state(self) -> int:
        temp_idx = np.clip(np.digitize(self.current_temp, self.temp_bins) - 1, 0, self.n_temp_states - 1)
        occ_idx = np.clip(np.digitize(self.occupancy, self.occupancy_bins) - 1, 0, self.n_occupancy_states - 1)
        time_idx = self.hour_of_day % 24
        return (temp_idx * self.n_occupancy_states * self.n_time_states + occ_idx * self.n_time_states + time_idx)
    
    
    def _get_comfort_level(self) -> ComfortLevel:
        if self.current_temp < self.config.temp_very_cold: return ComfortLevel.VERY_COLD
        elif self.current_temp < self.config.temp_comfort_min: return ComfortLevel.COLD
        elif self.current_temp <= self.config.temp_comfort_max: return ComfortLevel.COMFORTABLE
        elif self.current_temp < self.config.temp_very_hot: return ComfortLevel.WARM
        else: return ComfortLevel.VERY_HOT
    
    def _calculate_outdoor_temp(self) -> float:
        """Calcula temperatura externa baseada na hora do dia"""
        # Simula variação diária da temperatura externa
        base_temp = 22.0  # Média diária um pouco mais alta
        amplitude = 10.0  # Aumentado de 8.0 (diferença maior entre dia e noite)
        phase = (self.hour_of_day - 6) * np.pi / 12
        return base_temp + amplitude * np.sin(phase)
    
    def _calculate_heat_gain(self) -> float:
        """Calcula ganho de calor total na sala"""
        # --- MODIFICAÇÃO 3: Agora lê o valor do objeto de configuração ---
        people_heat = self.occupancy * self.config.heat_gain_per_person
        outdoor_temp = self._calculate_outdoor_temp()
        external_heat = self.config.heat_transfer_coeff * (outdoor_temp - self.current_temp)
        return people_heat + external_heat
    
    def _calculate_cooling_effect(self) -> float:
        """Calcula efeito de refrigeração do ar-condicionado"""
        return self.config.ac_cooling_power[self.ac_state]
    
    def _calculate_energy_consumption(self) -> float:
        """Calcula consumo energético do ar-condicionado"""
        return self.config.ac_energy_consumption[self.ac_state]
    
    def _calculate_reward(self, previous_ac_state: ACState) -> float:
        comfort_level = self._get_comfort_level()
        energy_consumption = self._calculate_energy_consumption()
        
        # Lê a recompensa por conforto diretamente da configuração do ambiente, usando o nome do enum como chave.
        comfort_reward = self.config.reward_structure[comfort_level.name]
        
        energy_penalty = -energy_consumption * 0.1
        action_change_penalty = -2.0 if self.ac_state != previous_ac_state else 0.0
        
        return comfort_reward + energy_penalty + action_change_penalty
    
    def reset(self, start_temp: Optional[float] = None) -> int:
        self.current_temp = start_temp if start_temp is not None else self.config.initial_temp
        self.occupancy = np.random.randint(0, self.config.max_occupancy + 1)
        self.ac_state, self.time_step, self.hour_of_day = ACState.OFF, 0, 8
        self.previous_action = ACState.OFF
        self.temp_history = [self.current_temp]
        self.energy_history, self.comfort_history, self.ac_state_history = [0.0], [self._get_comfort_level()], [self.ac_state]
        return self._discretize_state()
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        previous_ac_state = self.ac_state
        self.ac_state = ACState(action)
        
        outdoor_temp = 22.0 + 10.0 * np.sin((self.hour_of_day - 6) * np.pi / 12)
        people_heat = self.occupancy * self.config.heat_gain_per_person
        external_heat = self.config.heat_transfer_coeff * (outdoor_temp - self.current_temp)
        cooling_effect = self._calculate_cooling_effect()
        net_heat = people_heat + external_heat - cooling_effect
        
        temp_change = (net_heat / self.config.thermal_mass) * self.dt
        noise = np.random.normal(0, self.temperature_noise_std)
        self.current_temp += temp_change + noise
        self.current_temp = np.clip(self.current_temp, 10.0, 40.0)
        
        if np.random.random() < 0.4: self.occupancy = np.random.randint(0, self.config.max_occupancy + 1)
        self.time_step += 1
        self.hour_of_day = (self.hour_of_day + 1) % 24
        reward = self._calculate_reward(previous_ac_state)
        done = self.time_step >= 240
        
        self.temp_history.append(self.current_temp)
        self.energy_history.append(self._calculate_energy_consumption())
        self.comfort_history.append(self._get_comfort_level())
        self.ac_state_history.append(self.ac_state)
        
        info = {'temperature': self.current_temp, 'occupancy': self.occupancy, 'ac_state': self.ac_state.name,
                'comfort_level': self._get_comfort_level().name, 'energy_consumption': self.config.ac_energy_consumption[self.ac_state],
                'hour': self.hour_of_day}
        
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
        
        return {'temperature': temp, 'occupancy': occupancy, 'hour': hour}
    
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
            'avg_temperature': np.mean(self.temp_history), 'temp_std': np.std(self.temp_history),
            'comfort_percentage': np.mean([c == ComfortLevel.COMFORTABLE for c in self.comfort_history]) * 100,
            'total_energy_consumption': np.sum(self.energy_history) * self.dt,
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