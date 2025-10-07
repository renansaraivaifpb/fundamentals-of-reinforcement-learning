# classroom_ac_env.py
# -*- coding: utf-8 -*-
"""
Ambiente de Sala de Aula com Ar-Condicionado para Aprendizagem por Reforço (Versão Aprimorada)

Melhorias nesta versão:
- Adição de variações sazonais (verão/inverno) para a temperatura externa.
- Estrutura de recompensa dinâmica com penalidade de energia variável por hora do dia.
- Randomização da hora inicial no reset para maior variabilidade no treinamento.

Autor: Renan Saraiva dos Santos 
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List
from enum import Enum
import matplotlib.pyplot as plt


# --- MODIFICAÇÃO: Usando um logger padrão para melhor rastreamento ---
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    # --- MODIFICAÇÃO: Parâmetros sazonais e de custo de energia ---
    season: str = 'summer'  # 'summer' ou 'winter'
    energy_cost_peak_hours: Tuple[int, int] = (18, 21)
    energy_penalty_multiplier_peak: float = 1.5 # Penalidade 50% maior no horário de pico
    
    ac_cooling_power: Dict[ACState, float] = field(default_factory=lambda: {ACState.OFF: 0.0, ACState.LOW: 2.0, ACState.MEDIUM: 4.0, ACState.HIGH: 6.0})
    ac_energy_consumption: Dict[ACState, float] = field(default_factory=lambda: {ACState.OFF: 0.0, ACState.LOW: 1.5, ACState.MEDIUM: 3.0, ACState.HIGH: 5.0})
    reward_structure: Dict[str, float] = field(default_factory=lambda: {'VERY_COLD': -15.0, 'COLD': -5.0, 'COMFORTABLE': 1.0, 'WARM': -5.0, 'VERY_HOT': -15.0})

    
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
                'VERY_COLD': -19.0, 'COLD': -10.0,
                'COMFORTABLE': 8.0,
                'WARM': -5.0, 'VERY_HOT': -19.0
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
        self.occupancy = 0
        self.ac_state = ACState.OFF
        self.time_step = 0
        self.hour_of_day = 8
        self.dt = 0.1
        self.temperature_noise_std = 0.1
        self.temp_bins = np.linspace(15, 35, 21)
        self.occupancy_bins = np.array([0, 5, 10, 15, 20, 25, 30])
        self.time_bins = np.arange(24)
        self.n_actions = len(ACState)
        self.n_temp_states = len(self.temp_bins) - 1
        self.n_occupancy_states = len(self.occupancy_bins)
        self.n_time_states = len(self.time_bins)
        self.n_states = self.n_temp_states * self.n_occupancy_states * self.n_time_states
        self.history: Dict[str, List] = {}
        
    def _discretize_state(self) -> int:
        """Discretiza o estado contínuo para uso em Q-Learning tabular."""
        temp_idx = np.clip(np.digitize(self.current_temp, self.temp_bins) - 1, 0, self.n_temp_states - 1)
        occ_idx = np.clip(np.digitize(self.occupancy, self.occupancy_bins) -1, 0, self.n_occupancy_states -1)
        time_idx = self.hour_of_day
        
        state_index = np.ravel_multi_index((temp_idx, occ_idx, time_idx), 
                                           (self.n_temp_states, self.n_occupancy_states, self.n_time_states))
        return int(state_index)
    
    def get_continuous_state(self) -> np.ndarray:
        """Retorna o estado na forma contínua para uso em DQN."""
        # Normalização é crucial para redes neurais
        norm_temp = (self.current_temp - 15) / (35 - 15)
        norm_occ = self.occupancy / self.config.max_occupancy
        # Usar seno/cosseno para capturar a natureza cíclica do tempo
        sin_hour = np.sin(2 * np.pi * self.hour_of_day / 24)
        cos_hour = np.cos(2 * np.pi * self.hour_of_day / 24)
        return np.array([norm_temp, norm_occ, sin_hour, cos_hour], dtype=np.float32)

    def _get_comfort_level(self) -> ComfortLevel:
        if self.current_temp < self.config.temp_very_cold: return ComfortLevel.VERY_COLD
        elif self.current_temp < self.config.temp_comfort_min: return ComfortLevel.COLD
        elif self.current_temp <= self.config.temp_comfort_max: return ComfortLevel.COMFORTABLE
        elif self.current_temp < self.config.temp_very_hot: return ComfortLevel.WARM
        else: return ComfortLevel.VERY_HOT
    
    def _calculate_outdoor_temp(self) -> float:
        """Calcula temperatura externa com base na hora e na estação do ano."""
        # --- MODIFICAÇÃO: Adiciona variação sazonal ---
        if self.config.season == 'summer':
            base_temp, amplitude = 28.0, 8.0 # Dia de verão quente
        else: # winter
            base_temp, amplitude = 20.0, 6.0 # Dia de inverno ameno
            
        # Simula variação diária da temperatura externa
        phase = (self.hour_of_day - 14) * np.pi / 12 # Pico de temperatura às 14h
        return base_temp - amplitude * np.cos(phase)
    
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
        """Calcula a recompensa com penalidade de energia dinâmica."""
        comfort_level = self._get_comfort_level()
        energy_consumption = self.config.ac_energy_consumption[self.ac_state]
        
        comfort_reward = self.config.reward_structure[comfort_level.name]
        
        # --- MODIFICAÇÃO: Penalidade de energia dinâmica ---
        energy_penalty_multiplier = 1.0
        peak_start, peak_end = self.config.energy_cost_peak_hours
        if peak_start <= self.hour_of_day <= peak_end:
            energy_penalty_multiplier = self.config.energy_penalty_multiplier_peak
        
        energy_penalty = -energy_consumption * 0.1 * energy_penalty_multiplier
        
        action_change_penalty = -2.0 if self.ac_state != previous_ac_state else 0.0
        
        return comfort_reward + energy_penalty + action_change_penalty
    
    def reset(self, start_temp: Optional[float] = None, options: Optional[Dict] = None) -> Tuple[int, Dict]:
        """Reseta o ambiente para um novo episódio."""
        self.current_temp = start_temp if start_temp is not None else self.config.initial_temp
        self.occupancy = np.random.randint(0, self.config.max_occupancy + 1)
        self.ac_state = ACState.OFF
        self.time_step = 0
        
        # --- MODIFICAÇÃO: Randomiza a hora de início para diversificar o treinamento ---
        self.hour_of_day = np.random.randint(6, 23) # Simula início em qualquer horário comercial
        
        # Permite sobrescrever parâmetros no reset (útil para avaliação)
        if options:
            self.current_temp = options.get('start_temp', self.current_temp)
            self.occupancy = options.get('occupancy', self.occupancy)
            self.hour_of_day = options.get('hour_of_day', self.hour_of_day)

        self.history = {'temperatures': [self.current_temp], 'ac_states': [self.ac_state.value], 
                        'rewards': [0.0], 'energy': [0.0], 'comfort_levels': [self._get_comfort_level().value]}
        
        # Retorna estado discretizado por padrão, mas a info contém o contínuo
        info = {'continuous_state': self.get_continuous_state()}
        return self._discretize_state(), info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """Executa uma ação no ambiente."""
        if action not in [s.value for s in ACState]:
            raise ValueError(f"Ação inválida: {action}")

        previous_ac_state = self.ac_state
        self.ac_state = ACState(action)
        
        # Física da atualização de temperatura (sem alterações)
        outdoor_temp = self._calculate_outdoor_temp()
        people_heat = self.occupancy * self.config.heat_gain_per_person
        external_heat = self.config.heat_transfer_coeff * (outdoor_temp - self.current_temp)
        cooling_effect = self.config.ac_cooling_power[self.ac_state]
        net_heat = people_heat + external_heat - cooling_effect
        
        temp_change = (net_heat / self.config.thermal_mass) * self.dt
        noise = np.random.normal(0, self.temperature_noise_std)
        self.current_temp += temp_change + noise
        self.current_temp = np.clip(self.current_temp, 10.0, 40.0)
        
        # Simula dinâmica de ocupação e tempo
        if np.random.random() < 0.1: # Chance de mudança de ocupação
            self.occupancy = np.clip(self.occupancy + np.random.randint(-5, 6), 0, self.config.max_occupancy)
        self.time_step += 1
        self.hour_of_day = (self.hour_of_day + 1) % 24
        
        # Cálculo da recompensa e término
        reward = self._calculate_reward(previous_ac_state)
        terminated = self.time_step >= 240 # Episódio de 24h (240 * 6 min)
        truncated = False # Não usado aqui
        
        # Logging
        energy_consumed = self.config.ac_energy_consumption[self.ac_state]
        self.history['temperatures'].append(self.current_temp)
        self.history['ac_states'].append(self.ac_state.value)
        self.history['rewards'].append(reward)
        self.history['energy'].append(energy_consumed)
        self.history['comfort_levels'].append(self._get_comfort_level().value)
        
        info = {
            'temperature': self.current_temp,
            'occupancy': self.occupancy,
            'ac_state': self.ac_state.name,
            'hour': self.hour_of_day,
            'comfort_level': self._get_comfort_level().name,
            'energy_consumption': energy_consumed,
            'continuous_state': self.get_continuous_state()
        }
        
        return self._discretize_state(), reward, terminated, truncated, info

    
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
        """Retorna estatísticas do episódio."""
        comfort_array = np.array(self.history['comfort_levels'])
        return {
            'avg_temperature': np.mean(self.history['temperatures']),
            'comfort_percentage': np.mean(comfort_array == ComfortLevel.COMFORTABLE.value) * 100,
            'total_energy_consumption': np.sum(self.history['energy']) * self.dt,
            'ac_usage_percentage': np.mean(np.array(self.history['ac_states']) > 0) * 100
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