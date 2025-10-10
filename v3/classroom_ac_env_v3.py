# classroom_ac_env_v3.py
# -*- coding: utf-8 -*-
"""
Ambiente de Sala de Aula (Versão 3) - Compatível com Gymnasium.

Esta versão herda de `gymnasium.Env` e implementa a API padrão,
tornando-a compatível com bibliotecas como o Stable-Baselines3.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List
from enum import Enum

# --- DEFINIÇÕES AUXILIARES (COPIADAS DA VERSÃO ANTERIOR) ---
class ACState(Enum):
    OFF, LOW, MEDIUM, HIGH = 0, 1, 2, 3

class ComfortLevel(Enum):
    VERY_COLD, COLD, COMFORTABLE, WARM, VERY_HOT = 0, 1, 2, 3, 4

@dataclass
class ClassroomConfig:
    # --- PARÂMETROS DE RECOMPENSA APRIMORADOS ---
    # Define o tipo de cálculo para a recompensa de conforto ('step' ou 'quadratic')
    comfort_reward_type: str = 'quadratic'
    
    # Ponto central da faixa de conforto, onde a recompensa é máxima
    ideal_temp: float = 24.0
    
    # Bônus máximo ao atingir a temperatura ideal no modo quadrático
    comfort_bonus: float = 5.0
    
    # Controla o quão "severa" é a penalidade quadrática. Valores maiores = mais severa.
    comfort_sensitivity: float = 0.5
    
    # Fator de escala para a penalidade de energia
    energy_penalty_factor: float = 0.1
    
    # Penalidade por mudar a ação do AC
    action_change_penalty: float = -5.0

    length: float = 8.0; width: float = 6.0; height: float = 3.0
    max_occupancy: int = 30
    thermal_mass: float = 10.0
    heat_transfer_coeff: float = 0.2
    heat_gain_per_person: float = 0.1
    temp_comfort_min: float = 22.0; temp_comfort_max: float = 26.0
    temp_very_cold: float = 18.0; temp_very_hot: float = 30.0
    initial_temp: float = 24.0
    season: str = 'summer'
    energy_cost_peak_hours: Tuple[int, int] = (18, 21)
    energy_penalty_multiplier_peak: float = 1.5
    ac_cooling_power: Dict[ACState, float] = field(default_factory=lambda: {ACState.OFF: 0.0, ACState.LOW: 8.0, ACState.MEDIUM: 16.0, ACState.HIGH: 24.0})
    ac_energy_consumption: Dict[ACState, float] = field(default_factory=lambda: {ACState.OFF: 0.0, ACState.LOW: 1.5, ACState.MEDIUM: 3.0, ACState.HIGH: 5.0})
    reward_structure: Dict[str, float] = field(default_factory=lambda: {'VERY_COLD': -15.0, 'COLD': -5.0, 'COMFORTABLE': 1.0, 'WARM': -5.0, 'VERY_HOT': -15.0})

# --- CLASSE PRINCIPAL DO AMBIENTE (REESCRITA PARA GYMNASIUM) ---
class ClassroomACEnv(gym.Env):
    metadata = {"render_modes": ["human"]} 

    def __init__(self, config: ClassroomConfig = None, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config or ClassroomConfig()
        self.render_mode = render_mode # Armazena o modo de renderização

        # --- MELHORIA SUGERIDA: Lógica de autocorreção dos dicionários ---
        def _rebuild_ac_dict(d: dict) -> dict:
            """Converte chaves de texto ('OFF') para chaves de Enum (ACState.OFF)."""
            # Se o dicionário já estiver no formato correto, não faz nada.
            if not d or isinstance(list(d.keys())[0], ACState):
                return d
            
            rebuilt = {}
            for key_str, value in d.items():
                try:
                    # Tenta converter a string para um membro da Enum ACState
                    key_enum = ACState[key_str.upper()]
                    rebuilt[key_enum] = value
                except KeyError:
                    print(f"Aviso: Chave '{key_str}' no dicionário do AC é inválida e será ignorada.")
            return rebuilt
        
        self.config.ac_cooling_power = _rebuild_ac_dict(self.config.ac_cooling_power)
        self.config.ac_energy_consumption = _rebuild_ac_dict(self.config.ac_energy_consumption)
        
        # 1. DEFINIÇÃO DOS ESPAÇOS DE AÇÃO E OBSERVAÇÃO (OBRIGATÓRIO)
        self.action_space = spaces.Discrete(len(ACState))
        low = np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # Atributos internos da simulação
        self._init_simulation_vars()

    def _init_simulation_vars(self):
        """Inicializa ou reseta as variáveis de estado da simulação."""
        self.current_temp = self.config.initial_temp
        self.occupancy = 0
        self.ac_state = ACState.OFF
        self.time_step = 0
        self.hour_of_day = 8
        self.dt = 0.1 # Simula passos de 6 minutos
        self.history = []

    def _get_obs(self) -> np.ndarray:
        """Retorna a observação atual do agente no formato normalizado."""
        norm_temp = np.clip((self.current_temp - 15) / (35 - 15), 0, 1)
        norm_occ = self.occupancy / self.config.max_occupancy
        sin_hour = np.sin(2 * np.pi * self.hour_of_day / 24)
        cos_hour = np.cos(2 * np.pi * self.hour_of_day / 24)
        return np.array([norm_temp, norm_occ, sin_hour, cos_hour], dtype=np.float32)

    def _get_info(self) -> Dict:
        """Retorna dados de diagnóstico (não usados pelo agente para decidir)."""
        return {
            'temperature': self.current_temp,
            'occupancy': self.occupancy,
            'ac_state': self.ac_state.name,
            'hour': self.hour_of_day
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._init_simulation_vars()
        
        # 1. Define valores aleatórios como padrão (usado durante o TREINAMENTO)
        self.current_temp = self.np_random.uniform(18.0, 30.0)
        self.occupancy = self.np_random.integers(0, self.config.max_occupancy + 1)
        self.hour_of_day = self.np_random.integers(0, 24)

        # Sobrescreve os valores se 'options' for fornecido (usado na ANÁLISE) ---
        if options is not None:
            self.current_temp = options.get('start_temp', self.current_temp)
            self.occupancy = options.get('occupancy', self.occupancy)
            self.hour_of_day = options.get('hour_of_day', self.hour_of_day)


        observation = self._get_obs()
        info = self._get_info()
        self.history = [info]
        
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        previous_ac_state = self.ac_state
        self.ac_state = ACState(action)
        
        outdoor_temp = self._calculate_outdoor_temp()
        people_heat = self.occupancy * self.config.heat_gain_per_person
        external_heat = self.config.heat_transfer_coeff * (outdoor_temp - self.current_temp)
        cooling_effect = self.config.ac_cooling_power[self.ac_state]
        
        total_heat_gain = people_heat + external_heat
        net_heat = total_heat_gain - cooling_effect
        
        temp_change = (net_heat / self.config.thermal_mass) * self.dt
        
        noise = self.np_random.normal(0, 0.01)
        self.current_temp += temp_change + noise
        self.current_temp = np.clip(self.current_temp, 10.0, 40.0)
        
        self.time_step += 1
        self.hour_of_day = (self.hour_of_day + 1) % 24
        if self.np_random.random() < 0.1:
            self.occupancy = np.clip(self.occupancy + self.np_random.integers(-5, 6), 0, self.config.max_occupancy)

        reward = self._calculate_reward(previous_ac_state)
        terminated = self.time_step >= 240
        truncated = False

        observation = self._get_obs()
        info = self._get_info()
        info['debug_total_heat_gain'] = total_heat_gain
        info['debug_cooling_effect'] = cooling_effect
        info['debug_net_heat'] = net_heat
        
        # --- CORREÇÃO DO BUG DA VÍRGULA ---
        # Removido a vírgula do final das linhas abaixo
        info['temperature'] = self.current_temp
        info['occupancy'] = self.occupancy
        info['ac_state'] = self.ac_state.name
        info['debug_temp_change'] = temp_change

        self.history.append(info)
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            last_info = self.history[-1]
            print(f"Passo: {self.time_step}, Hora: {last_info['hour']}, Temp: {last_info['temperature']:.2f}°C, Ocup.: {last_info['occupancy']}, Ação: {last_info['ac_state']}")
    def close(self):
        pass # Limpeza de recursos, se necessário

    def _get_comfort_level(self) -> ComfortLevel:
        if self.current_temp < self.config.temp_very_cold: return ComfortLevel.VERY_COLD
        elif self.current_temp < self.config.temp_comfort_min: return ComfortLevel.COLD
        elif self.current_temp <= self.config.temp_comfort_max: return ComfortLevel.COMFORTABLE
        elif self.current_temp < self.config.temp_very_hot: return ComfortLevel.WARM
        else: return ComfortLevel.VERY_HOT

    def _calculate_outdoor_temp(self) -> float:
        base_temp, amplitude = (28.0, 8.0) if self.config.season == 'summer' else (20.0, 6.0)
        phase = (self.hour_of_day - 14) * np.pi / 12
        return base_temp + amplitude * np.cos(phase)

    def _calculate_reward(self, previous_ac_state: ACState) -> float:
        """
        Calcula a recompensa com base na configuração do ambiente.
        Pode usar um método por faixas ('step') ou um método contínuo ('quadratic').
        """
        
        # --- 1. CÁLCULO DA RECOMPENSA DE CONFORTO ---
        if self.config.comfort_reward_type == 'quadratic':
            # Abordagem Contínua (Quadrática)
            delta_temp = self.current_temp - self.config.ideal_temp
            
            # A recompensa é uma parábola invertida: -k * (T - T_ideal)^2
            # O bônus máximo é aplicado no topo.
            comfort_reward = self.config.comfort_bonus - self.config.comfort_sensitivity * (delta_temp ** 2)
            
        else: # 'step'
            # Abordagem por Faixas (legado)
            comfort_level = self._get_comfort_level()
            comfort_reward = self.config.reward_structure[comfort_level.name]

        # --- 2. CÁLCULO DA PENALIDADE DE ENERGIA ---
        energy_consumption = self.config.ac_energy_consumption[self.ac_state]
        energy_penalty_multiplier = 1.0
        peak_start, peak_end = self.config.energy_cost_peak_hours
        if peak_start <= self.hour_of_day <= peak_end:
            energy_penalty_multiplier = self.config.energy_penalty_multiplier_peak
            
        energy_penalty = -energy_consumption * self.config.energy_penalty_factor * energy_penalty_multiplier

        # --- 3. CÁLCULO DA PENALIDADE POR MUDANÇA DE AÇÃO ---
        action_change_penalty = self.config.action_change_penalty if self.ac_state != previous_ac_state else 0.0
        
        # --- 4. RECOMPENSA FINAL ---
        return comfort_reward + energy_penalty + action_change_penalty