# -*- coding: utf-8 -*-
"""
Configuração do ambiente e os três perfis operacionais (Tabela 2 do paper).

NOTA DE REPRODUÇÃO — parâmetros inferidos
------------------------------------------
O paper especifica numericamente `B_c`, a penalidade de energia e a de troca
(Tabela 2), mas NÃO publica `B` (bônus base), `k` (curvatura fora da faixa),
`rho` (peso do anti-short-cycling) nem a penalidade de frio. Os valores abaixo
foram inferidos da Figura 1 e estão marcados com `# INFERIDO`:

  B = 10.0  -> a Figura 1 mostra as bordas do platô (22 e 26 °C) em ≈ +10.
  k = 0.6   -> em 32 °C a curva laranja cai a ≈ −11: 10 − k·6² = −11 => k ≈ 0,58.
               0,6 também é o valor usado no v4 original (comfort_sensitivity).
  rho = -5.0 -> não observável na Figura 1. Escolhido na ordem de grandeza do
               conforto, para conter a comutação sem dominar a recompensa.
  cold_action_penalty = -2.0 -> a Conclusão do paper alerta que uma penalidade
               severa no frio induziu "medo de resfriar" e degradou a política.
               Mantida deliberadamente branda (o v4 original usava -20).

Qualquer divergência dos números da Tabela 4 deve ser lida à luz destes quatro
parâmetros antes de se suspeitar da implementação.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

from ac_physics import ACPhysicsModel


@dataclass
class ClassroomConfig:
    """Parâmetros da sala, da recompensa e da tarifa."""

    # --- Geometria e ocupação (Seção 4.1: sala fictícia, 45 ocupantes) ---
    max_occupancy: int = 45
    heat_gain_per_person: float = 0.3   # 45 · 0,3 = 13,5 u -> carga dominante

    # --- Dinâmica térmica (eq. 2) ---
    thermal_mass: float = 15.0          # C_th da sala de treino (Seção 5.5)
    heat_transfer_coeff: float = 0.5    # K_transf: 0,5·(36−24) = 6 u no pico
    temperature_noise_std: float = 0.01
    dt: float = 0.1                     # 0,1 h = 6 min por passo
    episode_steps: int = 240            # 240 · 0,1 h = 24 h
    temp_min_clip: float = 10.0
    temp_max_clip: float = 40.0

    # --- Temperatura externa senoidal (pico às 14h) ---
    season: str = "summer"
    outdoor_base_temp: float = 28.0
    outdoor_amplitude: float = 8.0
    outdoor_peak_hour: float = 14.0

    # --- Faixas de conforto ---
    ideal_temp: float = 24.0
    temp_comfort_min: float = 22.0
    temp_comfort_max: float = 26.0
    temp_narrow_min: float = 23.0       # faixa estreita [23,25] da Tabela 4
    temp_narrow_max: float = 25.0

    # --- Janela ocupada (Seção 4.4) ---
    occupied_hour_start: int = 7
    occupied_hour_end: int = 22

    # --- Recompensa de conforto (eq. 4) ---
    # Topologia: 'plateau' é a eq. 4 do paper (platô + gradiente interno);
    # 'quadratic' e 'step' existem para a ABLAÇÃO exigida pelo Revisor 2 —
    # sem poder desligar a contribuição, a tese "recompensa > algoritmo" não é
    # verificável. Com comfort_gradient = 0, 'plateau' vira o platô plano, que é
    # a hipótese explícita do paper para o "estacionar na borda".
    comfort_type: str = "plateau"          # 'plateau' | 'quadratic' | 'step'
    comfort_bonus: float = 10.0            # B          # INFERIDO
    comfort_gradient: float = 4.0          # B_c        (Tabela 2: perfil)
    comfort_sensitivity: float = 0.6       # k          # INFERIDO

    # --- Demais termos da recompensa ---
    energy_penalty_factor: float = 0.10    # (Tabela 2: perfil)
    action_change_penalty: float = -1.5    # (Tabela 2: perfil)
    cold_action_penalty: float = -2.0                   # INFERIDO
    short_cycle_penalty: float = -5.0      # rho        # INFERIDO
    min_dwell_minutes: float = 36.0        # d_min (eq. 5)

    # --- Tarifa (Seção 5) ---
    energy_tariff_brl_per_kwh: float = 0.80
    peak_hours: Tuple[int, int] = (18, 21)
    peak_tariff_multiplier: float = 1.6
    # O paper escala a PENALIDADE de energia no pico (Seção 4.3) e a TARIFA
    # no pico (Seção 5). Mantidos separados de propósito: um é sinal de
    # treino, o outro é métrica de avaliação.
    peak_penalty_multiplier: float = 1.6

    # --- Safety shield (Seção 5.4) ---
    shield_force_cool_above: float = 28.0
    shield_force_off_below: float = 20.0

    # --- Modelo do equipamento ---
    physics: ACPhysicsModel = field(default_factory=ACPhysicsModel)

    @property
    def min_dwell_steps(self) -> int:
        """d_min convertido para passos: 36 min / 6 min = 6 passos."""
        return int(round(self.min_dwell_minutes / (self.dt * 60.0)))

    def is_occupied_hour(self, hour: int) -> bool:
        return self.occupied_hour_start <= hour < self.occupied_hour_end

    def is_peak_hour(self, hour: int) -> bool:
        return self.peak_hours[0] <= hour <= self.peak_hours[1]


# --- Tabela 2: os três perfis operacionais -------------------------------
# Obtidos APENAS variando pesos da recompensa — arquitetura e hiperparâmetros
# do DQN são idênticos entre perfis. É esta a tese central do paper:
# "a modelagem da recompensa, mais que o algoritmo, é o fator determinante".
REWARD_PROFILES: Dict[str, Dict[str, float]] = {
    "Agressivo": {
        "comfort_gradient": 7.0,
        "energy_penalty_factor": 0.03,
        "action_change_penalty": -0.5,
    },
    "Equilibrado": {
        "comfort_gradient": 4.0,
        "energy_penalty_factor": 0.10,
        "action_change_penalty": -1.5,
    },
    "Passivo": {
        "comfort_gradient": 3.0,
        "energy_penalty_factor": 0.12,
        "action_change_penalty": -1.5,
    },
}

# --- Tabela 6: variações físicas para o teste de generalização ------------
ROOM_VARIATIONS: Dict[str, Dict[str, float]] = {
    "C_th=10 (leve)":            {"thermal_mass": 10.0, "heat_transfer_coeff": 0.5},
    "C_th=15 (treino)":          {"thermal_mass": 15.0, "heat_transfer_coeff": 0.5},
    "C_th=20":                   {"thermal_mass": 20.0, "heat_transfer_coeff": 0.5},
    "C_th=30 (pesada)":          {"thermal_mass": 30.0, "heat_transfer_coeff": 0.5},
    "K=0,3 (bem isolada)":       {"thermal_mass": 15.0, "heat_transfer_coeff": 0.3},
    "K=0,8 (mal isolada)":       {"thermal_mass": 15.0, "heat_transfer_coeff": 0.8},
}


def config_for_profile(profile: str, **overrides) -> ClassroomConfig:
    """Constrói a config de um perfil da Tabela 2, com overrides opcionais."""
    if profile not in REWARD_PROFILES:
        raise KeyError(f"Perfil '{profile}' inexistente. Use: {list(REWARD_PROFILES)}")
    return replace(ClassroomConfig(), **{**REWARD_PROFILES[profile], **overrides})
