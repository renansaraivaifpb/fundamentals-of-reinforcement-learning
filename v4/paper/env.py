# -*- coding: utf-8 -*-
"""
ClassroomACEnv — ambiente Gymnasium do paper (Seções 4.1 a 4.3).

Implementa:
  eq. 2  balanço térmico agregado (lumped)
  eq. 3  espaço de observação normalizado
  eq. 4  recompensa de conforto "Platô Quadrático com gradiente"
  eq. 5  penalidade anti-short-cycling
"""
from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ac_physics import ACState
from config import ClassroomConfig


class ComfortLevel(IntEnum):
    VERY_COLD, COLD, COMFORTABLE, WARM, VERY_HOT = 0, 1, 2, 3, 4


class ClassroomACEnv(gym.Env):
    """
    Sala de aula fictícia com AC de 4 níveis.

    Correções deliberadas em relação ao v4 original, todas necessárias para
    que a avaliação do paper seja válida:

    1. `reset(options=...)` aceita a chave `hour` (o v4 lia `hour_of_day` e
       descartava silenciosamente a hora dos cenários, avaliando tudo em
       horas aleatórias).
    2. A observação é sempre clipada em [0,1]; com 45 ocupantes o v4 emitia
       `o_norm = 1,33`, fora do `observation_space` declarado.
    3. A ocupação segue a janela ocupada (7h–22h) nos cenários de avaliação,
       em vez de um passeio aleatório — sem isso "conforto na janela ocupada"
       não é mensurável de forma reprodutível.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[ClassroomConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.config = config or ClassroomConfig()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(len(ACState))
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._init_state()

    # ------------------------------------------------------------------ setup

    def _init_state(self) -> None:
        cfg = self.config
        self.current_temp: float = cfg.ideal_temp
        self.occupancy: int = 0
        self.ac_state: ACState = ACState.OFF
        self.time_step: int = 0
        self.start_hour: int = 0
        self.hour_float: float = 0.0
        self.hour_of_day: int = 0
        self.steps_since_change: int = 10**6  # começa "estabilizado"
        self.scenario_occupancy: Optional[int] = None
        self.history: List[Dict] = []

    # ------------------------------------------------------------ observação

    def _get_obs(self) -> np.ndarray:
        """eq. 3 — t_norm = clip((T−15)/20, 0, 1); o_norm = N/N_max."""
        cfg = self.config
        t_norm = np.clip((self.current_temp - 15.0) / 20.0, 0.0, 1.0)
        o_norm = np.clip(self.occupancy / cfg.max_occupancy, 0.0, 1.0)
        angle = 2.0 * np.pi * self.hour_float / 24.0
        return np.array(
            [t_norm, o_norm, np.sin(angle), np.cos(angle)], dtype=np.float32
        )

    def _get_info(self) -> Dict:
        cfg = self.config
        kw = cfg.physics.electrical_kw(self.ac_state)
        return {
            "temperature": float(self.current_temp),
            "occupancy": int(self.occupancy),
            "ac_state": self.ac_state.name,
            "action": int(self.ac_state),
            "hour": int(self.hour_of_day),
            "hour_float": float(self.hour_float),
            "step": int(self.time_step),
            "occupied": cfg.is_occupied_hour(self.hour_of_day),
            "electrical_kw": kw,
            "energy_kwh": kw * cfg.dt,
            "cost_brl": kw * cfg.dt * self._tariff(),
            "outdoor_temp": self._outdoor_temp(),
        }

    # ----------------------------------------------------------------- reset

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        cfg = self.config
        self._init_state()

        # Padrão: amostragem aleatória (treino).
        self.start_hour = int(self.np_random.integers(0, 24))
        self.current_temp = float(self.np_random.uniform(18.0, 30.0))
        self.occupancy = int(self.np_random.integers(0, cfg.max_occupancy + 1))

        if options:
            # Aceita 'hour' (paper/scenarios.json) e 'hour_of_day' (legado v4).
            self.start_hour = int(
                options.get("hour", options.get("hour_of_day", self.start_hour))
            )
            self.current_temp = float(options.get("start_temp", self.current_temp))
            if "occupancy" in options:
                # Cenário fixa a lotação de pico; a janela ocupada modula.
                self.scenario_occupancy = int(options["occupancy"])
                self.occupancy = self._scheduled_occupancy(self.start_hour)

        self.hour_float = float(self.start_hour)
        self.hour_of_day = int(self.start_hour) % 24
        self.ac_state = ACState.OFF

        obs, info = self._get_obs(), self._get_info()
        self.history = [info]
        return obs, info

    # ------------------------------------------------------------------ step

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        cfg = self.config
        previous_state = self.ac_state
        self.ac_state = ACState(int(action))

        changed = self.ac_state != previous_state
        dwell = self.steps_since_change
        self.steps_since_change = 0 if changed else self.steps_since_change + 1

        # --- eq. 2: balanço térmico agregado ---
        outdoor = self._outdoor_temp()
        people_heat = self.occupancy * cfg.heat_gain_per_person
        external_heat = cfg.heat_transfer_coeff * (outdoor - self.current_temp)
        heat_gain = people_heat + external_heat
        cooling = cfg.physics.cooling_units(self.ac_state)
        net_heat = heat_gain - cooling

        noise = float(self.np_random.normal(0.0, cfg.temperature_noise_std))
        self.current_temp = float(
            np.clip(
                self.current_temp + (cfg.dt / cfg.thermal_mass) * net_heat + noise,
                cfg.temp_min_clip,
                cfg.temp_max_clip,
            )
        )

        # --- avanço do tempo ---
        self.time_step += 1
        self.hour_float = self.start_hour + self.time_step * cfg.dt
        self.hour_of_day = int(self.hour_float) % 24
        self._advance_occupancy()

        reward = self._reward(changed=changed, dwell_before_change=dwell)

        terminated = self.time_step >= cfg.episode_steps
        info = self._get_info()
        info.update(
            {
                "reward": reward,
                "people_heat": people_heat,
                "external_heat": external_heat,
                "net_heat": net_heat,
                "cooling_units": cooling,
                "noise": noise,
                "action_changed": changed,
            }
        )
        self.history.append(info)
        return self._get_obs(), reward, terminated, False, info

    # ------------------------------------------------------------ recompensa

    def _comfort_reward(self, temp: float) -> float:
        """
        Topologias de conforto. A do paper é 'plateau' (eq. 4); as outras existem
        para a ablação — sem elas a tese "recompensa > algoritmo" não é testável.
        """
        cfg = self.config

        if cfg.comfort_type == "step":
            # Faixas discretas (legado v1/v2).
            return {
                ComfortLevel.VERY_COLD: -15.0,
                ComfortLevel.COLD: -5.0,
                ComfortLevel.COMFORTABLE: 1.0,
                ComfortLevel.WARM: -5.0,
                ComfortLevel.VERY_HOT: -15.0,
            }[self.comfort_level()]

        if cfg.comfort_type == "quadratic":
            # Quadrática pura sobre o erro: a recompensa "convencional" da
            # literatura, sem platô nem gradiente.
            return cfg.comfort_bonus - cfg.comfort_sensitivity * (
                temp - cfg.ideal_temp
            ) ** 2

        # --- eq. 4: Platô Quadrático COM GRADIENTE ---
        # O gradiente interno é a correção central do paper: um platô plano não
        # recompensa ENTRAR na faixa (k·0,1² ≈ 0 logo após a borda), o que levava
        # a política a estacionar pouco acima de 26 °C. Com comfort_gradient = 0
        # esta expressão degenera exatamente no platô plano.
        if cfg.temp_comfort_min <= temp <= cfg.temp_comfort_max:
            half_width = (cfg.temp_comfort_max - cfg.temp_comfort_min) / 2.0
            closeness = 1.0 - abs(temp - cfg.ideal_temp) / half_width
            return cfg.comfort_bonus + cfg.comfort_gradient * closeness
        if temp > cfg.temp_comfort_max:
            return cfg.comfort_bonus - cfg.comfort_sensitivity * (
                temp - cfg.temp_comfort_max
            ) ** 2
        return cfg.comfort_bonus - cfg.comfort_sensitivity * (
            cfg.temp_comfort_min - temp
        ) ** 2

    def _reward(self, changed: bool, dwell_before_change: int) -> float:
        """R_t = R_conforto + R_energia + R_mudança + R_frio + R_ciclo."""
        cfg = self.config
        temp = self.current_temp

        comfort = self._comfort_reward(temp)

        # --- R_energia: proporcional à potência ELÉTRICA, escalada no pico ---
        kw = cfg.physics.electrical_kw(self.ac_state)
        mult = cfg.peak_penalty_multiplier if cfg.is_peak_hour(self.hour_of_day) else 1.0
        energy = -kw * cfg.energy_penalty_factor * mult

        # --- R_mudança ---
        change = cfg.action_change_penalty if changed else 0.0

        # --- R_frio: resfriar abaixo do limite inferior ---
        cold = 0.0
        if temp < cfg.temp_comfort_min and self.ac_state in (
            ACState.MEDIUM,
            ACState.HIGH,
        ):
            cold = cfg.cold_action_penalty

        # --- eq. 5: anti-short-cycling ---
        cycle = 0.0
        d_min = cfg.min_dwell_steps
        if changed and dwell_before_change < d_min:
            cycle = cfg.short_cycle_penalty * (d_min - dwell_before_change) / d_min

        return comfort + energy + change + cold + cycle

    # ------------------------------------------------------------- dinâmicas

    def _outdoor_temp(self) -> float:
        """Senoide da hora do dia, pico às 14h."""
        cfg = self.config
        base, amp = (
            (cfg.outdoor_base_temp, cfg.outdoor_amplitude)
            if cfg.season == "summer"
            else (20.0, 6.0)
        )
        phase = (self.hour_of_day - cfg.outdoor_peak_hour) * np.pi / 12.0
        return float(base + amp * np.cos(phase))

    def _scheduled_occupancy(self, hour: int) -> int:
        """Lotação do cenário dentro da janela ocupada, 0 fora dela."""
        if self.scenario_occupancy is None:
            return self.occupancy
        return self.scenario_occupancy if self.config.is_occupied_hour(int(hour) % 24) else 0

    def _advance_occupancy(self) -> None:
        cfg = self.config
        if self.scenario_occupancy is not None:
            # Avaliação: determinística, guiada pela janela ocupada.
            self.occupancy = self._scheduled_occupancy(self.hour_of_day)
        elif self.np_random.random() < 0.1:
            # Treino: passeio aleatório, para a política não memorizar um perfil.
            self.occupancy = int(
                np.clip(
                    self.occupancy + self.np_random.integers(-5, 6),
                    0,
                    cfg.max_occupancy,
                )
            )

    # ------------------------------------------------------------- utilities

    def _tariff(self) -> float:
        cfg = self.config
        m = cfg.peak_tariff_multiplier if cfg.is_peak_hour(self.hour_of_day) else 1.0
        return cfg.energy_tariff_brl_per_kwh * m

    def comfort_level(self) -> ComfortLevel:
        cfg = self.config
        t = self.current_temp
        if t < 18.0:
            return ComfortLevel.VERY_COLD
        if t < cfg.temp_comfort_min:
            return ComfortLevel.COLD
        if t <= cfg.temp_comfort_max:
            return ComfortLevel.COMFORTABLE
        if t < 30.0:
            return ComfortLevel.WARM
        return ComfortLevel.VERY_HOT

    def render(self):
        if self.render_mode == "human" and self.history:
            i = self.history[-1]
            print(
                f"passo {i['step']:3d} | {i['hour']:02d}h | "
                f"{i['temperature']:5.2f}°C | ocup {i['occupancy']:2d} | {i['ac_state']}"
            )

    def close(self):
        pass
