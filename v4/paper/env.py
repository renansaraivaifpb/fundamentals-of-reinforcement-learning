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

        cfg = self.config
        # A física do equipamento herda a flag de aquecimento da config, para
        # que discrete_levels() e thermal_units_signed() fiquem consistentes.
        if cfg.heating_enabled and not cfg.physics.heating_enabled:
            from dataclasses import replace as _replace
            cfg.physics = _replace(cfg.physics, heating_enabled=True)

        self.levels = cfg.physics.discrete_levels()
        if cfg.continuous_action:
            # Carga com sinal: -1 = aquecimento máximo, +1 = resfriamento máximo.
            lo = -1.0 if cfg.heating_enabled else 0.0
            self.action_space = spaces.Box(low=lo, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(self.levels))

        # Observação base (eq. 3) + features opcionais. O tamanho é montado aqui
        # para que o Box declarado corresponda sempre ao que _get_obs emite —
        # foi a divergência entre os dois que gerou o bug de o_norm=1,33 no v4.
        low = [0.0, 0.0, -1.0, -1.0]
        high = [1.0, 1.0, 1.0, 1.0]
        if cfg.observe_scaled_error:
            low += [-1.0]; high += [1.0]          # erro escalado pela tolerância
        if cfg.observe_integral:
            low += [-1.0]; high += [1.0]          # erro acumulado (com fuga)
        if cfg.observe_derivative:
            low += [-1.0]; high += [1.0]          # dT/dt normalizado
        if cfg.observe_time_to_peak:
            low += [0.0, 0.0]; high += [1.0, 1.0] # tempo até ponta, tarifa
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
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
        self.prev_temp: float = cfg.ideal_temp
        self.load: float = 0.0
        self.integral_error: float = 0.0
        self.occupancy_episode = None
        self.sensor_buffer: List[float] = []
        self.measured_temp: float = cfg.ideal_temp
        self.scenario_occupancy: Optional[int] = None
        self.history: List[Dict] = []

    # ------------------------------------------------------------ observação

    def _minutes_to_peak(self) -> float:
        """Minutos até o INÍCIO do próximo posto de ponta; 0 se já está nele."""
        cfg = self.config
        h = self.hour_float % 24.0
        if cfg.tariff.posto(h) == "ponta":
            return 0.0
        starts = [w[0] for w in cfg.tariff.peak_windows]
        if not starts:
            return cfg.peak_lookahead_hours * 60.0
        # Próxima ocorrência, considerando a virada do dia.
        deltas = [((s0 - h) % 24.0) for s0 in starts]
        return min(deltas) * 60.0

    def _read_sensor(self) -> float:
        """
        Temperatura MEDIDA: verdade + ruído de medição + atraso de transporte.
        Distinta de `current_temp` (a verdade), que segue disponível para as
        métricas — avaliar contra a leitura do sensor mediria o sensor, não o
        controle.
        """
        cfg = self.config
        valor = self.current_temp
        if cfg.sensor_noise_std > 0:
            valor += float(self.np_random.normal(0.0, cfg.sensor_noise_std))
        if cfg.sensor_lag_steps > 0:
            self.sensor_buffer.append(valor)
            if len(self.sensor_buffer) > cfg.sensor_lag_steps:
                valor = self.sensor_buffer.pop(0)
            else:
                valor = self.sensor_buffer[0]
        self.measured_temp = float(valor)
        return self.measured_temp

    def _get_obs(self) -> np.ndarray:
        """
        eq. 3 — t_norm = clip((T−15)/20, 0, 1); o_norm = N/N_max — mais as
        features opcionais de derivada e de horizonte tarifário.
        """
        cfg = self.config
        t_medida = self._read_sensor()
        t_norm = np.clip((t_medida - 15.0) / 20.0, 0.0, 1.0)
        o_norm = np.clip(self.occupancy / cfg.max_occupancy, 0.0, 1.0)
        angle = 2.0 * np.pi * self.hour_float / 24.0
        obs = [t_norm, o_norm, np.sin(angle), np.cos(angle)]

        if cfg.observe_scaled_error:
            # A escala é 4x a tolerância: ±tol -> ±0,25, saturando fora de ±4·tol.
            # O canal t_norm acima é mantido de propósito, grosseiro, para que o
            # agente ainda saiba que está a 32 C e não a 26 C quando este saturar.
            escala = max(cfg.error_scale_tolerances * cfg.lab_tolerance, 1e-9)
            obs.append(float(np.clip((t_medida - cfg.ideal_temp) / escala,
                                     -1.0, 1.0)))

        if cfg.observe_integral:
            lim = max(cfg.integral_clip_degree_hours, 1e-9)
            obs.append(float(np.clip(self.integral_error / lim, -1.0, 1.0)))

        if cfg.observe_derivative:
            # Normalizado pela maior variação possível por passo (HIGH a plena
            # carga), para ficar em [-1,1] sem depender da escala da física.
            max_d = (cfg.physics.cooling_units_at_full_load / cfg.thermal_mass) * cfg.dt
            d = (t_medida - self.prev_temp) / max(max_d, 1e-9)
            obs.append(float(np.clip(d, -1.0, 1.0)))

        if cfg.observe_time_to_peak:
            look = max(cfg.peak_lookahead_hours * 60.0, 1e-9)
            obs.append(float(np.clip(self._minutes_to_peak() / look, 0.0, 1.0)))
            rates = [cfg.tariff.off_peak_brl_kwh, cfg.tariff.peak_brl_kwh]
            if cfg.tariff.intermediate_brl_kwh is not None:
                rates.append(cfg.tariff.intermediate_brl_kwh)
            obs.append(float(np.clip(cfg.tariff_rate(self.hour_float) / max(rates), 0.0, 1.0)))

        return np.array(obs, dtype=np.float32)

    def _nearest_ac_state(self, load: float) -> ACState:
        """Nível de refrigeração mais próximo (OFF quando aquecendo)."""
        if load <= 0.0:
            return ACState.OFF
        best, bd = ACState.OFF, 1e9
        for st in ACState:
            d = abs(load - self.config.physics.load_fraction[st])
            if d < bd:
                best, bd = st, d
        return best

    def _get_info(self) -> Dict:
        cfg = self.config
        kw = cfg.physics.electrical_kw_signed(self.load)
        return {
            "temperature": float(self.current_temp),
            "occupancy": int(self.occupancy),
            "ac_state": self.ac_state.name,
            "action": int(self.ac_state),
            "load": float(self.load),
            "mode": ("aquecendo" if self.load < 0 else
                     "resfriando" if self.load > 0 else "desligado"),
            "minutes_to_peak": self._minutes_to_peak(),
            "hour": int(self.hour_of_day),
            "hour_float": float(self.hour_float),
            "step": int(self.time_step),
            "occupied": cfg.is_occupied_hour(self.hour_of_day),
            "electrical_kw": kw,
            "energy_kwh": kw * cfg.dt,
            "cost_brl": kw * cfg.dt * self._tariff(),
            "tariff_brl_kwh": self._tariff(),
            "posto": cfg.tariff.posto(self.hour_float),
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

        if cfg.train_occupancy_mode == "realistic":
            # Mesma família de perturbações no treino e na avaliação: só os
            # parâmetros da agenda randomizam, a estrutura é fixa.
            from occupancy import OccupancyModel
            modelo = OccupancyModel(
                max_occupancy=cfg.max_occupancy,
                tau_minutes=cfg.occupancy_tau_minutes,
                churn_per_hour=cfg.occupancy_churn_per_hour,
            )
            pico = int(self.np_random.integers(5, cfg.max_occupancy + 1))
            self.occupancy_episode = modelo.sample_episode(self.np_random, pico)
            self.occupancy = self.occupancy_episode.reset(float(self.start_hour))
        elif self.occupancy_episode is not None:
            self.occupancy = self.occupancy_episode.step(
                self.hour_float, cfg.dt, self.np_random)
        elif cfg.train_occupancy_mode == "schedule":
            # Domain randomization sobre a AGENDA: lotação de pico e bordas da
            # janela variam, mas a ESTRUTURA (degrau ao encher, degrau ao
            # esvaziar) é a mesma da avaliação. Sem isso o agente nunca treina
            # nas transições que definem todo cenário de teste.
            pico = int(np.clip(
                self.np_random.integers(5, cfg.max_occupancy + 1)
                + self.np_random.integers(-cfg.schedule_occupancy_jitter,
                                          cfg.schedule_occupancy_jitter + 1),
                0, cfg.max_occupancy))
            self.scenario_occupancy = pico
            self.occupancy = self._scheduled_occupancy(self.start_hour)

        if options:
            # Aceita 'hour' (paper/scenarios.json) e 'hour_of_day' (legado v4).
            self.start_hour = int(
                options.get("hour", options.get("hour_of_day", self.start_hour))
            )
            self.current_temp = float(options.get("start_temp", self.current_temp))
            if "occupancy" in options:
                # Cenário fixa a lotação de PICO. Em modo 'realistic' a grade
                # horária a modula (aulas, intervalos, almoço); nos outros, a
                # janela ocupada.
                self.scenario_occupancy = int(options["occupancy"])
                if cfg.train_occupancy_mode == "realistic":
                    from occupancy import OccupancyModel
                    modelo = OccupancyModel(
                        max_occupancy=cfg.max_occupancy,
                        tau_minutes=cfg.occupancy_tau_minutes,
                        churn_per_hour=cfg.occupancy_churn_per_hour,
                    )
                    self.occupancy_episode = modelo.sample_episode(
                        self.np_random, self.scenario_occupancy)
                    self.occupancy = self.occupancy_episode.reset(
                        float(self.start_hour))
                else:
                    self.occupancy = self._scheduled_occupancy(self.start_hour)

        self.hour_float = float(self.start_hour)
        self.hour_of_day = int(self.start_hour) % 24
        self.ac_state = ACState.OFF
        self.prev_temp = self.current_temp
        self.load = 0.0
        self.integral_error = 0.0

        obs, info = self._get_obs(), self._get_info()
        self.history = [info]
        return obs, info

    # ------------------------------------------------------------------ step

    def _decode_action(self, action) -> float:
        """Ação -> carga com sinal em [-1,1] (negativo aquece)."""
        cfg = self.config
        if cfg.continuous_action:
            lo = -1.0 if cfg.heating_enabled else 0.0
            return float(np.clip(np.asarray(action).reshape(-1)[0], lo, 1.0))
        return float(self.levels[int(action)])

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        cfg = self.config
        self.prev_temp = self.current_temp
        previous_load = self.load
        self.load = self._decode_action(action)

        # `ac_state` é mantido para compatibilidade de logs e do termostato: mapeia
        # a carga para o nível de refrigeração mais próximo (0 quando aquecendo).
        self.ac_state = self._nearest_ac_state(self.load)

        if cfg.continuous_action:
            # "Trocou" só quando o ajuste é significativo — comparar floats
            # exatos contaria ruído numérico como comutação de compressor.
            changed = abs(self.load - previous_load) > 0.05
        else:
            changed = self.load != previous_load
        dwell = self.steps_since_change
        self.steps_since_change = 0 if changed else self.steps_since_change + 1

        # --- eq. 2: balanço térmico agregado ---
        outdoor = self._outdoor_temp()
        people_heat = self.occupancy * cfg.heat_gain_per_person
        external_heat = cfg.heat_transfer_coeff * (outdoor - self.current_temp)
        heat_gain = people_heat + external_heat
        cooling = cfg.physics.thermal_units_signed(self.load)
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

        # Integral com fuga: acumula erro em graus·hora e esquece lentamente.
        # A fuga cumpre o papel do anti-windup — sem ela o acumulador divergiria
        # em regimes saturados, que é exatamente o bug que encontrei no meu PI.
        if cfg.observe_integral:
            self.integral_error = (
                cfg.integral_leak * self.integral_error
                + (self.current_temp - cfg.ideal_temp) * cfg.dt
            )
            lim = cfg.integral_clip_degree_hours
            self.integral_error = float(np.clip(self.integral_error, -lim, lim))

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

        if cfg.comfort_type == "huber":
            # Quadratica perto do setpoint (resolucao de precisao), LINEAR longe
            # (escala limitada). Uma quadratica com curvatura alta o bastante para
            # +-0,5 C importar gera recompensa de +10 a -2038 sobre a faixa
            # 10-40 C: faixa dinamica de 204x, que faz os alvos de TD explodirem e
            # o DQN nao converge. A continuacao linear mantem gradiente constante
            # longe do setpoint -- ao contrario de um piso, que zeraria o gradiente
            # e recriaria o problema do plato plano.
            d = abs(temp - cfg.ideal_temp)
            delta = cfg.comfort_huber_delta
            if d <= delta:
                return cfg.comfort_bonus - cfg.comfort_sensitivity * d ** 2
            slope = 2.0 * cfg.comfort_sensitivity * delta   # derivada casada em delta
            return (cfg.comfort_bonus - cfg.comfort_sensitivity * delta ** 2
                    - slope * (d - delta))

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
        kw = cfg.physics.electrical_kw_signed(self.load)
        if cfg.price_aware_penalty:
            # A penalidade segue a TARIFA REAL. Sem isto o agente não tem sinal
            # para antecipar o posto de ponta — ele só vê um multiplicador
            # constante, e antecipação exige saber quando o preço muda.
            energy = -kw * cfg.dt * cfg.tariff_rate(self.hour_float) * cfg.energy_penalty_factor
        else:
            mult = cfg.peak_penalty_multiplier if cfg.is_peak_hour(self.hour_of_day) else 1.0
            energy = -kw * cfg.energy_penalty_factor * mult

        # --- R_mudança ---
        change = cfg.action_change_penalty if changed else 0.0

        # --- R_frio: resfriar abaixo do limite inferior ---
        # Com aquecimento disponível a penalidade de frio perde sentido: resfriar
        # no frio deixa de ser o único erro possível, e o próprio conforto já
        # penaliza o desvio nos dois sentidos.
        cold = 0.0
        if not cfg.heating_enabled and temp < cfg.temp_comfort_min and self.load > 0.2:
            cold = cfg.cold_action_penalty

        # --- Custo de variação (termo derivativo do custo de controle) ---
        # Penaliza a TAXA de variação da temperatura, amortecendo oscilação de
        # forma direta. Diferente da penalidade de troca de ação, que conta
        # comutações; esta penaliza o efeito, não o comando.
        rate = 0.0
        if cfg.derivative_penalty > 0.0:
            dtemp = (self.current_temp - self.prev_temp) / cfg.dt   # °C/h
            # PORTÃO POR TOLERÂNCIA. A versão anterior penalizava variação em
            # qualquer temperatura — inclusive durante o pulldown, quando variar
            # rápido é exatamente o desejado. Isso contradizia o objetivo: o
            # termo existe para amortecer oscilação PERTO do alvo, não para
            # frear a aproximação.
            #
            # O portão vale 1 dentro da tolerância e decai linearmente a 0 até
            # 2x a tolerância. Rampa em vez de degrau para não introduzir
            # descontinuidade na recompensa na borda da faixa.
            erro = abs(temp - cfg.ideal_temp)
            tol = max(cfg.lab_tolerance, 1e-9)
            portao = float(np.clip(2.0 - erro / tol, 0.0, 1.0))
            rate = -cfg.derivative_penalty * portao * dtemp ** 2

        # --- eq. 5: anti-short-cycling ---
        cycle = 0.0
        d_min = cfg.min_dwell_steps
        if changed and dwell_before_change < d_min:
            cycle = cfg.short_cycle_penalty * (d_min - dwell_before_change) / d_min

        return comfort + energy + change + cold + cycle + rate

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
        if self.occupancy_episode is not None:
            self.occupancy = self.occupancy_episode.step(
                self.hour_float, cfg.dt, self.np_random)
        elif self.scenario_occupancy is not None:
            # Avaliação legada: determinística, guiada pela janela ocupada.
            self.occupancy = self._scheduled_occupancy(self.hour_of_day)
        elif self.occupancy_episode is not None:
            self.occupancy = self.occupancy_episode.step(
                self.hour_float, cfg.dt, self.np_random)
        elif cfg.train_occupancy_mode == "schedule":
            # Mesma estrutura da avaliação: janela ocupada com lotação de pico.
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
        """
        R$/kWh vigente. Usa `hour_float`, não `hour_of_day`: a Tarifa Branca tem
        fronteiras em meia hora (17h30), que a hora inteira não representa.
        """
        return self.config.tariff_rate(self.hour_float)

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
