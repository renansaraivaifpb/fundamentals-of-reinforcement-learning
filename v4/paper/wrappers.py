# -*- coding: utf-8 -*-
"""
Wrappers: horizonte de decisão, escudo de segurança e trilha de auditoria
(Seções 4.4 e 5.4 do paper).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ac_physics import ACState


class ActionRepeatWrapper(gym.Wrapper):
    """
    Mantém a ação por `repeat` passos (o paper usa action repeat = 2, i.e.
    12 min por decisão).

    Atenção: este wrapper faz parte do CONTRATO do modelo. Avaliar a política
    sem ele — como o `analyzer_v4.1.py` original fazia — muda o horizonte de
    decisão e mede uma política diferente da que foi treinada.
    """

    def __init__(self, env: gym.Env, repeat: int = 2):
        super().__init__(env)
        if repeat < 1:
            raise ValueError("repeat deve ser >= 1")
        self.repeat = repeat

    def step(self, action):
        total = 0.0
        terminated = truncated = False
        obs, info = None, {}
        # Acumuladores: grandezas EXTENSIVAS precisam somar sobre os passos
        # internos, e `action_changed` precisa ser um OR. Devolver apenas o
        # info do último passo interno descartaria metade da energia e TODAS as
        # trocas de nível (a troca ocorre no primeiro passo interno, e o
        # segundo já reporta changed=False).
        energy_kwh = 0.0
        cost_brl = 0.0
        changed = False
        inner = 0

        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += reward
            energy_kwh += float(info.get("energy_kwh", 0.0))
            cost_brl += float(info.get("cost_brl", 0.0))
            changed = changed or bool(info.get("action_changed", False))
            inner += 1
            if terminated or truncated:
                break

        info = dict(info)
        info.update(
            {
                "energy_kwh": energy_kwh,
                "cost_brl": cost_brl,
                "action_changed": changed,
                "inner_steps": inner,
            }
        )
        return obs, total, terminated, truncated, info


class SafetyShieldWrapper(gym.Wrapper):
    """
    Escudo de segurança: sobrescreve a ação nos limites de temperatura,
    garantindo segurança térmica INDEPENDENTEMENTE da política e mitigando
    reward hacking.

    Acima de `shield_force_cool_above` força resfriamento; abaixo de
    `shield_force_off_below` desliga. Registra cada intervenção — sem esse
    contador não há como saber se o ganho medido vem da política ou do escudo.
    """

    def __init__(self, env: gym.Env, enabled: bool = True):
        super().__init__(env)
        self.enabled = enabled
        self.interventions: int = 0

    def reset(self, **kwargs):
        self.interventions = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.enabled:
            cfg = self.env.unwrapped.config
            temp = self.env.unwrapped.current_temp
            forced: Optional[int] = None
            if temp > cfg.shield_force_cool_above and int(action) == ACState.OFF:
                forced = int(ACState.HIGH)
            elif temp < cfg.shield_force_off_below and int(action) != ACState.OFF:
                forced = int(ACState.OFF)
            if forced is not None:
                self.interventions += 1
                action = forced

        obs, reward, terminated, truncated, info = self.env.step(action)
        info["shield_interventions"] = self.interventions
        return obs, reward, terminated, truncated, info


class MinDwellWrapper(gym.Wrapper):
    """
    Impõe a permanência mínima d_min como RESTRIÇÃO DURA, não como penalidade.

    Motivação empírica: a penalidade da eq. 5 não funciona. Medindo a
    distribuição de permanência dos agentes treinados, a mediana é 12 min (uma
    única decisão) e 71–83 % das comutações violam o d_min de 36 min. O motivo é
    aritmético — a penalidade vale no máximo |rho| = 5, contra ganhos de conforto
    de até B + B_c = 14. Um termo de recompensa que pode ser superado por outro
    termo não é uma proteção; é uma sugestão.

    A correção segue a lógica de shielding de Xu et al. (2025): garantir a
    propriedade estruturalmente, independentemente da política. Enquanto o
    compressor não cumprir d_min, a ação é congelada no nível vigente.

    Trade-off explícito: isto reduz a autoridade de controle e deve custar algum
    conforto. Esse custo é o resultado honesto a reportar — não um defeito.
    """

    def __init__(self, env: gym.Env, min_dwell_steps: Optional[int] = None,
                 enabled: bool = True):
        super().__init__(env)
        self.enabled = enabled
        base = env.unwrapped
        self.min_dwell_steps = (
            min_dwell_steps if min_dwell_steps is not None
            else base.config.min_dwell_steps
        )
        self.blocked: int = 0

    def reset(self, **kwargs):
        self.blocked = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        base = self.env.unwrapped
        if self.enabled:
            current = int(base.ac_state)
            # `steps_since_change` é contado em passos do ambiente, mesma unidade
            # de min_dwell_steps — não misturar com o horizonte de decisão.
            if int(action) != current and base.steps_since_change < self.min_dwell_steps:
                self.blocked += 1
                action = current

        obs, reward, terminated, truncated, info = self.env.step(action)
        info["dwell_blocked"] = self.blocked
        return obs, reward, terminated, truncated, info


class CompressorProtectionWrapper(gym.Wrapper):
    """
    Camada de proteção do compressor, imposta como RESTRIÇÃO DURA.

    Endereça três modos de desgaste distintos, medidos em `compressor_health.py`:

    1. `min_run_minutes` — tempo mínimo ligado antes de poder desligar. Partir e
       desligar em poucos minutos não permite o retorno de óleo ao cárter. O DQN
       discreto operava com mediana de 12 min.
    2. `min_mode_minutes` — tempo mínimo num sentido antes de REVERTER o ciclo.
       Inverter a válvula de 4 vias exige equalização de pressão; em unidade real
       o compressor para durante a reversão. Este modo de desgaste só passou a
       existir quando habilitei o aquecimento (6–14 reversões/dia medidas).
    3. `max_ramp_per_min` — limite de rampa da carga, protegendo a eletrônica de
       potência.

    Por que restrição e não penalidade: já está medido nesta base que uma
    penalidade de recompensa limitada a |rho|=5 é superada por um conforto de 14
    em 71–83 % das comutações. Integridade de hardware é requisito, não item
    negociável na função objetivo.
    """

    def __init__(self, env: gym.Env, min_run_minutes: float = 5.0,
                 min_mode_minutes: float = 10.0,
                 max_ramp_per_min: Optional[float] = 0.35,
                 enabled: bool = True):
        super().__init__(env)
        self.enabled = enabled
        base = env.unwrapped
        self.minutes_per_step = base.config.dt * 60.0
        self.min_run_steps = int(round(min_run_minutes / self.minutes_per_step))
        self.min_mode_steps = int(round(min_mode_minutes / self.minutes_per_step))
        self.max_ramp = max_ramp_per_min
        self.blocked_off = 0
        self.blocked_reversal = 0
        self.clipped_ramp = 0

    def reset(self, **kwargs):
        self.blocked_off = 0
        self.blocked_reversal = 0
        self.clipped_ramp = 0
        self._run_steps = 0
        self._mode_steps = 10**6      # começa "estabilizado"
        self._mode = 0
        return self.env.reset(**kwargs)

    def _sign(self, load: float) -> int:
        return 0 if abs(load) < 1e-6 else (1 if load > 0 else -1)

    def step(self, action):
        base = self.env.unwrapped
        cfg = base.config
        continuo = cfg.continuous_action

        if self.enabled:
            atual = base.load
            pedido = base._decode_action(action)

            # (3) Limite de rampa.
            if self.max_ramp is not None:
                limite = self.max_ramp * self.minutes_per_step
                if abs(pedido - atual) > limite:
                    pedido = atual + np.sign(pedido - atual) * limite
                    self.clipped_ramp += 1

            s_atual, s_pedido = self._sign(atual), self._sign(pedido)

            # (1) Tempo mínimo ligado antes de desligar.
            if s_atual != 0 and s_pedido == 0 and self._run_steps < self.min_run_steps:
                pedido = atual
                self.blocked_off += 1
                s_pedido = s_atual

            # (2) Tempo mínimo no sentido antes de reverter.
            if (s_atual != 0 and s_pedido != 0 and s_pedido != s_atual
                    and self._mode_steps < self.min_mode_steps):
                # Bloqueia a reversão levando a carga a zero em vez de inverter:
                # desligar é benigno, reverter cedo não é.
                pedido = 0.0
                self.blocked_reversal += 1

            action = np.array([pedido], dtype=np.float32) if continuo else \
                int(np.argmin([abs(pedido - L) for L in base.levels]))

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Contadores de estado após o passo.
        s = self._sign(base.load)
        self._run_steps = self._run_steps + 1 if s != 0 else 0
        self._mode_steps = 0 if (s != 0 and s != self._mode) else self._mode_steps + 1
        if s != 0:
            self._mode = s

        info.update({"blocked_off": self.blocked_off,
                     "blocked_reversal": self.blocked_reversal,
                     "clipped_ramp": self.clipped_ramp})
        return obs, reward, terminated, truncated, info


class DecisionLoggerWrapper(gym.Wrapper):
    """
    Registrador de decisões: trilha de auditoria com estado, ação, recompensa
    e um motivo legível por passo — endereça observabilidade e auditabilidade.
    """

    def __init__(self, env: gym.Env, path: Optional[str] = None):
        super().__init__(env)
        self.path = path
        self.records: List[Dict] = []

    def reset(self, **kwargs):
        self.records = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.records.append(
            {
                "step": info.get("step"),
                "hour": info.get("hour"),
                "temperature": round(float(info.get("temperature", 0.0)), 3),
                "occupancy": info.get("occupancy"),
                "action": info.get("ac_state"),
                "reward": round(float(reward), 4),
                "electrical_kw": round(float(info.get("electrical_kw", 0.0)), 4),
                "reason": self._explain(info),
            }
        )
        if (terminated or truncated) and self.path:
            self.flush()
        return obs, reward, terminated, truncated, info

    def _explain(self, info: Dict) -> str:
        cfg = self.env.unwrapped.config
        t = float(info.get("temperature", 0.0))
        parts = []
        if t > cfg.temp_comfort_max:
            parts.append(f"acima do teto ({cfg.temp_comfort_max}°C): resfriar")
        elif t < cfg.temp_comfort_min:
            parts.append(f"abaixo do piso ({cfg.temp_comfort_min}°C): não resfriar")
        else:
            parts.append("dentro da faixa de conforto")
        if info.get("action_changed"):
            parts.append("trocou de nível")
        if not info.get("occupied", True):
            parts.append("fora da janela ocupada")
        if info.get("shield_interventions"):
            parts.append("escudo ativo no episódio")
        return "; ".join(parts)

    def flush(self) -> None:
        if not self.path:
            return
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.records, fh, ensure_ascii=False, indent=2)


class ContinuousActionWrapper(gym.Wrapper):
    """
    Expõe a ação como fração contínua de potência a ∈ [0,1] para o SAC
    (Seção 4.5).

    O ambiente base é discreto, então o wrapper reescreve a física e a
    recompensa de energia usando o modelo contínuo de COP — assim DQN e SAC
    são avaliados sob o MESMO modelo de eficiência, condição para que a
    Tabela 5 seja uma comparação justa.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._last_load = 0.0

    def reset(self, **kwargs):
        self._last_load = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        load = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        base = self.env.unwrapped
        cfg = base.config

        # Mapeia a carga contínua para o nível discreto mais próximo apenas
        # para reaproveitar o step do ambiente; em seguida corrige o efeito de
        # resfriamento e a energia para os valores contínuos reais.
        fractions = [cfg.physics.load_fraction[s] for s in ACState]
        nearest = int(np.argmin([abs(load - f) for f in fractions]))

        prev_temp = base.current_temp
        obs, reward, terminated, truncated, info = self.env.step(nearest)

        # Refaz o passo térmico com a potência contínua exata.
        cooling_exact = cfg.physics.cooling_units_continuous(load)
        net_exact = info["people_heat"] + info["external_heat"] - cooling_exact
        base.current_temp = float(
            np.clip(
                prev_temp + (cfg.dt / cfg.thermal_mass) * net_exact + info["noise"],
                cfg.temp_min_clip,
                cfg.temp_max_clip,
            )
        )

        kw_exact = cfg.physics.electrical_kw_continuous(load)
        significant_change = abs(load - self._last_load) > 0.05
        self._last_load = load

        info.update(
            {
                "temperature": base.current_temp,
                "electrical_kw": kw_exact,
                "energy_kwh": kw_exact * cfg.dt,
                "cost_brl": kw_exact * cfg.dt * base._tariff(),
                "cooling_units": cooling_exact,
                "load": load,
                "action_changed": significant_change,
            }
        )
        if base.history:
            base.history[-1] = info
        return base._get_obs(), reward, terminated, truncated, info
