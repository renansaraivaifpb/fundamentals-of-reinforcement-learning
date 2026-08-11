# -*- coding: utf-8 -*-
"""
Controladores de referência.

Motivação (Boutahri & Tilioua 2025, BOPTEST): o baseline do manuscrito é um
termostato de zona morta — um adversário fraco. A literatura de controle predial
compara contra **PI**, que é o que de fato roda em prédio comercial. Ganhar de um
PI honestamente sintonizado é um teste muito mais forte, e é o que um revisor vai
pedir.

Também expõe os limiares do termostato como parâmetros. No manuscrito eles são
implícitos, o que torna a linha "Termostato" da Tabela 4 não reproduzível por
terceiros — foi exatamente a divergência de energia/trocas que encontrei.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ac_physics import ACState
from config import ClassroomConfig


def _temp_from(obs, info: Optional[Dict], ) -> float:
    """Temperatura a partir do info, ou reconstruída de t_norm (eq. 3 invertida)."""
    if info is not None and "temperature" in info:
        return float(info["temperature"])
    return float(np.asarray(obs).reshape(-1)[0]) * 20.0 + 15.0


@dataclass
class ThermostatAgent:
    """
    Termostato com zona morta, agora com limiares EXPLÍCITOS.

    `deadband` cria histerese real: liga ao passar de `temp_comfort_max` e só
    desliga ao cair abaixo de `temp_comfort_max - deadband`. Com deadband = 0 o
    controlador comuta a cada oscilação (bang-bang), o que reproduz as ~2,5
    trocas/h relatadas no manuscrito; com deadband > 0 ele comuta menos e gasta
    menos. É este parâmetro — não publicado — que explica a divergência.
    """

    config: ClassroomConfig
    deadband: float = 0.0
    high_threshold_offset: float = 2.0

    def __post_init__(self) -> None:
        self._on = False

    def reset(self) -> None:
        self._on = False

    def predict(self, obs, deterministic: bool = True, info: Optional[Dict] = None):
        cfg = self.config
        temp = _temp_from(obs, info)
        upper = cfg.temp_comfort_max
        lower = cfg.temp_comfort_max - self.deadband

        if self._on:
            self._on = temp > lower           # histerese: só desliga abaixo de lower
        else:
            self._on = temp > upper

        if not self._on:
            return int(ACState.OFF), None
        if temp > upper + self.high_threshold_offset:
            return int(ACState.HIGH), None
        return int(ACState.MEDIUM), None


@dataclass
class PIController:
    """
    Controlador PI sobre o erro de temperatura, com anti-windup.

    u_t = clip(Kp·e_t + Ki·Σe, 0, 1),  e_t = T_t − setpoint

    A saída contínua é mapeada para o nível discreto mais próximo, de modo que o
    PI opere sob o MESMO espaço de ação do DQN — condição para a comparação ser
    justa. `discrete=False` devolve a carga contínua, para comparar com o SAC.

    Anti-windup por *clamping condicional*: o integrador só acumula quando a
    saída não está saturada. Sem isso, um episódio de pulldown longo satura o
    integrador e o controlador demora a desligar depois — um erro clássico que
    tornaria o baseline artificialmente ruim.
    """

    config: ClassroomConfig
    kp: float = 0.60
    ki: float = 0.08
    setpoint: Optional[float] = None
    discrete: bool = True

    def __post_init__(self) -> None:
        self._integral = 0.0
        if self.setpoint is None:
            self.setpoint = self.config.ideal_temp

    def reset(self) -> None:
        self._integral = 0.0

    def _load(self, temp: float) -> float:
        error = temp - float(self.setpoint)
        candidate = self.kp * error + self.ki * (self._integral + error)
        # Integra apenas se não saturar (anti-windup condicional).
        if 0.0 < candidate < 1.0:
            self._integral += error
        elif candidate <= 0.0 and self._integral > 0.0:
            self._integral = max(0.0, self._integral + error)
        return float(np.clip(self.kp * error + self.ki * self._integral, 0.0, 1.0))

    def predict(self, obs, deterministic: bool = True, info: Optional[Dict] = None):
        load = self._load(_temp_from(obs, info))
        if not self.discrete:
            return np.array([load], dtype=np.float32), None

        fractions = [self.config.physics.load_fraction[s] for s in ACState]
        nearest = int(np.argmin([abs(load - f) for f in fractions]))
        return nearest, None


class AlwaysOffAgent:
    """Contrafactual do filtro de controlabilidade (Seção 4.4)."""

    def reset(self) -> None:
        pass

    def predict(self, obs, deterministic: bool = True, info: Optional[Dict] = None):
        return int(ACState.OFF), None


# ------------------------------------------------------------------ sintonia

def tune_pi(
    env_factory,
    scenarios: List[Dict],
    config: ClassroomConfig,
    kp_grid: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.9, 1.3),
    ki_grid: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.20),
    seed: int = 0,
) -> Dict[str, float]:
    """
    Sintoniza o PI por busca em grade, maximizando conforto na faixa estreita.

    Sintonizar o BASELINE é uma exigência metodológica, não um detalhe: comparar
    um DQN treinado por 550k passos contra um PI com ganhos arbitrários infla a
    vantagem do RL. O grid é pequeno de propósito — o objetivo é um adversário
    honesto, não vencer o DQN.
    """
    from metrics import evaluate_agent

    best = {"kp": kp_grid[0], "ki": ki_grid[0], "score": -np.inf}
    for kp in kp_grid:
        for ki in ki_grid:
            agent = PIController(config, kp=kp, ki=ki)
            res = evaluate_agent(
                agent, env_factory, scenarios, config, seed=seed,
                pass_info_to_agent=True,
            )
            s = res["summary"]
            # Critério: conforto na faixa estreita, penalizado pelo desvio do ideal.
            score = s["comfort_narrow_pct"] - 5.0 * s["abs_dev_from_ideal"]
            if score > best["score"]:
                best = {"kp": kp, "ki": ki, "score": score,
                        "comfort_narrow_pct": s["comfort_narrow_pct"],
                        "comfort_wide_pct": s["comfort_wide_pct"],
                        "abs_dev_from_ideal": s["abs_dev_from_ideal"]}
    return best
