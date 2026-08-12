# -*- coding: utf-8 -*-
"""
Matriz 3×3 de avaliação (Tabela 3) e o baseline termostático (Seção 4.4).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .physics import ACState
from .config import ClassroomConfig

# Tabela 3: condição inicial (temperatura, hora de início) × ocupação.
INITIAL_CONDITIONS: List[Tuple[str, float, int]] = [
    ("Frio",   17.0,  7),
    ("Normal", 24.0, 10),
    ("Quente", 30.0, 14),
]
OCCUPANCY_LEVELS: List[Tuple[str, int]] = [
    ("Poucas", 5),
    ("Médias", 22),
    ("Muitas", 45),
]


# --- Matriz para o regime de DEMANDA CONTRATADA ---------------------------
# Todos partem DO SETPOINT, de propósito. Medição que motivou: com partidas de
# 17 °C ou 30 °C, 6 dos 9 cenários estouravam a demanda no PASSO 0 — provocados
# pela condição inicial, onde antecipação é IMPOSSÍVEL (não há como pré-resfriar
# antes de t=0). Isso testaria uma hipótese que não pode pagar.
#
# Partindo do setpoint, a pressão passa a vir da OCUPAÇÃO — evento intra-episódio,
# no passo ~36, e previsível pela grade horária. É a única configuração em que
# antecipar é possível, e portanto a única em que uma política aprendida pode
# superar realimentação reativa.
DEMAND_START_HOURS = (0, 4, 20)          # antes do enchimento da manhã
DEMAND_OCCUPANCY = (15, 30, 45)


def build_demand_scenario_matrix() -> List[Dict]:
    """Matriz 3×3 partindo do setpoint, para o regime de demanda contratada."""
    out, idx = [], 0
    for hora in DEMAND_START_HOURS:
        for occ in DEMAND_OCCUPANCY:
            idx += 1
            rot = "Poucas" if occ <= 15 else "Médias" if occ <= 32 else "Muitas"
            out.append({
                "id": f"D{idx}", "name": f"D{idx}: início {hora}h + {rot}",
                "condition": "Setpoint", "occupancy_label": rot,
                "start_temp": 24.0, "hour": hora, "occupancy": occ,
            })
    return out


def build_scenario_matrix() -> List[Dict]:
    """Gera C1..C9 na ordem da Tabela 3 (linha = temperatura, coluna = ocupação)."""
    scenarios = []
    idx = 0
    for cond_name, temp, hour in INITIAL_CONDITIONS:
        for occ_name, occ in OCCUPANCY_LEVELS:
            idx += 1
            scenarios.append(
                {
                    "id": f"C{idx}",
                    "name": f"C{idx}: {cond_name} + {occ_name}",
                    "condition": cond_name,
                    "occupancy_label": occ_name,
                    "start_temp": temp,
                    "hour": hour,
                    "occupancy": occ,
                }
            )
    return scenarios


# --- Separação calibração / teste (Revisor 2) -----------------------------
#
#   "Caso os pesos tenham sido selecionados a partir dos resultados nessa
#    matriz, ela não constitui um conjunto independente de avaliação. Recomendo
#    separar os cenários empregados na calibração daqueles usados no teste ou
#    realizar uma avaliação em um conjunto mais amplo de condições geradas
#    aleatoriamente."
#
# A matriz 3×3 é pequena demais para dividir sem perder cobertura: cada célula é
# uma combinação distinta de temperatura × ocupação, e remover 4 delas eliminaria
# regimes inteiros. A solução correta é gerar cenários aleatórios para TESTE,
# mantendo a matriz como conjunto de CALIBRAÇÃO — assim o teste é independente e
# muito mais amplo.

# Célula da matriz reservada para calibração de hiperparâmetros/pesos.
CALIBRATION_IDS = ("C2", "C4", "C6", "C8")


def split_scenario_matrix() -> Tuple[List[Dict], List[Dict]]:
    """
    Divide a matriz 3×3 em calibração e teste (holdout).

    A divisão em xadrez preserva diversidade nos dois lados: cada subconjunto
    contém pelo menos uma condição fria, normal e quente. Uma divisão por linha
    (ex.: treinar em "frio", testar em "quente") mediria extrapolação, não
    generalização, e não é o que o revisor pede.
    """
    todos = build_scenario_matrix()
    calib = [s for s in todos if s["id"] in CALIBRATION_IDS]
    teste = [s for s in todos if s["id"] not in CALIBRATION_IDS]
    return calib, teste


def sample_random_scenarios(
    n: int = 100,
    seed: int = 12345,
    temp_range: Tuple[float, float] = (16.0, 33.0),
    occupancy_range: Tuple[int, int] = (0, 45),
    hour_range: Tuple[int, int] = (0, 24),
) -> List[Dict]:
    """
    Conjunto de teste amplo e independente, gerado aleatoriamente.

    Usa uma semente FIXA e distinta das sementes de treino, para que o conjunto
    de teste seja idêntico entre agentes (comparação emparelhada) e não coincida
    com nenhuma trajetória vista no treino.

    Cobre o espaço contínuo em vez de 9 pontos — é a evidência mais forte que o
    revisor pede: "Uma avaliação com episódios gerados aleatoriamente, diferentes
    perfis de ocupação e séries meteorológicas reais forneceria evidências mais
    fortes."
    """
    rng = np.random.default_rng(seed)
    cenarios: List[Dict] = []
    for i in range(n):
        temp = float(rng.uniform(*temp_range))
        occ = int(rng.integers(occupancy_range[0], occupancy_range[1] + 1))
        hour = int(rng.integers(*hour_range))
        cenarios.append(
            {
                "id": f"R{i+1:03d}",
                "name": f"R{i+1:03d}: {temp:.1f}°C, {occ}p, {hour}h",
                "condition": (
                    "Frio" if temp < 21.0 else "Normal" if temp < 27.0 else "Quente"
                ),
                "occupancy_label": (
                    "Poucas" if occ <= 15 else "Médias" if occ <= 32 else "Muitas"
                ),
                "start_temp": temp,
                "hour": hour,
                "occupancy": occ,
            }
        )
    return cenarios


class ThermostatAgent:
    """
    Termostato com zona morta (deadband) — o baseline do paper.

    Liga em MEDIUM ao passar do teto e vai a HIGH se exceder por mais de 2 °C;
    desliga ao voltar para dentro da faixa. É este comportamento que produz o
    "estacionar no teto" (|T−24| = 2,21 °C, 41,4 % de sobreaquecimento) que os
    perfis de RL superam.
    """

    def __init__(self, config: ClassroomConfig):
        self.config = config

    def predict(self, obs, deterministic: bool = True, info: Dict | None = None):
        cfg = self.config
        if info is not None and "temperature" in info:
            temp = float(info["temperature"])
        else:
            # Reconstrói a temperatura a partir de t_norm (eq. 3 invertida).
            temp = float(np.asarray(obs).reshape(-1)[0]) * 20.0 + 15.0

        if temp > cfg.temp_comfort_max + 2.0:
            action = int(ACState.HIGH)
        elif temp > cfg.temp_comfort_max:
            action = int(ACState.MEDIUM)
        else:
            action = int(ACState.OFF)
        return action, None


class AlwaysOffAgent:
    """
    Contrafactual usado pelo filtro de controlabilidade (Seção 4.4): um cenário
    só entra na média de conforto se, com o AC sempre desligado, a sala
    ultrapassaria 26 °C — isto é, se resfriar era de fato necessário.
    """

    def predict(self, obs, deterministic: bool = True, info: Dict | None = None):
        return int(ACState.OFF), None
