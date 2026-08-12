# -*- coding: utf-8 -*-
"""
A faixa estreita muda o vencedor? — experimento controlado.

PERGUNTA. O PI iguala o DQN quando o alvo é uma faixa larga de 4 °C. Se a
especificação exigir precisão — ±1,0 °C, ±0,5 °C —, o aprendizado passa a
compensar?

DESENHO. Para cada largura de tolerância, DOIS controladores são preparados
para AQUELA largura, e não reaproveitados de outra:

  * DQN treinado com a faixa de conforto correspondente (a recompensa muda com
    a especificação — treinar para ±2 °C e cobrar ±0,5 °C não testaria a
    hipótese, mediria descasamento de objetivo);
  * PI re-sintonizado por busca em grade para a mesma largura.

Re-sintonizar o PI a cada largura é a parte não negociável. Congelar os ganhos
de 4 °C e cobrar precisão de 0,5 °C reproduziria, dentro deste experimento,
exatamente o defeito que o trabalho acusa no manuscrito auditado: vencer um
adversário mal configurado.

Orçamento reduzido em relação ao protocolo do artigo (300k passos em vez de
550k) para caber em execução interativa. Está registrado no CSV, e a comparação
permanece justa porque o orçamento é o mesmo em todas as larguras.

    python experimentos/faixa_estreita.py
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import DQN

from hvac.baselines import PIController, ThermostatAgent
from hvac.config import config_for_profile
from hvac.env import ClassroomACEnv
from hvac.metrics import evaluate_agent
from hvac.scenarios import build_scenario_matrix
from hvac.wrappers import ActionRepeatWrapper

TOLERANCIAS = (2.0, 1.0, 0.5)
SEEDS = (0, 1, 2)
PASSOS = 300_000
KP_GRID = (0.3, 0.6, 1.0, 1.3, 1.8, 2.5, 3.5)
KI_GRID = (0.0, 0.05, 0.1, 0.2, 0.4)

SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "resultados_faixa_estreita.csv")


def cfg_para(tol: float):
    """Config cuja faixa de conforto — e cuja métrica — têm meia-largura `tol`."""
    return config_for_profile(
        "Equilibrado",
        temp_comfort_min=24.0 - tol, temp_comfort_max=24.0 + tol,
        lab_tolerance=tol,
    )


def fabrica(cfg, repeat: int = 2):
    return lambda: ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=repeat)


def avalia(agente, cfg, info: bool = False) -> dict:
    r = evaluate_agent(agente, fabrica(cfg), build_scenario_matrix(), cfg,
                       seed=0, pass_info_to_agent=info)["summary"]
    return {"na_tolerancia_pct": r["in_tolerance_pct"],
            "desvio_ideal": r["abs_dev_from_ideal"],
            "sigma": r["temp_std"],
            "energia_kwh": r["energy_kwh_day"],
            "custo_brl": r["cost_brl_day"],
            "trocas_h": r["changes_per_hour"]}


def sintoniza_pi(cfg) -> tuple:
    """Busca em grade maximizando o tempo dentro da tolerância VIGENTE."""
    melhor, score = (1.3, 0.2), -np.inf
    for kp in KP_GRID:
        for ki in KI_GRID:
            s = avalia(PIController(cfg, kp=kp, ki=ki), cfg, info=True)
            # Desempate pelo desvio: várias combinações saturam em 100 %.
            v = (s["na_tolerancia_pct"], -s["desvio_ideal"])
            if v > (score if isinstance(score, tuple) else (score, 0)):
                melhor, score = (kp, ki), v
    return melhor, score


def main() -> None:
    linhas = []
    for tol in TOLERANCIAS:
        cfg = cfg_para(tol)
        print(f"\n{'='*66}\nTolerância ±{tol} °C  (faixa "
              f"[{24-tol:.1f}, {24+tol:.1f}])\n{'='*66}", flush=True)

        (kp, ki), _ = sintoniza_pi(cfg)
        s = avalia(PIController(cfg, kp=kp, ki=ki), cfg, info=True)
        print(f"  PI sintonizado Kp={kp} Ki={ki}: "
              f"{s['na_tolerancia_pct']:.1f}% na tolerância", flush=True)
        linhas.append({"tolerancia": tol, "controlador": "PI sintonizado",
                       "seed": -1, "kp": kp, "ki": ki, "passos": 0, **s})

        s = avalia(ThermostatAgent(cfg, deadband=1.0), cfg, info=True)
        linhas.append({"tolerancia": tol, "controlador": "Termostato (zm=1 °C)",
                       "seed": -1, "kp": np.nan, "ki": np.nan, "passos": 0, **s})

        for seed in SEEDS:
            t0 = time.time()
            env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
            env.reset(seed=seed)
            model = DQN("MlpPolicy", env, learning_rate=5e-5, batch_size=64,
                        verbose=0, seed=seed)
            model.learn(total_timesteps=PASSOS)
            s = avalia(model, cfg)
            print(f"  DQN seed {seed} ({(time.time()-t0)/60:.1f} min): "
                  f"{s['na_tolerancia_pct']:.1f}% na tolerância", flush=True)
            linhas.append({"tolerancia": tol, "controlador": "DQN", "seed": seed,
                           "kp": np.nan, "ki": np.nan, "passos": PASSOS, **s})

        pd.DataFrame(linhas).to_csv(SAIDA, index=False)   # salva incremental

    print(f"\ngravado: {SAIDA}")


if __name__ == "__main__":
    main()
