# -*- coding: utf-8 -*-
"""
Desempenho contra ORÇAMENTO DE TREINO — inspirado nas Figs. 9-10 de Yuan et al.

PERGUNTA QUE ISTO RESPONDE. Todos os resultados anteriores usam orçamento fixo,
o que deixa aberta a objecao mais obvia de um revisor: "o DQN perde porque foi
mal treinado?". Yuan et al. plotam custo e desconforto ano a ano (Y1..Y7) e
mostram que o RL leva anos para superar as referencias. O analogo aqui e avaliar
periodicamente durante o treino e traçar a curva contra a linha do PI.

Se a curva SATURA abaixo do PI, o argumento deixa de ser "treinou pouco" e passa
a ser "o metodo converge para pior". Se ainda sobe ao fim do orcamento, a
conclusao honesta e que faltou treino — e isso precisa ser dito.

    python experimentos/curva_aprendizado.py
"""
from __future__ import annotations

import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from hvac.baselines import PIController
from hvac.config import config_for_profile
from hvac.env import ClassroomACEnv
from hvac.metrics import evaluate_agent
from hvac.scenarios import build_scenario_matrix
from hvac.wrappers import ActionRepeatWrapper

TOLERANCIAS = (2.0, 0.5)
SEEDS = (0, 1, 2)
PASSOS = 400_000
INTERVALO = 25_000
SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "resultados_curva_aprendizado.csv")


def cfg_para(tol):
    return config_for_profile("Equilibrado", temp_comfort_min=24 - tol,
                              temp_comfort_max=24 + tol, lab_tolerance=tol)


def fab(cfg):
    return lambda: ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)


def mede(agente, cfg, info=False):
    r = evaluate_agent(agente, fab(cfg), build_scenario_matrix(), cfg, seed=0,
                       pass_info_to_agent=info)["summary"]
    return r["in_tolerance_pct"], r["energy_kwh_day"]


class Curva(BaseCallback):
    """Avalia na matriz de cenarios a cada `INTERVALO` passos."""

    def __init__(self, cfg, tol, seed, acc):
        super().__init__()
        self.cfg, self.tol, self.seed, self.acc = cfg, tol, seed, acc
        self.proximo = INTERVALO

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.proximo:
            self.proximo += INTERVALO
            tol_pct, kwh = mede(self.model, self.cfg)
            self.acc.append({"tolerancia": self.tol, "seed": self.seed,
                             "passos": int(self.num_timesteps),
                             "na_tolerancia_pct": tol_pct, "energia_kwh": kwh})
        return True


def main():
    linhas = []
    for tol in TOLERANCIAS:
        cfg = cfg_para(tol)
        kp, ki = (1.8, 0.4) if tol >= 1.0 else (1.3, 0.0)   # sintonia do experimento
        pi_tol, pi_kwh = mede(PIController(cfg, kp=kp, ki=ki), cfg, info=True)
        print(f"\n±{tol} °C | PI (Kp={kp}, Ki={ki}): {pi_tol:.1f}%", flush=True)
        linhas.append({"tolerancia": tol, "seed": -1, "passos": 0,
                       "na_tolerancia_pct": pi_tol, "energia_kwh": pi_kwh,
                       "controlador": "PI sintonizado"})

        for seed in SEEDS:
            t0 = time.time()
            acc = []
            env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
            env.reset(seed=seed)
            m = DQN("MlpPolicy", env, learning_rate=5e-5, batch_size=64,
                    verbose=0, seed=seed)
            m.learn(total_timesteps=PASSOS, callback=Curva(cfg, tol, seed, acc))
            for r in acc:
                r["controlador"] = "DQN"
            linhas.extend(acc)
            print(f"  seed {seed} ({(time.time()-t0)/60:.1f} min): "
                  f"{acc[-1]['na_tolerancia_pct']:.1f}% ao final", flush=True)
            pd.DataFrame(linhas).to_csv(SAIDA, index=False)
    print(f"\ngravado: {SAIDA}")


if __name__ == "__main__":
    main()
