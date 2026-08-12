# -*- coding: utf-8 -*-
"""
Por que o DQN degrada quando a faixa aperta? Hipótese: falta o integrador.

MECANISMO PROPOSTO. PI e DQN operam sobre o MESMO espaço de ação (4 níveis
discretos), de modo que quantização não explica a diferença. O que o PI tem e o
DQN do manuscrito não tem é ESTADO SUFICIENTE para rastreamento: o integrador
acumula o erro e elimina offset persistente. A observação do manuscrito é
(temperatura, ocupação, hora) — puramente reativa.

Com faixa larga o offset cabe na faixa e não custa nada. Com ±0,5 °C ele passa a
ser a diferença entre estar dentro e fora. Se a hipótese estiver certa, dar ao
agente o erro integral (e o erro escalado pela tolerância) deve recuperar parte
da distância.
"""
from __future__ import annotations

import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd
from dataclasses import replace
from stable_baselines3 import DQN

from hvac.config import config_for_profile
from hvac.env import ClassroomACEnv
from hvac.metrics import evaluate_agent
from hvac.scenarios import build_scenario_matrix
from hvac.wrappers import ActionRepeatWrapper

TOL, PASSOS, SEEDS = 0.5, 300_000, (0, 1, 2)
SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "resultados_teste_integral.csv")

base = config_for_profile("Equilibrado", temp_comfort_min=24 - TOL,
                          temp_comfort_max=24 + TOL, lab_tolerance=TOL)
VARIANTES = {
    "obs do manuscrito (T, ocup., hora)": base,
    "+ erro escalado e integral": replace(base, observe_scaled_error=True,
                                          observe_integral=True),
}

linhas = []
for nome, cfg in VARIANTES.items():
    n_canais = ClassroomACEnv(config=cfg).observation_space.shape[0]
    print(f"\n{nome}  ({n_canais} canais)", flush=True)
    for seed in SEEDS:
        t0 = time.time()
        env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
        env.reset(seed=seed)
        m = DQN("MlpPolicy", env, learning_rate=5e-5, batch_size=64,
                verbose=0, seed=seed)
        m.learn(total_timesteps=PASSOS)
        s = evaluate_agent(m, lambda: ActionRepeatWrapper(
            ClassroomACEnv(config=cfg), repeat=2),
            build_scenario_matrix(), cfg, seed=0)["summary"]
        print(f"  seed {seed} ({(time.time()-t0)/60:.1f} min): "
              f"{s['in_tolerance_pct']:.1f}% na tolerância", flush=True)
        linhas.append({"variante": nome, "n_canais": n_canais, "seed": seed,
                       "na_tolerancia_pct": s["in_tolerance_pct"],
                       "desvio_ideal": s["abs_dev_from_ideal"],
                       "energia_kwh": s["energy_kwh_day"]})
        pd.DataFrame(linhas).to_csv(SAIDA, index=False)
print(f"\ngravado: {SAIDA}")
