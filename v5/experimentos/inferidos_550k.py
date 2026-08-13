# -*- coding: utf-8 -*-
"""
Sementes 1 e 2 sob os parâmetros INFERIDOS a 550k passos.

Sem elas, a comparação com a configuração derivada opõe três sementes a uma, e a
diferença observada poderia ser variabilidade de semente e não efeito dos
parâmetros.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import pandas as pd
from stable_baselines3 import DQN
from hvac.config import config_for_profile
from hvac.env import ClassroomACEnv
from hvac.metrics import evaluate_agent
from hvac.scenarios import build_scenario_matrix
from hvac.wrappers import ActionRepeatWrapper

SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "resultados_inferidos_550k.csv")
linhas = []
for perfil in ("Agressivo", "Equilibrado", "Passivo"):
    cfg = config_for_profile(perfil)
    for seed in (1, 2):
        t0 = time.time()
        env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
        env.reset(seed=seed)
        m = DQN("MlpPolicy", env, learning_rate=5e-5, batch_size=64,
                verbose=0, seed=seed)
        m.learn(total_timesteps=550_000)
        s = evaluate_agent(m, lambda: ActionRepeatWrapper(
            ClassroomACEnv(config=cfg), repeat=2),
            build_scenario_matrix(), cfg, seed=0)["summary"]
        linhas.append({"perfil": perfil, "seed": seed,
                       "conf_larga": s["comfort_wide_pct"],
                       "conf_estreita": s["comfort_narrow_pct"],
                       "desvio": s["abs_dev_from_ideal"],
                       "energia": s["energy_kwh_day"]})
        print(f"  {perfil} seed {seed} ({(time.time()-t0)/60:.1f} min): "
              f"estreita {s['comfort_narrow_pct']:.1f}%", flush=True)
        pd.DataFrame(linhas).to_csv(SAIDA, index=False)
print(f"\ngravado: {SAIDA}")
