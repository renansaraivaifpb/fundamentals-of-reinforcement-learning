# -*- coding: utf-8 -*-
"""B = 0 e k = 1,0 somam? Os dois isolados melhoraram; a questão é se compõem."""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from stable_baselines3 import DQN
from hvac.config import config_for_profile
from hvac.env import ClassroomACEnv
from hvac.metrics import evaluate_agent
from hvac.scenarios import build_scenario_matrix
from hvac.wrappers import ActionRepeatWrapper

CFGS = {"referência (B=10, k=0,6)": {},
        "derivados (B=0, k=1,0)": {"comfort_bonus": 0.0, "comfort_sensitivity": 1.0}}
linhas = []
for rot, over in CFGS.items():
    cfg = config_for_profile("Equilibrado", **over)
    for seed in (0, 1, 2):
        t0 = time.time()
        env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2); env.reset(seed=seed)
        m = DQN("MlpPolicy", env, learning_rate=5e-5, batch_size=64, verbose=0, seed=seed)
        m.learn(total_timesteps=300_000)
        s = evaluate_agent(m, lambda: ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2),
                           build_scenario_matrix(), cfg, seed=0)["summary"]
        linhas.append({"config": rot, "seed": seed, "conf_larga": s["comfort_wide_pct"],
                       "conf_estreita": s["comfort_narrow_pct"],
                       "desvio": s["abs_dev_from_ideal"], "energia": s["energy_kwh_day"]})
        print(f"  {rot:<26} seed {seed} ({(time.time()-t0)/60:.1f} min): "
              f"estreita {s['comfort_narrow_pct']:.1f}%", flush=True)
        pd.DataFrame(linhas).to_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "resultados_combinacao.csv"), index=False)
