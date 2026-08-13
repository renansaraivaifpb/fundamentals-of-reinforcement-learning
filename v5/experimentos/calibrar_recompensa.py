# -*- coding: utf-8 -*-
"""
Os quatro parâmetros inferidos podem ser calculados?

B, k, rho e a penalidade de frio nao sao publicados pelo manuscrito auditado e
foram inferidos da figura de recompensa. Este experimento separa o que se decide
por ARGUMENTO do que exige MEDICAO.

B, teoricamente irrelevante. O bonus base aparece nos tres ramos da eq. 4, e o
episodio tem duracao fixa. Somar uma constante a cada passo soma o mesmo valor
ao retorno de qualquer politica, logo nao altera a ordenacao entre elas. A
previsao e que B nao mude o desempenho; o que ele pode mudar e a ESCALA dos
valores de acao, e portanto o condicionamento numerico do aprendizado.

k, incoerente por analise. Ir do centro a borda da faixa custa 4,00; sair 2 C
alem dela custa 2,40. A penalidade marginal cai de 2,00/C para 0,12/C ao cruzar
a fronteira. Testam-se valores que restauram a coerencia: k = 1 iguala o custo de
sair 2 C ao de atravessar a meia-faixa; k = 4 iguala esse custo ja a 1 C.

rho, indeterminado analiticamente. O ganho de comutar nao e a recompensa
imediata, e sim o valor da trajetoria que a comutacao abre, o que depende da
politica. Resta varrer.

    python experimentos/calibrar_recompensa.py
"""
from __future__ import annotations

import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, torch
from stable_baselines3 import DQN

from hvac.config import config_for_profile
from hvac.env import ClassroomACEnv
from hvac.metrics import evaluate_agent, is_controllable, run_episode
from hvac.scenarios import build_scenario_matrix
from hvac.stats_analysis import dwell_times
from hvac.wrappers import ActionRepeatWrapper

PASSOS, SEEDS = 300_000, (0, 1)
SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "resultados_calibracao.csv")

VARIACOES = (
    [("B", {"comfort_bonus": b}, f"B = {b}") for b in (0.0, 10.0)] +
    [("k", {"comfort_sensitivity": k}, f"k = {k}") for k in (0.6, 1.0, 4.0)] +
    [("rho", {"short_cycle_penalty": r}, f"rho = {r}") for r in (-5.0, -20.0)] +
    [("frio", {"cold_action_penalty": c}, f"frio = {c}") for c in (-2.0, -10.0)]
)


def fab(cfg):
    return lambda: ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)


def main():
    cen = build_scenario_matrix()
    linhas = []
    for eixo, over, rot in VARIACOES:
        cfg = config_for_profile("Equilibrado", **over)
        for seed in SEEDS:
            t0 = time.time()
            env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
            env.reset(seed=seed)
            m = DQN("MlpPolicy", env, learning_rate=5e-5, batch_size=64,
                    verbose=0, seed=seed)
            m.learn(total_timesteps=PASSOS)

            s = evaluate_agent(m, fab(cfg), cen, cfg, seed=0)["summary"]
            dfs = [run_episode(m, fab(cfg)(), c, seed=0)
                   for c in cen if is_controllable(fab(cfg)(), c, seed=0)]
            iv = np.concatenate([dwell_times(d, cfg) for d in dfs if len(d)])

            # escala dos valores de acao, para o argumento sobre B
            obs = np.array([o for d in dfs for o in [None]][:0] or
                           [env.observation_space.sample() for _ in range(200)],
                           dtype=np.float32)
            with torch.no_grad():
                q = m.q_net(torch.as_tensor(obs)).numpy()

            linhas.append({
                "eixo": eixo, "variante": rot, "seed": seed,
                "conf_larga": s["comfort_wide_pct"],
                "conf_estreita": s["comfort_narrow_pct"],
                "desvio": s["abs_dev_from_ideal"],
                "energia": s["energy_kwh_day"],
                "trocas_h": s["changes_per_hour"],
                "violacoes_dmin_pct": float((iv < cfg.min_dwell_minutes).mean()*100),
                "mediana_dwell_min": float(np.median(iv)),
                "q_absoluto": float(np.abs(q).mean()),
                "q_faixa_entre_acoes": float((q.max(1) - q.min(1)).mean()),
            })
            print(f"  {rot:<14} seed {seed} ({(time.time()-t0)/60:.1f} min): "
                  f"conf {s['comfort_wide_pct']:.1f}% | estreita "
                  f"{s['comfort_narrow_pct']:.1f}% | viol {linhas[-1]['violacoes_dmin_pct']:.0f}%",
                  flush=True)
            pd.DataFrame(linhas).to_csv(SAIDA, index=False)
    print(f"\ngravado: {SAIDA}")


if __name__ == "__main__":
    main()
