# -*- coding: utf-8 -*-
"""
Ablação da recompensa — responde à crítica central do Revisor 2.

    "A conclusão de que a modelagem da recompensa é mais decisiva do que o
     algoritmo também não é demonstrada pelos experimentos. Para sustentá-la,
     seria necessário realizar uma ablação da recompensa, comparando a
     formulação proposta com uma recompensa convencional, sem gradiente interno,
     sem penalidade de troca e sem penalidade de short-cycling."

O manuscrito AFIRMA que "a modelagem da recompensa, mais que o algoritmo, é o
fator determinante" mas nunca remove um termo para medir seu efeito. Comparar
DQN×SAC não resolve: são espaços de ação diferentes, então a comparação confunde
algoritmo com atuação.

Este módulo isola cada termo. A variante `sem_gradiente` é a mais importante:
é a hipótese explícita do paper (platô plano => estacionar na borda) e a causa
raiz do bug de 26,1 °C da v4 original. Se a ablação não mostrar degradação, a
contribuição alegada não existe.

    python ablation.py --timesteps 550000 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from typing import Dict, List

import pandas as pd

from config import ClassroomConfig, config_for_profile

# Cada variante desliga UM mecanismo, mantendo todo o resto fixo. `comfort_type`
# é lido pelo ambiente para escolher a topologia da recompensa de conforto.
ABLATIONS: Dict[str, Dict] = {
    "completa": {},
    "sem_gradiente": {"comfort_gradient": 0.0},
    "sem_penal_troca": {"action_change_penalty": 0.0},
    "sem_short_cycling": {"short_cycle_penalty": 0.0},
    "sem_penal_frio": {"cold_action_penalty": 0.0},
    "quadratica_pura": {"comfort_type": "quadratic"},
    "degraus_legado": {"comfort_type": "step"},
    "convencional": {
        # Recompensa "convencional" da literatura: quadrática sobre o erro, sem
        # nenhum dos mecanismos propostos. É o adversário honesto da tese.
        "comfort_type": "quadratic",
        "comfort_gradient": 0.0,
        "action_change_penalty": 0.0,
        "short_cycle_penalty": 0.0,
        "cold_action_penalty": 0.0,
    },
}


def config_for_ablation(variant: str, profile: str = "Equilibrado") -> ClassroomConfig:
    if variant not in ABLATIONS:
        raise KeyError(f"variante '{variant}' inexistente. Use: {list(ABLATIONS)}")
    return replace(config_for_profile(profile), **ABLATIONS[variant])


def run_ablation(
    variants: List[str],
    timesteps: int,
    seeds: List[int],
    models_dir: str = "models_ablation",
    logs_dir: str = "logs_ablation",
    profile: str = "Equilibrado",
) -> pd.DataFrame:
    """Treina e avalia cada variante, em cada semente."""
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor

    from env import ClassroomACEnv
    from metrics import evaluate_agent
    from scenarios import build_scenario_matrix
    from train import ACTION_REPEAT, DQN_HYPERPARAMS
    from wrappers import ActionRepeatWrapper

    os.makedirs(models_dir, exist_ok=True)
    scenarios = build_scenario_matrix()
    rows: List[Dict] = []

    for variant in variants:
        cfg = config_for_ablation(variant, profile)
        for seed in seeds:
            tag = f"ablation_{variant}_seed{seed}"
            path = os.path.join(models_dir, tag)

            if os.path.exists(f"{path}.zip"):
                print(f"  [cache] {tag}")
                model = DQN.load(f"{path}.zip", device="cpu")
            else:
                print(f"  treinando {tag} ({timesteps:,} passos)...")
                env = Monitor(
                    ActionRepeatWrapper(
                        ClassroomACEnv(config=cfg), repeat=ACTION_REPEAT
                    )
                )
                model = DQN(
                    "MlpPolicy", env, verbose=0, seed=seed,
                    tensorboard_log=os.path.join(logs_dir, tag),
                    **DQN_HYPERPARAMS,
                )
                model.learn(total_timesteps=timesteps, log_interval=None)
                model.save(path)

            factory = lambda c=cfg: ActionRepeatWrapper(
                ClassroomACEnv(config=c), repeat=ACTION_REPEAT
            )
            res = evaluate_agent(model, factory, scenarios, cfg, seed=seed)
            s = res["summary"]

            # Diagnóstico direto da hipótese do paper: a política estaciona
            # logo acima do teto de conforto?
            per = res["per_scenario"]
            parked = float(
                ((per["mean_temp"] > cfg.temp_comfort_max)
                 & (per["mean_temp"] < cfg.temp_comfort_max + 1.0)).mean() * 100.0
            )

            rows.append({
                "variante": variant,
                "seed": seed,
                "conf_larga_pct": s["comfort_wide_pct"],
                "conf_estreita_pct": s["comfort_narrow_pct"],
                "desvio_ideal": s["abs_dev_from_ideal"],
                "sobreaq_pct": s["overheat_pct"],
                "violacao_pct": s["violation_rate_pct"],
                "energia_kwh": s["energy_kwh_day"],
                "custo_brl": s["cost_brl_day"],
                "trocas_h": s["changes_per_hour"],
                "cenarios_estacionados_pct": parked,
            })
            print(f"    -> conf {s['comfort_wide_pct']:.1f}% | "
                  f"|T-24| {s['abs_dev_from_ideal']:.2f} | "
                  f"estacionado {parked:.0f}%")

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega por variante com média ± desvio entre sementes."""
    metrics = ["conf_larga_pct", "conf_estreita_pct", "desvio_ideal",
               "sobreaq_pct", "trocas_h", "energia_kwh",
               "cenarios_estacionados_pct"]
    agg = df.groupby("variante")[metrics].agg(["mean", "std"]).round(2)
    return agg.reindex([v for v in ABLATIONS if v in agg.index])


def main() -> None:
    p = argparse.ArgumentParser(description="Ablação da recompensa.")
    p.add_argument("--timesteps", type=int, default=550_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--variants", type=str, nargs="+", default=list(ABLATIONS))
    p.add_argument("--profile", type=str, default="Equilibrado")
    p.add_argument("--out", type=str, default="results_ablation")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = run_ablation(args.variants, args.timesteps, args.seeds, profile=args.profile)

    df.to_csv(os.path.join(args.out, "ablation_raw.csv"), index=False)
    summary = summarize(df)
    summary.to_csv(os.path.join(args.out, "ablation_summary.csv"))

    print("\n" + "=" * 100)
    print("ABLAÇÃO DA RECOMPENSA — média ± desvio entre sementes")
    print("=" * 100)
    print(summary.to_string())

    if "completa" in df["variante"].values:
        base = df[df["variante"] == "completa"]["conf_larga_pct"].mean()
        print(f"\nDegradação em conforto vs. recompensa completa ({base:.1f}%):")
        for v in df["variante"].unique():
            if v == "completa":
                continue
            d = df[df["variante"] == v]["conf_larga_pct"].mean() - base
            print(f"  {v:<22} {d:+6.1f} pp")

    with open(os.path.join(args.out, "ablations.json"), "w", encoding="utf-8") as fh:
        json.dump(ABLATIONS, fh, ensure_ascii=False, indent=2)
    print(f"\nresultados em {args.out}/")


if __name__ == "__main__":
    main()
