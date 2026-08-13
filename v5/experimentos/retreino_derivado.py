# -*- coding: utf-8 -*-
"""
Retreino completo sob a configuração DERIVADA (B = 0, k = 1,0).

POR QUE OS MODELOS ORIGINAIS SÃO PRESERVADOS. As seções que auditam o manuscrito
precisam dos agentes treinados com os parâmetros inferidos, pois é isso que
reproduz o trabalho publicado. Os modelos aqui gerados respondem a outra
pergunta, complementar: na melhor configuração conhecida, o agente supera o
controlador clássico? Substituir uns pelos outros apagaria a primeira pergunta.

APENAS B E K MUDAM. Deliberadamente não se aplicam aqui as melhorias de
treinamento da v5 (normalização de recompensa, seleção do melhor modelo), que
alterariam simultaneamente várias variáveis e impediriam atribuir o efeito aos
parâmetros da recompensa. Hiperparâmetros, arquitetura, número de passos e
sementes são idênticos aos do protocolo original.

    python experimentos/retreino_derivado.py
"""
from __future__ import annotations

import json, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from dataclasses import asdict

from stable_baselines3 import DQN, SAC

from hvac import features
from hvac.config import REWARD_PROFILES, config_for_profile
from hvac.env import ClassroomACEnv
from hvac.wrappers import ActionRepeatWrapper, ContinuousActionWrapper

PASSOS, SEEDS = 550_000, (0, 1, 2)
DERIVADOS = {"comfort_bonus": 0.0, "comfort_sensitivity": 1.0}
DESTINO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "models_derivados")
os.makedirs(DESTINO, exist_ok=True)


def salva_meta(caminho, perfil, seed, algo, cfg, model, repeat):
    campos = ("learning_rate", "batch_size", "gamma", "tau", "buffer_size",
              "learning_starts", "target_update_interval", "exploration_fraction",
              "exploration_initial_eps", "exploration_final_eps", "max_grad_norm")
    efet = {}
    for k in campos:
        if hasattr(model, k):
            v = getattr(model, k)
            if callable(v):
                try:
                    v = float(v(1.0))
                except Exception:
                    v = str(v)
            efet[k] = v
    payload = {
        "algo": algo, "profile": perfil, "seed": seed, "timesteps": PASSOS,
        "action_repeat": repeat, "obs_schema": features.schema(cfg),
        "parametros_derivados": DERIVADOS,
        "effective_hyperparams": efet,
        "env_params": {k: v for k, v in asdict(cfg).items()
                       if not isinstance(v, dict)},
        "ac_physics": {"capacity_btu_per_hour": cfg.physics.capacity_btu_per_hour,
                       "load_fraction": {s.name: f for s, f
                                         in cfg.physics.load_fraction.items()},
                       "cop": {s.name: c for s, c in cfg.physics.cop.items()}},
        "reward_profile": REWARD_PROFILES.get(perfil, {}),
        "tariff": cfg.tariff.name,
    }
    with open(caminho, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)


def main():
    total = len(SEEDS) * 3 + 1
    print(f"{total} treinos de {PASSOS:,} passos sob B=0, k=1,0\n", flush=True)
    feito = 0
    for perfil in ("Agressivo", "Equilibrado", "Passivo"):
        cfg = config_for_profile(perfil, **DERIVADOS)
        for seed in SEEDS:
            t0 = time.time()
            env = ActionRepeatWrapper(ClassroomACEnv(config=cfg), repeat=2)
            env.reset(seed=seed)
            m = DQN("MlpPolicy", env, learning_rate=5e-5, batch_size=64,
                    verbose=0, seed=seed)
            m.learn(total_timesteps=PASSOS)
            base = os.path.join(DESTINO, f"DQN_{perfil}_seed{seed}")
            m.save(base)
            salva_meta(base + "_config.json", perfil, seed, "DQN", cfg, m, 2)
            feito += 1
            print(f"  [{feito}/{total}] DQN {perfil} seed {seed}: "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

    # SAC: comparação discreto x contínuo, mesma configuração derivada
    cfg = config_for_profile("Equilibrado", **DERIVADOS)
    t0 = time.time()
    env = ActionRepeatWrapper(ContinuousActionWrapper(ClassroomACEnv(config=cfg)),
                              repeat=2)
    env.reset(seed=0)
    m = SAC("MlpPolicy", env, learning_rate=3e-4, batch_size=256, verbose=0, seed=0)
    m.learn(total_timesteps=PASSOS)
    base = os.path.join(DESTINO, "SAC_Equilibrado_seed0")
    m.save(base)
    salva_meta(base + "_config.json", "Equilibrado", 0, "SAC", cfg, m, 2)
    print(f"  [{total}/{total}] SAC Equilibrado seed 0: {(time.time()-t0)/60:.1f} min",
          flush=True)
    print(f"\nmodelos em {DESTINO}")


if __name__ == "__main__":
    main()
