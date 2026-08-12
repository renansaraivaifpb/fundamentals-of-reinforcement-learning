# -*- coding: utf-8 -*-
"""
Treinamento — v5.

TRÊS MUDANÇAS QUE AFETAM O DESEMPENHO DO AGENTE, não só a organização do código.
Todas motivadas por evidência medida na v4:

1. SELEÇÃO DO MELHOR MODELO, não do último (`EvalCallback`).
   A v4 salvava o modelo ao fim de `learn()`. O comentário do próprio `ALGOS` da
   v4 registra a variância entre sementes do DQN na MESMA configuração:
   97,8 % / 96,4 % / 5,3 %. Uma semente colapsou. Salvar o estado final significa
   que, se o colapso ocorrer perto do fim, ele é o que vai para o disco — mesmo
   que o agente tenha passado a maior parte do treino numa política boa. Avaliar
   periodicamente e guardar o melhor é a correção de maior retorno por linha de
   código deste arquivo.

2. NORMALIZAÇÃO DA RECOMPENSA (`VecNormalize(norm_reward=True)`).
   A observação já é normalizada por construção (ver features.py), então
   normalizá-la agrega pouco. A RECOMPENSA, não: o conforto vale até +10, mas a
   topologia 'band' aplica parede de inclinação 40 por grau fora da faixa, o que
   a 15 °C de desvio chega a centenas de unidades negativas. O próprio código da
   v4 documenta o sintoma no conforto Huber — "faixa dinâmica de 204x, que faz os
   alvos de TD explodirem e o DQN não converge". Normalizar por média móvel do
   retorno ataca a causa em vez de reescrever a recompensa.

3. WARM START COM EXPLORAÇÃO REDUZIDA.
   A v4 implementou inicialização por especialista e obteve 80,1 % de
   concordância com o PI após a clonagem — e então o resultado degradava para
   37,2 % de conforto. O diagnóstico já estava escrito no REVISAO.md: "a
   exploração ε-greedy default do DQN destrói a política clonada antes que o
   TD-learning a recupere. Aproveitar o warm start exige também reduzir a
   exploração inicial, o que não foi ajustado." Aqui é ajustado: ao usar warm
   start, `exploration_initial_eps` cai para 0,15 e `learning_starts` é zerado,
   já que o replay buffer chega semeado. O fio solto era explícito; fica atado.

Uso:
    python -m hvac.train --lab2 --algos TD3 --seeds 0 1 2
    python -m hvac.train --ablate-group obs_pid --algos TD3 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from . import features
from .config import (ABLATION_GROUPS, LAB_PROFILES, REWARD_PROFILES,
                     config_ablacao, config_for_lab, config_for_lab2,
                     config_for_profile)
from .env import ClassroomACEnv
from .wrappers import ActionRepeatWrapper, ContinuousActionWrapper, MinDwellWrapper

PAPER_TIMESTEPS = 550_000
ACTION_REPEAT = 2

ALGOS = {
    "DQN":  (DQN,  {"learning_rate": 5e-5, "batch_size": 64}, False),
    "PPO":  (PPO,  {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64}, False),
    "A2C":  (A2C,  {"learning_rate": 7e-4, "n_steps": 5}, False),
    "SAC":  (SAC,  {"learning_rate": 3e-4, "batch_size": 256}, True),
    "TD3":  (TD3,  {"learning_rate": 3e-4, "batch_size": 256,
                    "policy_delay": 2, "target_policy_noise": 0.2}, True),
    "DDPG": (DDPG, {"learning_rate": 3e-4, "batch_size": 256}, True),
}
try:
    from sb3_contrib import TQC
    ALGOS["TQC"] = (TQC, {"learning_rate": 3e-4, "batch_size": 256}, True)
except ImportError:
    pass

# Exploração inicial sob warm start. 1,0 (o default do DQN) destrói em poucos
# milhares de passos a política clonada do especialista.
WARM_START_EPS = 0.15


def build_config(profile: str, lab: bool = False, lab2: bool = False,
                 ablate_group: Optional[str] = None, **overrides):
    if ablate_group:
        return config_ablacao(profile, ablate_group, **overrides)
    if lab2:
        return config_for_lab2(profile, **overrides)
    if lab:
        return config_for_lab(profile, **overrides)
    return config_for_profile(profile, **overrides)


def make_env(cfg, seed: Optional[int] = None, hard_dwell: bool = False,
             continuous_wrapper: bool = False):
    env = ClassroomACEnv(config=cfg)
    if continuous_wrapper:
        env = ContinuousActionWrapper(env)
    if hard_dwell:
        env = MinDwellWrapper(env)
    env = ActionRepeatWrapper(env, repeat=ACTION_REPEAT)
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def _vec(cfg, seed, hard_dwell, continuous_wrapper, normalize: bool,
         treino: bool = True):
    """Ambiente vetorizado, opcionalmente com normalização de recompensa."""
    venv = DummyVecEnv([lambda: make_env(cfg, seed, hard_dwell, continuous_wrapper)])
    if not normalize:
        return venv
    # `norm_obs=False` de propósito: os canais já saem normalizados de
    # features.py, e renormalizar por estatística móvel destruiria o significado
    # fixo de cada um (o zero do erro escalado deixaria de ser o setpoint).
    return VecNormalize(venv, norm_obs=False, norm_reward=treino,
                        clip_reward=10.0, gamma=0.99)


def _effective_hyperparams(model) -> Dict:
    keys = ("learning_rate", "batch_size", "gamma", "tau", "buffer_size",
            "learning_starts", "train_freq", "gradient_steps",
            "target_update_interval", "exploration_fraction",
            "exploration_initial_eps", "exploration_final_eps",
            "max_grad_norm", "n_steps", "ent_coef", "vf_coef", "gae_lambda",
            "n_epochs", "clip_range")
    out: Dict = {}
    for k in keys:
        if not hasattr(model, k):
            continue
        v = getattr(model, k)
        if callable(v):
            try:
                v = {"schedule_at_1.0": float(v(1.0)), "schedule_at_0.0": float(v(0.0))}
            except Exception:
                v = str(v)
        elif hasattr(v, "__dict__") and not isinstance(v, (int, float, str, bool)):
            v = str(v)
        out[k] = v
    try:
        out["policy_class"] = type(model.policy).__name__
        out["n_parameters"] = sum(p.numel() for p in model.policy.parameters()
                                  if p.requires_grad)
    except Exception:
        pass
    return out


def save_metadata(path: str, profile: str, seed: int, timesteps: int, algo: str,
                  cfg, model=None, extra: Optional[Dict] = None) -> None:
    """
    Persiste a configuração completa E o contrato da observação.

    `obs_schema` é a adição da v5: sem os nomes ordenados dos canais, uma troca
    de ordem entre treino e avaliação é indetectável — ver
    `model_io.assert_schema_compatible`.
    """
    payload = {
        "algo": algo, "profile": profile, "seed": seed, "timesteps": timesteps,
        "action_repeat": ACTION_REPEAT,
        "obs_schema": features.schema(cfg),
        "declared_hyperparams": ALGOS[algo][1],
        "effective_hyperparams": _effective_hyperparams(model) if model else {},
        "env_params": {k: v for k, v in asdict(cfg).items()
                       if not isinstance(v, dict)},
        "ac_physics": {
            "capacity_btu_per_hour": cfg.physics.capacity_btu_per_hour,
            "load_fraction": {s.name: f for s, f in cfg.physics.load_fraction.items()},
            "cop": {s.name: c for s, c in cfg.physics.cop.items()},
        },
        "reward_profile": (LAB_PROFILES if profile in LAB_PROFILES
                           else REWARD_PROFILES).get(profile, {}),
        "tariff": cfg.tariff.name,
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)


def train_one(profile: str, seed: int, timesteps: int, models_dir: str,
              logs_dir: str, algo: str = "DQN", lab: bool = False,
              lab2: bool = False, ablate_group: Optional[str] = None,
              hard_dwell: bool = False, expert_init: bool = False,
              normalize: bool = True, eval_freq: int = 10_000) -> str:
    if algo not in ALGOS:
        raise KeyError(f"algoritmo '{algo}' desconhecido. Use: {list(ALGOS)}")
    cls, hyper, continuous = ALGOS[algo]
    hyper = dict(hyper)

    cfg = build_config(profile, lab=lab, lab2=lab2, ablate_group=ablate_group,
                       **({"continuous_action": True} if continuous and lab2 else {}))
    # O wrapper de ação contínua só existe para o modo de reprodução do
    # manuscrito, onde o ambiente base é discreto; no lab2 a ação é nativa.
    continuous_wrapper = continuous and not cfg.continuous_action

    sufixo = (("_dwell" if hard_dwell else "") + ("_expert" if expert_init else "")
              + (f"_abl-{ablate_group}" if ablate_group else "")
              + ("_lab2" if lab2 else "_lab" if lab else ""))
    tag = f"{algo}_{profile}_seed{seed}{sufixo}"
    os.makedirs(models_dir, exist_ok=True)
    print(f"\n{'='*70}\n{tag}  ({timesteps:,} passos)\n{'='*70}")
    print(f"  observação: {features.schema(cfg)['n']} canais "
          f"-> {features.schema(cfg)['nomes']}")
    t0 = time.time()

    env = _vec(cfg, seed, hard_dwell, continuous_wrapper, normalize)

    warm: Dict = {}
    if expert_init:
        if algo != "DQN":
            raise ValueError("inicialização por especialista implementada só para DQN")
        # A correção do fio solto da v4: sem reduzir a exploração, a política
        # clonada é destruída antes de o TD-learning aproveitá-la.
        hyper["exploration_initial_eps"] = WARM_START_EPS
        hyper["exploration_fraction"] = 0.05
        hyper["learning_starts"] = 0      # o buffer já chega semeado
        warm["exploration_initial_eps"] = WARM_START_EPS

    model = cls(policy="MlpPolicy", env=env, verbose=0, seed=seed,
                tensorboard_log=os.path.join(logs_dir, tag), **hyper)

    if expert_init:
        from .baselines import PIController
        from .expert_init import (pretrain_from_expert, rollout_expert,
                                  seed_replay_buffer)
        print(f"  warm start a partir do PI (eps inicial = {WARM_START_EPS})...")
        warm.update(pretrain_from_expert(
            model, PIController(cfg), make_env(cfg, seed, hard_dwell), seed=seed))
        _, _, transitions = rollout_expert(
            PIController(cfg), make_env(cfg, seed, hard_dwell), 5_000, seed=seed)
        seed_replay_buffer(model, transitions)

    # --- seleção do melhor modelo -----------------------------------------
    # Ambiente de avaliação SEPARADO e com sementes distintas das de treino.
    # `norm_reward=False` na avaliação: normalizar ali mediria o retorno numa
    # escala móvel, e o "melhor" passaria a depender de quando foi medido.
    # Não há estatística de observação a sincronizar entre os dois ambientes:
    # `norm_obs=False` por decisão de projeto (os canais já saem normalizados de
    # features.py com significado fixo).
    eval_env = _vec(cfg, seed + 500, hard_dwell, continuous_wrapper,
                    normalize, treino=False)
    melhor_dir = os.path.join(models_dir, f"{tag}_best")
    callback = EvalCallback(
        eval_env, best_model_save_path=melhor_dir, log_path=melhor_dir,
        eval_freq=max(eval_freq, 1), n_eval_episodes=5,
        deterministic=True, verbose=0,
    )

    model.learn(total_timesteps=timesteps, log_interval=None, callback=callback)

    # Se a avaliação periódica encontrou um estado melhor que o final, é ele que
    # vai para o disco — é exatamente o caso da semente que colapsa no fim.
    melhor_zip = os.path.join(melhor_dir, "best_model.zip")
    usou_melhor = os.path.exists(melhor_zip)
    if usou_melhor:
        model = cls.load(melhor_zip, device="cpu")

    model_path = os.path.join(models_dir, tag)
    model.save(model_path)
    if normalize and isinstance(env, VecNormalize):
        env.save(f"{model_path}_vecnormalize.pkl")

    save_metadata(f"{model_path}_config.json", profile, seed, timesteps, algo, cfg,
                  model=model,
                  extra={"expert_init": expert_init, "hard_dwell": hard_dwell,
                         "ablate_group": ablate_group, "normalize_reward": normalize,
                         "best_model_selected": usou_melhor, "warm_start": warm})
    print(f"-> {(time.time()-t0)/60:.1f} min | melhor-modelo={usou_melhor} | "
          f"{model_path}.zip")
    return model_path


def main() -> None:
    p = argparse.ArgumentParser(description="Treino v5.")
    p.add_argument("--timesteps", type=int, default=PAPER_TIMESTEPS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--profiles", type=str, nargs="+", default=None)
    p.add_argument("--algos", type=str, nargs="+", default=["DQN"],
                   help=f"disponíveis: {list(ALGOS)}")
    p.add_argument("--lab", action="store_true")
    p.add_argument("--lab2", action="store_true")
    p.add_argument("--ablate-group", type=str, default=None,
                   help=f"desliga um grupo: {list(ABLATION_GROUPS)}")
    p.add_argument("--hard-dwell", action="store_true")
    p.add_argument("--expert-init", action="store_true")
    p.add_argument("--no-normalize", action="store_true",
                   help="desliga a normalização de recompensa")
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--models_dir", type=str, default="models_v5")
    p.add_argument("--logs_dir", type=str, default="logs_v5")
    a = p.parse_args()

    perfis = a.profiles or (list(LAB_PROFILES) if (a.lab or a.lab2)
                            else list(REWARD_PROFILES))
    total = len(perfis) * len(a.seeds) * len(a.algos)
    print(f"{total} treinos de {a.timesteps:,} passos. algoritmos={a.algos}")
    for seed in a.seeds:
        for algo in a.algos:
            for profile in perfis:
                train_one(profile, seed, a.timesteps, a.models_dir, a.logs_dir,
                          algo=algo, lab=a.lab, lab2=a.lab2,
                          ablate_group=a.ablate_group, hard_dwell=a.hard_dwell,
                          expert_init=a.expert_init,
                          normalize=not a.no_normalize, eval_freq=a.eval_freq)
    print("\nTreinamento concluído.")


if __name__ == "__main__":
    main()
