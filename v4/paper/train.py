# -*- coding: utf-8 -*-
"""
Treinamento dos três perfis DQN e do controlador contínuo SAC (Seção 4.4/4.5).

Protocolo do paper: MlpPolicy, lr = 5e-5, batch = 64, action repeat = 2,
550k timesteps, mesmo orçamento para o SAC.

Uso:
    python train.py                        # 3 perfis DQN, semente 0
    python train.py --seeds 0 1 2          # robustez sobre 3 sementes (Seção 5)
    python train.py --sac                  # inclui o SAC
    python train.py --timesteps 20000      # smoke test rápido
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Dict, Optional

from stable_baselines3 import A2C, DQN, DDPG, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor

from config import (LAB_PROFILES, REWARD_PROFILES, config_for_lab,
                    config_for_lab2, config_for_profile)
from env import ClassroomACEnv
from wrappers import ActionRepeatWrapper, ContinuousActionWrapper, MinDwellWrapper

PAPER_TIMESTEPS = 550_000
DQN_HYPERPARAMS = {"learning_rate": 5e-5, "batch_size": 64}
ACTION_REPEAT = 2

# Revisor 1: "implementa apenas 2 agentes simples de RL (Deep Q e Soft Actor
# Crítico); considerando a fácil disponibilidade de agentes de RL, seria mais
# valioso comparar diferentes abordagens (PPO, por exemplo)."
# Espaço de ação: DQN/PPO/A2C são discretos, SAC é contínuo.
ALGOS = {
    # Discretos
    "DQN": (DQN, DQN_HYPERPARAMS, False),
    "PPO": (PPO, {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64}, False),
    "A2C": (A2C, {"learning_rate": 7e-4, "n_steps": 5}, False),
    # Contínuos
    "SAC": (SAC, {"learning_rate": 3e-4, "batch_size": 256}, True),
    # TD3 é a adição para controle de PRECISÃO. Três motivos, todos medidos
    # nesta base: (i) política DETERMINÍSTICA — sem ruído estocástico, melhor
    # para rastreamento fino de setpoint; (ii) critics gêmeos contra
    # superestimação, relevante porque a cauda linear da Huber é grande;
    # (iii) suavização do alvo, que ataca a variância entre sementes observada
    # no DQN (97,8% / 96,4% / 5,3% na mesma configuração).
    "TD3": (TD3, {"learning_rate": 3e-4, "batch_size": 256,
                  "policy_delay": 2, "target_policy_noise": 0.2}, True),
    "DDPG": (DDPG, {"learning_rate": 3e-4, "batch_size": 256}, True),
}

# TQC (sb3-contrib) é estado da arte em controle contínuo e entra automaticamente
# se o pacote estiver instalado: `pip install sb3-contrib`.
try:
    from sb3_contrib import TQC  # noqa: F401
    ALGOS["TQC"] = (TQC, {"learning_rate": 3e-4, "batch_size": 256}, True)
except ImportError:
    pass


def make_env(profile: str, continuous: bool = False, seed: int | None = None,
             hard_dwell: bool = False, lab: bool = False, lab2: bool = False,
             legacy_obs: bool = False):
    if lab2:
        # Ablação da observação: `legacy_obs` desliga as features PID (erro
        # escalado e integral), mantendo todo o resto igual. É a única forma de
        # isolar o efeito da OBSERVAÇÃO do efeito de tudo mais.
        # A ação contínua bidirecional é NATIVA do ambiente (config.continuous_action);
        # o ContinuousActionWrapper existe apenas para o modo de reprodução do
        # manuscrito, onde o ambiente base é discreto e só resfria.
        extra = ({"observe_scaled_error": False, "observe_integral": False}
                 if legacy_obs else {})
        cfg = config_for_lab2(profile, continuous_action=continuous, **extra)
    else:
        cfg = config_for_lab(profile) if lab else config_for_profile(profile)
    env = ClassroomACEnv(config=cfg)
    if continuous and not lab2:
        env = ContinuousActionWrapper(env)
    if hard_dwell:
        # Restrição dura de permanência mínima. Medições mostram que a
        # penalidade mole da eq. 5 é superada pelo conforto em 71–83 % das
        # comutações; treinar sob a restrição faz o agente aprender uma política
        # que já a respeita, em vez de ser corrigido só na inferência.
        env = MinDwellWrapper(env)
    env = ActionRepeatWrapper(env, repeat=ACTION_REPEAT)
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env, cfg


def _effective_hyperparams(model) -> Dict:
    """
    Extrai TODOS os hiperparâmetros efetivos do modelo, inclusive os default do
    SB3 que nunca foram escritos no código.

    Revisor 2: "Não são informados todos os valores de B, k, ρ, penalidade de
    frio, ruído, duração do replay buffer, fator de desconto, frequência de
    atualização da target network, arquitetura da rede e estratégia de
    exploração." Listar só o que foi passado explicitamente esconde os defaults,
    que são igualmente necessários para reproduzir.
    """
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
        if callable(v):  # schedules
            try:
                v = {"schedule_at_1.0": float(v(1.0)), "schedule_at_0.0": float(v(0.0))}
            except Exception:
                v = str(v)
        elif hasattr(v, "__dict__") and not isinstance(v, (int, float, str, bool)):
            v = str(v)
        out[k] = v

    # Arquitetura efetiva da rede.
    try:
        out["policy_class"] = type(model.policy).__name__
        out["network_architecture"] = str(model.policy)
        out["n_parameters"] = sum(
            p.numel() for p in model.policy.parameters() if p.requires_grad
        )
    except Exception:
        pass
    return out


def save_metadata(path: str, profile: str, seed: int, timesteps: int,
                  algo: str, cfg, model=None, extra: Optional[Dict] = None) -> None:
    """
    Persiste a configuração COMPLETA junto do modelo. Sem isto, um .zip é
    irreprodutível: a avaliação precisa reconstruir a mesma física, os mesmos
    pesos de recompensa e o mesmo action repeat.
    """
    payload = {
        "algo": algo,
        "profile": profile,
        "seed": seed,
        "timesteps": timesteps,
        "action_repeat": ACTION_REPEAT,
        "declared_hyperparams": ALGOS[algo][1],
        "effective_hyperparams": _effective_hyperparams(model) if model else {},
        # asdict() serializa a config inteira, incluindo B, k, rho, penalidade de
        # frio e ruído — os parâmetros que o Revisor 2 aponta como ausentes.
        "env_params": {
            k: v for k, v in asdict(cfg).items() if not isinstance(v, dict)
        },
        "ac_physics": {
            "capacity_btu_per_hour": cfg.physics.capacity_btu_per_hour,
            "load_fraction": {s.name: f for s, f in cfg.physics.load_fraction.items()},
            "cop": {s.name: c for s, c in cfg.physics.cop.items()},
            "electrical_kw": {
                s.name: round(cfg.physics.electrical_kw(s), 4)
                for s in cfg.physics.load_fraction
            },
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
              logs_dir: str, algo: str = "DQN", expert_init: bool = False,
              hard_dwell: bool = False, lab: bool = False,
              lab2: bool = False, legacy_obs: bool = False) -> str:
    if algo not in ALGOS:
        raise KeyError(f"algoritmo '{algo}' desconhecido. Use: {list(ALGOS)}")
    cls, hyper, continuous = ALGOS[algo]

    suffix = (("_expert" if expert_init else "") + ("_dwell" if hard_dwell else "")
              + ("_lab2" if lab2 else "") + ("_obs7" if legacy_obs else "_obs9" if lab2 else ""))
    tag = f"{algo}_{profile}_seed{seed}{suffix}"
    print(f"\n{'='*70}\n{tag}  ({timesteps:,} passos)\n{'='*70}")
    t0 = time.time()

    env, cfg = make_env(profile, continuous=continuous, seed=seed,
                        hard_dwell=hard_dwell, lab=lab, lab2=lab2,
                        legacy_obs=legacy_obs)
    model = cls(
        policy="MlpPolicy", env=env, verbose=0, seed=seed,
        tensorboard_log=os.path.join(logs_dir, tag), **hyper,
    )

    warm: Dict = {}
    if expert_init:
        if algo != "DQN":
            raise ValueError("inicialização por especialista implementada só para DQN")
        # Xu et al. (2025): destilar o especialista antes do RL. O especialista
        # aqui é o PI sintonizado — já existe, custa zero passos de ambiente.
        from baselines import PIController
        from expert_init import pretrain_from_expert, rollout_expert, seed_replay_buffer

        print("  warm start a partir do PI (especialista)...")
        expert_env, _ = make_env(profile, seed=seed, hard_dwell=hard_dwell)
        warm = pretrain_from_expert(model, PIController(cfg), expert_env, seed=seed)
        _, _, transitions = rollout_expert(
            PIController(cfg), make_env(profile, seed=seed)[0], 5_000, seed=seed
        )
        seed_replay_buffer(model, transitions)

    model.learn(total_timesteps=timesteps, log_interval=None)

    model_path = os.path.join(models_dir, tag)
    model.save(model_path)
    save_metadata(f"{model_path}_config.json", profile, seed, timesteps, algo, cfg,
                  model=model, extra={"expert_init": expert_init, "hard_dwell": hard_dwell,
                         "warm_start": warm})

    print(f"-> concluído em {(time.time()-t0)/60:.1f} min | {model_path}.zip")
    return model_path


def main() -> None:
    p = argparse.ArgumentParser(description="Treino dos perfis do paper.")
    p.add_argument("--timesteps", type=int, default=PAPER_TIMESTEPS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--profiles", type=str, nargs="+", default=list(REWARD_PROFILES))
    p.add_argument("--algos", type=str, nargs="+", default=["DQN"],
                   help="algoritmos: DQN PPO A2C SAC")
    p.add_argument("--sac", action="store_true", help="atalho para incluir SAC")
    p.add_argument("--expert-init", action="store_true",
                   help="warm start a partir do PI (Xu et al. 2025)")
    p.add_argument("--hard-dwell", action="store_true",
                   help="permanência mínima como restrição dura")
    p.add_argument("--lab", action="store_true",
                   help="perfis de LABORATÓRIO de precisão com tarifa real da Enel")
    p.add_argument("--legacy-obs", action="store_true",
                   help="ablação: desliga erro escalado e integral na observação")
    p.add_argument("--lab2", action="store_true",
                   help="laboratório bidirecional: aquecimento + dT/dt + horizonte tarifário")
    p.add_argument("--models_dir", type=str, default="models_paper")
    p.add_argument("--logs_dir", type=str, default="logs_paper")
    args = p.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    algos = list(args.algos)
    if args.sac and "SAC" not in algos:
        algos.append("SAC")

    total = sum(
        len(args.seeds) * (1 if a == "SAC" else len(args.profiles)) for a in algos
    )
    print(f"{total} treinos de {args.timesteps:,} passos. algoritmos={algos}")

    for seed in args.seeds:
        for algo in algos:
            # SAC é um único agente (não há perfis no caso contínuo); o paper
            # adota o Equilibrado como recompensa de referência.
            continuo = ALGOS[algo][2]
            profiles = ((["Lab_Equilibrado"] if (args.lab or args.lab2) else ["Equilibrado"])
                        if (continuo and not args.lab2) else args.profiles)
            for profile in profiles:
                train_one(
                    profile, seed, args.timesteps, args.models_dir, args.logs_dir,
                    algo=algo,
                    expert_init=args.expert_init and algo == "DQN",
                    hard_dwell=args.hard_dwell, lab=args.lab, lab2=args.lab2,
                    legacy_obs=args.legacy_obs,
                )

    print("\nTreinamento concluído.")


if __name__ == "__main__":
    main()
