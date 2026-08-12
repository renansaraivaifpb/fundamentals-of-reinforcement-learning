# -*- coding: utf-8 -*-
"""
Inicialização de política guiada por especialista (Xu et al. 2025, Sci. Reports).

O paper original treina 550k passos por perfil a partir de política aleatória.
Xu et al. mostram até 8,8× de speedup destilando conhecimento de especialista
— e neste projeto o especialista **já existe**: o termostato e o PI de
`baselines.py`.

Duas alavancas independentes, ambas implementadas aqui:

1. `pretrain_from_expert` — clonagem de comportamento sobre a rede Q do DQN.
   Sutileza importante: o DQN não tem cabeça de política, ele age por argmax de
   Q. Então não basta clonar ações; é preciso que a AÇÃO DO ESPECIALISTA tenha o
   maior valor Q. Treinamos com entropia cruzada sobre os Q-values tratados como
   logits — o que molda o argmax sem fixar a escala de Q, deixando o TD-learning
   posterior livre para corrigir os valores absolutos.

2. `collect_expert_transitions` — pré-carrega o replay buffer com transições do
   especialista, para que os primeiros gradientes de TD já vejam dados de boa
   qualidade em vez de exploração aleatória.

Ambas preservam a garantia do DQN: nada aqui altera o objetivo de Bellman, só o
ponto de partida. A política final continua sendo aprendida por RL.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def rollout_expert(
    expert,
    env,
    n_steps: int,
    seed: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Coleta (observações, ações) do especialista sobre o ambiente de TREINO
    (resets aleatórios), não sobre os cenários de avaliação — clonar no conjunto
    de teste seria vazamento.
    """
    obs_buf: List[np.ndarray] = []
    act_buf: List[int] = []
    transitions: List[Dict] = []

    if hasattr(expert, "reset"):
        expert.reset()
    obs, info = env.reset(seed=seed)

    for _ in range(n_steps):
        action, _ = expert.predict(obs, deterministic=True, info=info)
        action = int(np.asarray(action).reshape(-1)[0])
        obs_buf.append(np.asarray(obs, dtype=np.float32).copy())
        act_buf.append(action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        transitions.append(
            {
                "obs": np.asarray(obs, dtype=np.float32).copy(),
                "next_obs": np.asarray(next_obs, dtype=np.float32).copy(),
                "action": action,
                "reward": float(reward),
                "done": bool(terminated or truncated),
            }
        )
        obs = next_obs
        if terminated or truncated:
            if hasattr(expert, "reset"):
                expert.reset()
            obs, info = env.reset()

    return (
        np.asarray(obs_buf, dtype=np.float32),
        np.asarray(act_buf, dtype=np.int64),
        transitions,
    )


def pretrain_from_expert(
    model,
    expert,
    env,
    n_steps: int = 20_000,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: Optional[int] = 0,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Clona o especialista na rede Q do DQN por entropia cruzada sobre Q-values.

    Devolve métricas de concordância para que o warm start seja auditável — um
    pré-treino que "roda" mas não concorda com o especialista é pior que nenhum,
    porque dá falsa confiança.
    """
    obs, actions, _ = rollout_expert(expert, env, n_steps, seed=seed)
    device = model.device

    obs_t = torch.as_tensor(obs, device=device)
    act_t = torch.as_tensor(actions, device=device)

    q_net = model.policy.q_net
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

    n = len(obs_t)
    history: List[float] = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            q_values = q_net(obs_t[idx])
            # Q-values como logits: molda o argmax sem fixar a escala de Q.
            loss = F.cross_entropy(q_values, act_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(idx)
        history.append(total / n)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    BC época {epoch+1:3d}/{epochs}  loss={history[-1]:.4f}")

    with torch.no_grad():
        pred = q_net(obs_t).argmax(dim=1)
        agreement = float((pred == act_t).float().mean().item())

    # Sincroniza a target network: deixá-la aleatória enquanto a q_net já é
    # competente cria um alvo de TD inconsistente e desfaz o warm start nos
    # primeiros gradientes.
    model.policy.q_net_target.load_state_dict(q_net.state_dict())

    if verbose:
        print(f"    concordância com o especialista: {agreement*100:.1f}%")
    return {
        "bc_final_loss": history[-1] if history else float("nan"),
        "expert_agreement": agreement,
        "bc_samples": int(n),
    }


def seed_replay_buffer(model, transitions: List[Dict], verbose: bool = True) -> int:
    """
    Pré-carrega o replay buffer com transições do especialista.

    Complementa a clonagem: o BC molda o argmax, mas o buffer garante que os
    primeiros passos de TD-learning aprendam de trajetórias boas em vez de
    exploração aleatória.
    """
    added = 0
    for tr in transitions:
        if model.replay_buffer.full:
            break
        model.replay_buffer.add(
            obs=tr["obs"].reshape(1, -1),
            next_obs=tr["next_obs"].reshape(1, -1),
            action=np.array([[tr["action"]]]),
            reward=np.array([tr["reward"]]),
            done=np.array([tr["done"]]),
            infos=[{}],
        )
        added += 1
    if verbose:
        print(f"    replay buffer semeado com {added} transições do especialista")
    return added
