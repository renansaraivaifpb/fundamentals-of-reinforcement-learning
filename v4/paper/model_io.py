# -*- coding: utf-8 -*-
"""
Carregamento de modelos com reconstrução fiel da configuração de treino.

Motivação: um `.zip` do SB3 é irreprodutível sozinho. A avaliação precisa
reconstruir o MESMO ambiente — mesma física, mesmos pesos de recompensa, mesmo
espaço de observação e mesmo horizonte de decisão. Usar o default atual do módulo
`config` funciona até alguém mudar o default, e então quebra em silêncio ou com
erro obscuro.

Este projeto já sofreu essa falha duas vezes:

1. O `analyzer_v4.1.py` original avaliava sem o `ActionRepeatWrapper` usado no
   treino, medindo uma política com horizonte de decisão diferente. Erro
   silencioso — os números saíam, só estavam errados.
2. Ao adicionar erro escalado e integral à observação, os modelos de 7-D
   passaram a ser incarregáveis no ambiente de 9-D. Erro ruidoso, mas só
   descoberto por acaso.

A regra que este módulo impõe: **a configuração de treino é parte do contrato do
modelo**, e vem do metadado, nunca do default vigente.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import fields, replace
from typing import Dict, Optional, Tuple


def numpy_shim() -> None:
    """
    Permite ler modelos serializados com numpy 2.x sob numpy 1.x instalado.
    numpy 2.x renomeou `numpy.core` -> `numpy._core`; o pickle guarda o caminho.
    """
    import numpy.core  # noqa: F401
    for sub in ("", ".numeric", ".multiarray", ".umath", "._multiarray_umath"):
        try:
            __import__("numpy.core" + sub)
            sys.modules["numpy._core" + sub] = sys.modules["numpy.core" + sub]
        except Exception:
            pass


def read_metadata(model_path: str) -> Dict:
    """Metadado gravado ao lado do modelo, ou {} se ausente."""
    meta_path = model_path.replace(".zip", "_config.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, encoding="utf-8") as fh:
        return json.load(fh)


def config_from_metadata(meta: Dict):
    """
    Reconstrói a `ClassroomConfig` exata do treino a partir do metadado.

    Filtra apenas campos que existem na dataclass atual: se o projeto ganhou
    campos novos desde o treino, eles assumem o default — o que é correto, porque
    o modelo antigo foi treinado sem eles. Campos removidos são ignorados em vez
    de estourar.
    """
    from config import ClassroomConfig

    validos = {f.name for f in fields(ClassroomConfig)}
    params = {k: v for k, v in (meta.get("env_params") or {}).items() if k in validos}

    cfg = replace(ClassroomConfig(), **params)

    # A física do equipamento não é serializada como dataclass; reconstrói do
    # bloco `ac_physics` quando presente.
    fis = meta.get("ac_physics") or {}
    if fis:
        from ac_physics import ACPhysicsModel, ACState

        campos = {}
        if "capacity_btu_per_hour" in fis:
            campos["capacity_btu_per_hour"] = fis["capacity_btu_per_hour"]
        if fis.get("load_fraction"):
            campos["load_fraction"] = {
                ACState[k]: v for k, v in fis["load_fraction"].items()
            }
        if fis.get("cop"):
            campos["cop"] = {ACState[k]: v for k, v in fis["cop"].items()}
        campos["heating_enabled"] = bool(cfg.heating_enabled)
        cfg.physics = replace(ACPhysicsModel(), **campos)

    return cfg


def load_agent(model_path: str, device: str = "cpu") -> Tuple[object, object, Dict]:
    """
    Carrega (modelo, config_de_treino, metadado).

    Levanta erro claro se o metadado estiver ausente — avaliar sem saber a
    configuração de treino é pior que falhar, porque produz números plausíveis e
    errados.
    """
    numpy_shim()
    meta = read_metadata(model_path)
    if not meta:
        raise FileNotFoundError(
            f"metadado ausente para {model_path}. Sem ele não é possível "
            "reconstruir o ambiente de treino, e avaliar no ambiente errado "
            "produz números plausíveis mas inválidos."
        )

    algo = meta.get("algo", "DQN")
    from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

    classes = {"DQN": DQN, "PPO": PPO, "A2C": A2C,
               "SAC": SAC, "TD3": TD3, "DDPG": DDPG}
    try:
        from sb3_contrib import TQC
        classes["TQC"] = TQC
    except ImportError:
        pass

    if algo not in classes:
        raise KeyError(f"algoritmo '{algo}' desconhecido em {model_path}")

    model = classes[algo].load(model_path, device=device)
    cfg = config_from_metadata(meta)

    # Verificação do contrato: o espaço de observação do modelo tem de casar com
    # o do ambiente reconstruído. Se não casar, a config do metadado está
    # incompleta e é melhor falhar aqui que reportar métricas inválidas.
    from env import ClassroomACEnv

    env = ClassroomACEnv(config=cfg)
    esperado = model.observation_space.shape
    obtido = env.observation_space.shape
    if esperado != obtido:
        raise ValueError(
            f"contrato violado em {os.path.basename(model_path)}: modelo espera "
            f"obs {esperado}, ambiente reconstruído dá {obtido}. O metadado não "
            "descreve a configuração de treino por completo."
        )

    return model, cfg, meta


def make_env_factory(cfg, meta: Dict, extra_wrappers=()):
    """
    Fábrica de ambientes que respeita o contrato do modelo, incluindo o
    `action_repeat` — cuja omissão foi o bug original do `analyzer_v4.1.py`.
    """
    from env import ClassroomACEnv
    from wrappers import ActionRepeatWrapper

    repeat = int(meta.get("action_repeat", 1))

    def factory():
        env = ClassroomACEnv(config=cfg)
        for w in extra_wrappers:
            env = w(env)
        return ActionRepeatWrapper(env, repeat=repeat)

    return factory
